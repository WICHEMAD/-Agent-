# agent/agent_core.py
import logging
import asyncio
import re
import time
import traceback
import random
import json
import os
from typing import AsyncGenerator

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage

import config
from agent import tools
from agent import prompt
from agent.memory import SummaryMemory
from agent.rag import get_rag_engine
from agent.tools import _current_session, JobScraperTool

logger = logging.getLogger(__name__)


# ==================== 消息类型检测 ====================

# 检测求职者是否在询问市场信息或需要工具查询
_MARKET_QUERY_PATTERNS = [
    (r'薪资.*多少', '薪资'),
    (r'工资.*多少', '薪资'),
    (r'.*薪资.*行情', '薪资'),
    (r'多少.*钱.*月', '薪资'),
    (r'.*行情.*怎么样', '行情'),
    (r'有什么.*岗位', '岗位'),
    (r'有哪些.*岗位', '岗位'),
    (r'招聘.*情况', '岗位'),
    (r'有什么.*职位', '岗位'),
    (r'(?:深圳|北京|上海|杭州|广州).*薪资', '薪资'),
    (r'薪资.*(?:深圳|北京|上海|杭州|广州)', '薪资'),
    (r'(?:深圳|北京|上海|杭州|广州).*后端.*岗位', '岗位'),
    (r'(?:深圳|北京|上海|杭州|广州).*岗位', '岗位'),
    (r'Python.*后端.*(?:薪资|行情|岗位|招聘)', '岗位'),
    (r'(?:后端|前端|全栈).*(?:薪资|行情|多少钱)', '薪资'),
    (r'(?:C\+\+|Java|Go|Python).*(?:北京|上海|深圳|杭州|广州).*(?:岗位|职位|招聘)', '岗位'),
    (r'(?:北京|上海|深圳|杭州|广州).*(?:C\+\+|Java|Go|Python).*(?:岗位|职位|招聘)', '岗位'),
    (r'岗位.*(?:深圳|北京|上海|杭州|广州)', '岗位'),
    (r'职位.*(?:深圳|北京|上海|杭州|广州)', '岗位'),
]


def _is_market_query(message: str) -> str | None:
    """检测消息是否为市场行情查询。返回搜索关键词，或 None 表示不是行情查询。"""
    for pattern, _ in _MARKET_QUERY_PATTERNS:
        if re.search(pattern, message, re.IGNORECASE):
            # 从消息中提取搜索关键词
            return _extract_search_keywords(message)
    return None


# 检测求职者是否在询问面试元信息（岗位、流程等），而非回答技术题
_META_QUESTION_PATTERNS = [
    r'(?:面试的?|应聘的?|这个|什么).*岗位.*(?:是?什么|叫?什么)',
    r'(?:我|这个).*(?:面试|应聘).*(?:岗位|职位).*',
    r'岗位.*(?:要求|JD|职责|描述)',
    r'(?:这个|什么).*(?:技术栈|技术).*要求',
    r'(?:还要|继续).*(?:面试|问).*多久',
    r'(?:面试|还要).*(?:多久|几轮|什么.*时候|流程)',
    r'(?:我的?|简历).*(?:表现|怎么样|如何)',
    r'(?:为什么|怎么).*(?:这样|这么).*问',
    r'为什么.*(?:一直|老是|总).*问',
    r'JD.*(?:是|有).*什么',
    r'(?:这个|你).*是.*面试.*(?:什么|什么岗位|哪个)',
    r'换个话题',
    r'能不能.*(?:换个|问).*问题',
]


def _is_meta_question(message: str) -> bool:
    """检测消息是否为面试元信息询问（非技术回答）。"""
    for pattern in _META_QUESTION_PATTERNS:
        if re.search(pattern, message, re.IGNORECASE):
            return True
    return False


def _extract_search_keywords(message: str) -> str:
    """从用户消息中提取搜索关键词"""
    # 提取技术栈关键词
    tech = []
    for kw in ['Python', 'Java', 'Go', 'C++', '前端', '全栈', '后端',
               'Django', 'FastAPI', 'Flask', 'React', 'Vue', 'Spring']:
        if kw.lower() in message.lower():
            tech.append(kw)
    tech_str = ' '.join(tech) if tech else 'Python'

    # 提取城市
    for city in ['深圳', '北京', '上海', '杭州', '广州', '成都', '武汉']:
        if city in message:
            return f"{tech_str} 后端 {city}"

    return f"{tech_str} 后端"


# 爬虫单例（避免每次请求都初始化 Chrome driver — 实际每次仍会新建，但保持接口一致）
_scraper: JobScraperTool | None = None


def _get_scraper() -> JobScraperTool:
    global _scraper
    if _scraper is None:
        _scraper = JobScraperTool()
    return _scraper


def _run_scraper_for_message(message: str) -> str:
    """根据用户消息直接运行爬虫，返回真实数据"""
    keywords = _extract_search_keywords(message)
    try:
        scraper = _get_scraper()
        result = scraper._run(keywords)
        logger.info(f"[爬虫直接调用] 关键词='{keywords}', 结果长度={len(result)}")
        return result
    except Exception as e:
        logger.error(f"[爬虫直接调用失败] {e}")
        return f"搜索关键词：{keywords}\n\n抱歉，岗位信息查询暂时失败，请稍后再试。错误：{e}"


# ==================== 响应过滤 ====================

# 自导自演检测
_CANDIDATE_MARKER = re.compile(r'求职者[：:说]')
_INTERVIEWER_MARKER = re.compile(r'^\s*面试官[：:]\s*')

# 废话信号词（整行匹配，用于删除）
_JUNK_LINES = re.compile(
    r'^\s*('
    r'根据求职者.*|'
    r'我们可以继续深入.*|'
    r'为了进一步验证.*|'
    r'我将提出一个.*|'
    r'我将通过.*|'
    r'接下来.*提问.*|'
    r'这样设计的?问题.*|'
    r'既符合每次.*|'
    r'通过这样的提问.*|'
    r'这个问题.*既.*|'
    r'可以进一步考察.*|'
    r'确保.*专业.*|'
    r'通过深入探讨.*|'
    r'通过这样的.*提问.*|'
    r'为.*验证.*技术能力.*'
    r')$',
    re.MULTILINE,
)

# 问句提取：找到第一个以 ？ 结尾的完整句子
_QUESTION_SENTENCE = re.compile(r'[^。？\n]*？', re.DOTALL)


def _filter_agent_response(text: str) -> str:
    """过滤 Agent 响应

    策略：自导自演截断 → 提取第一个问句 → 清理废话行。
    工具类响应（无问号的多段落文本）直接透传，不截断。
    """
    if not text or not text.strip():
        return text

    original_len = len(text)

    # ==== 1. 自导自演：出现"求职者："立即截断 ====
    cm = _CANDIDATE_MARKER.search(text)
    if cm:
        text = text[:cm.start()].strip()
        text = _INTERVIEWER_MARKER.sub('', text).strip()
        logger.warning(
            f"[响应过滤] 检测到自导自演（编造求职者回答），"
            f"{original_len} → {len(text)} 字符"
        )
        original_len = len(text)

    # ==== 2. 去掉"面试官："前缀 ====
    text = _INTERVIEWER_MARKER.sub('', text).strip()

    # ==== 3. 删除废话行 ====
    text = _JUNK_LINES.sub('', text).strip()

    # ==== 4. 检测是否为工具类响应（无问号的多段落文本） ====
    questions = _QUESTION_SENTENCE.findall(text)
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    if not questions and len(paragraphs) >= 2:
        # 无问号 + 多段落 → 判定为工具返回的数据，直接透传
        logger.info(
            f"[响应过滤] 检测到工具类响应（无问号，{len(paragraphs)} 段落），透传不截断"
        )
        return text

    # ==== 4.5 检测工具类响应 + 末尾提问（如爬虫结果后加"你对哪个方向感兴趣？"） ====
    if len(questions) == 1 and len(paragraphs) >= 3:
        # 只有一个问号，但前面有多段数据 → 判定为工具数据+对话收尾，保留全文
        last_q = questions[-1]
        text_end = text[text.find(last_q) + len(last_q):].strip()
        # 确认问号在文本末尾附近（最后 20% 区域）
        q_position_ratio = text.find(last_q) / len(text)
        if q_position_ratio > 0.5 and not text_end:
            logger.info(
                f"[响应过滤] 工具类响应+末尾提问，{len(paragraphs)} 段，透传不截断"
            )
            return text

    # ==== 5. 如果文本包含多个 ？，只取第一个问句 ====
    if len(questions) >= 2:
        first_q = questions[0].strip()
        logger.info(f"[响应过滤] 检测到 {len(questions)} 个问句，提取第一个")
        return first_q
    elif len(questions) == 1:
        # 只有一个问句：返回问句及其前面的简短上下文
        q_pos = text.find(questions[0])
        start = text.rfind('。', 0, q_pos)
        if start > 0 and q_pos - start < 120:
            result = text[start + 1:q_pos + len(questions[0])].strip()
        else:
            result = text[max(0, q_pos - 80):q_pos + len(questions[0])].strip()
        if result:
            logger.info(f"[响应过滤] 提取唯一问句，{original_len} → {len(result)} 字符")
            return result

    # ==== 6. 无问句的单段落兜底 ====
    if paragraphs:
        text = paragraphs[0]
    else:
        text = text.strip()

    if len(text) != original_len:
        logger.info(f"[响应过滤] 最终清理，{original_len} → {len(text)} 字符")

    return text


# ==================== 重试工具函数 ====================

def _call_with_retry(func, *args, **kwargs):
    """带指数退避的重试包装器

    Args:
        func: 要执行的函数
        *args, **kwargs: 传递给 func 的参数
    Returns:
        func 的返回值
    Raises:
        超过最大重试次数后抛出最后一次异常
    """
    last_exc = None
    max_retries = config.Config.MAX_RETRIES
    base_delay = config.Config.RETRY_BASE_DELAY

    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))  # 1s, 2s, 4s...
                logger.warning(
                    f"[重试 {attempt}/{max_retries}] {type(e).__name__}: {e} "
                    f"| 等待 {delay:.1f}s 后重试"
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"[重试耗尽] {type(e).__name__}: {e}\n"
                    f"{traceback.format_exc()}"
                )
    raise last_exc


class InterviewerAgent:
    """面试官 Agent

    封装 LLM、工具、记忆的初始化与调用，对外提供简洁的流式/非流式对话接口。

    使用示例:
        agent = InterviewerAgent()
        async for token in agent.stream_chat("user_001", "你好"):
            print(token, end="")
    """

    def __init__(
            self,
            model_name: str = None,
            temperature: float = 0.7,
            system_prompt: str = None,
    ):
        model_name = model_name or config.Config.CHAT_MODEL
        system_prompt = system_prompt or prompt.SYSTEM_PROMPT
        provider = config.Config.MODEL_PROVIDER

        stop_sequences = ["求职者：", "求职者说：", "求职者回答"]

        if provider == "openai_compatible":
            # OpenAI 兼容 API（OpenAI / DeepSeek / 其他兼容服务）
            api_key = config.Config.MODEL_API_KEY
            base_url = config.Config.MODEL_BASE_URL
            if not base_url.endswith("/v1"):
                base_url = base_url.rstrip("/")
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                openai_api_key=api_key,
                openai_api_base=base_url,
                stop=stop_sequences,
            )
        else:
            # 默认：Ollama 本地模型
            self.llm = ChatOllama(
                model=model_name,
                temperature=temperature,
                base_url=getattr(config.Config, "OLLAMA_BASE_URL", "http://localhost:11434"),
                stop=stop_sequences,
            )
        self.memory = SummaryMemory(llm=self.llm)
        self.agent = create_agent(
            model=self.llm,
            tools=tools.ALL_TOOLS,
            checkpointer=self.memory.checkpointer,
            system_prompt=system_prompt,
        )
        # 降级回复模板（当所有重试都失败时使用）
        self._fallback_responses = [
            "抱歉，当前使用的模型暂不支持此功能。你可以尝试在设置中切换到能力更强的模型（如 deepseek-chat、gpt-4o 等），或换一种方式描述你的问题。",
            "不好意思，当前模型无法完成这个操作 🙈 这可能是因为模型不支持工具调用，建议切换到支持 function calling 的模型后再试。",
            "抱歉，系统暂时无法处理你的请求。如果问题持续出现，可能是当前模型不支持工具调用，请在设置中切换模型。",
        ]

    # ==================== 流式对话 ====================

    def _build_resume_context(self, session_id: str) -> str | None:
        """预取简历内容，构建注入对话上下文的文本

        在 Agent 执行前程序化检索简历，不再依赖 LLM 主动调用工具。
        返回 None 表示该会话没有上传简历。
        """
        try:
            engine = get_rag_engine()
            docs = engine.retrieve(
                "求职者 技能 工作经历 项目经验 教育背景 姓名 联系方式",
                k=8,
                filter={"session_id": session_id},
            )
            if not docs:
                return None

            parts = [
                "[系统上下文] 以下是求职者的简历信息，"
                "请严格基于这些内容进行面试提问，不要编造简历中不存在的信息：\n"
            ]
            for i, doc in enumerate(docs, 1):
                content = doc.page_content.strip()
                if content:
                    parts.append(f"【简历片段{i}】{content}")
            return "\n\n".join(parts)
        except Exception as e:
            logger.warning(f"[预取简历] 检索失败: {e}")
            return None

    def _build_jd_context(self, session_id: str) -> str | None:
        """从 sessions.json 读取岗位 JD，构建注入对话上下文的文本

        返回 None 表示该会话没有上传岗位信息。
        """
        try:
            sessions_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "sessions.json"
            )
            with open(sessions_file, "r", encoding="utf-8") as f:
                sessions = json.load(f)
            for s in sessions:
                if s["id"] == session_id:
                    jd = s.get("jd")
                    if not jd or not jd.get("jd_text", "").strip():
                        return None
                    title = jd.get("job_title", "").strip()
                    text = jd["jd_text"].strip()
                    parts = [
                        "[系统上下文] 以下是求职者应聘的岗位信息（JD），"
                        "请结合岗位要求与简历内容有针对性地提问。\n"
                    ]
                    if title:
                        parts.append(f"【岗位名称】{title}")
                    parts.append(f"【岗位要求】\n{text}")
                    return "\n".join(parts)
            return None
        except Exception as e:
            logger.warning(f"[预取JD] 读取失败: {e}")
            return None

    def stream_chat(self, user_id: str, user_message: str):
        cfg = self.memory.get_config(user_id)

        if self.memory.needs_summary(user_id):
            try:
                logger.info(f"用户 {user_id} 对话过长，触发自动摘要")
                self.memory.summarize_history(user_id)
            except Exception as e:
                logger.warning(f"自动摘要失败，跳过摘要: {e}")

        max_retries = config.Config.MAX_RETRIES
        base_delay = config.Config.RETRY_BASE_DELAY
        last_exc = None

        for attempt in range(1, max_retries + 1):
            try:
                stream_started = False

                # 设置当前会话 ID，供 RAGResumeTool 按 session 过滤
                _current_session.set(user_id)

                history = self.memory.get_history(user_id)
                if not history:
                    # 首次对话：注入简历 + JD 上下文
                    resume_context = self._build_resume_context(user_id)
                    jd_context = self._build_jd_context(user_id)

                    if resume_context and jd_context:
                        # 既有简历又有岗位 — 对齐两者进行精准提问
                        user_message = (
                            "[系统指令] 你只代表面试官，只输出面试官说的话，"
                            "绝对不要编造求职者的回答。"
                            "基于简历与岗位要求的匹配度，提出第一个面试问题，"
                            "重点考察简历中与岗位要求相关的技术能力和项目经验。\n\n"
                            + resume_context
                            + "\n\n" + ("=" * 30) + "\n\n"
                            + jd_context
                            + "\n\n---\n求职者说：" + user_message
                        )
                    elif resume_context and not jd_context:
                        # 仅有简历
                        user_message = (
                            "[系统指令] 你只代表面试官，只输出面试官说的话，"
                            "绝对不要编造求职者的回答。"
                            "基于以下简历信息，向求职者提出第一个面试问题。\n\n"
                            + resume_context
                            + "\n\n---\n求职者说：" + user_message
                        )
                    elif jd_context and not resume_context:
                        # 仅有岗位 JD — 先了解背景，再对照 JD 提问
                        user_message = (
                            "[系统指令] 你只代表面试官，只输出面试官说的话，"
                            "绝对不要编造求职者的回答。"
                            "以下是求职者应聘的岗位信息，"
                            "由于尚未上传简历，请先通过对话了解求职者背景，"
                            "再结合岗位要求展开技术提问。\n\n"
                            + jd_context
                            + "\n\n---\n求职者说：" + user_message
                        )
                    else:
                        # 什么都没有
                        user_message = (
                            "[系统指令] 该求职者尚未上传简历，"
                            "请通过对话主动了解其技能、经验、项目背景。"
                            "你只代表面试官，只输出面试官说的话，"
                            "绝对不要编造求职者的回答。\n\n"
                            "求职者说：" + user_message
                        )
                else:
                    # 非首次对话：面试追问模式，但允许调用工具
                    jd_context = self._build_jd_context(user_id)

                    if _is_meta_question(user_message):
                        # 元问题：求职者在问岗位/流程等非技术问题，直接回答
                        user_message = (
                            "[本轮指令] 你是面试官，求职者在询问面试相关信息。"
                            "请直接回答他的问题，语气友好、信息准确。不要追问技术问题。\n\n"
                            + (jd_context + "\n\n" if jd_context else "")
                            + "求职者说：" + user_message
                        )
                    else:
                        # 技术回答或闲聊：追问模式
                        user_message = (
                            "[本轮指令] 你是面试官，只输出面试官说的话，不要编造求职者的回答。\n"
                            "- 如果求职者在回答面试题，简短评估后追问一个问题。\n"
                            "- 如果求职者询问岗位行情、薪资数据、招聘信息，立即调用 job_info_scraper 工具。\n"
                            "- 如果求职者要做薪资计算，调用 simple_calculator 工具。\n\n"
                            + (jd_context + "\n\n" if jd_context else "")
                            + "求职者说：" + user_message
                        )
                # 缓冲所有 token，用于响应过滤
                buffer: list[str] = []
                for chunk in self.agent.stream(
                        {"messages": [HumanMessage(content=user_message)]},
                        config=cfg,
                        stream_mode="messages",
                ):
                    if isinstance(chunk, tuple):
                        msg, _ = chunk
                        if hasattr(msg, "content") and msg.content:
                            stream_started = True
                            buffer.append(msg.content)

                if stream_started:
                    raw = "".join(buffer)
                    filtered = _filter_agent_response(raw)
                    if filtered != raw:
                        logger.info(
                            f"[响应过滤生效] 原始 {len(raw)} 字符 → 过滤后 {len(filtered)} 字符"
                        )
                    # 以合理大小的块输出，保持流式体验
                    chunk_size = 16
                    for i in range(0, len(filtered), chunk_size):
                        yield filtered[i:i + chunk_size]
                    return
                # 空回复也算异常，触发重试
                raise RuntimeError("Agent 流式返回为空，未生成任何内容")

            except Exception as e:
                last_exc = e
                if attempt < max_retries:
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        f"[流式重试 {attempt}/{max_retries}] 用户={user_id} "
                        f"{type(e).__name__}: {e} | 等待 {delay:.1f}s"
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"[流式重试耗尽] 用户={user_id} {type(e).__name__}: {e}\n"
                        f"{traceback.format_exc()}"
                    )

        # 所有重试失败 → 降级方案
        if config.Config.ENABLE_FALLBACK:
            fallback = random.choice(self._fallback_responses)
            logger.warning(f"[降级兜底] 用户={user_id}, 返回默认回复")
            yield fallback

    # ==================== 非流式对话 ====================

    def chat(self, user_id: str, user_message: str) -> str:
        cfg = self.memory.get_config(user_id)

        if self.memory.needs_summary(user_id):
            try:
                logger.info(f"用户 {user_id} 对话过长，触发自动摘要")
                self.memory.summarize_history(user_id)
            except Exception as e:
                logger.warning(f"自动摘要失败，跳过摘要: {e}")

        # 首次对话注入简历上下文；非首次注入评估指令
        history = self.memory.get_history(user_id)
        if not history:
            resume_context = self._build_resume_context(user_id)
            jd_context = self._build_jd_context(user_id)

            if resume_context and jd_context:
                user_message = (
                    "[系统指令] 你只代表面试官，只输出面试官说的话，"
                    "绝对不要编造求职者的回答。"
                    "基于简历与岗位要求的匹配度，提出面试问题，"
                    "重点考察简历中与岗位要求相关的技术能力和项目经验。\n\n"
                    + resume_context
                    + "\n\n" + ("=" * 30) + "\n\n"
                    + jd_context
                    + "\n\n---\n求职者说：" + user_message
                )
            elif resume_context and not jd_context:
                user_message = (
                    "[系统指令] 你只代表面试官，只输出面试官说的话，"
                    "绝对不要编造求职者的回答。基于简历信息提问。\n\n"
                    + resume_context
                    + "\n\n---\n求职者说：" + user_message
                )
            elif jd_context and not resume_context:
                user_message = (
                    "[系统指令] 你只代表面试官，只输出面试官说的话，"
                    "绝对不要编造求职者的回答。"
                    "以下是岗位信息，由于未上传简历，请先了解求职者背景，"
                    "再结合岗位要求展开技术提问。\n\n"
                    + jd_context
                    + "\n\n---\n求职者说：" + user_message
                )
            else:
                user_message = (
                    "[系统指令] 该求职者尚未上传简历，请通过对话了解其背景。"
                    "你只代表面试官，只输出面试官说的话，"
                    "绝对不要编造求职者的回答。\n\n"
                    "求职者说：" + user_message
                )
        else:
            # 非首次对话：面试追问模式，但允许调用工具
            jd_context = self._build_jd_context(user_id)

            if _is_meta_question(user_message):
                # 元问题：直接回答
                user_message = (
                    "[本轮指令] 你是面试官，求职者在询问面试相关信息。"
                    "请直接回答他的问题，语气友好、信息准确。不要追问技术问题。\n\n"
                    + (jd_context + "\n\n" if jd_context else "")
                    + "求职者说：" + user_message
                )
            else:
                # 技术回答或闲聊：追问模式
                user_message = (
                    "[本轮指令] 你是面试官，只输出面试官说的话，不要编造求职者的回答。\n"
                    "- 如果求职者在回答面试题，简短评估后追问一个问题。\n"
                    "- 如果求职者询问岗位行情、薪资数据、招聘信息，立即调用 job_info_scraper 工具。\n"
                    "- 如果求职者要做薪资计算，调用 simple_calculator 工具。\n\n"
                    + (jd_context + "\n\n" if jd_context else "")
                    + "求职者说：" + user_message
                )

        try:
            result = _call_with_retry(
                self.agent.invoke,
                {"messages": [HumanMessage(content=user_message)]},
                config=cfg,
            )
            messages = result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
                if content and content.strip():
                    # 应用响应过滤
                    filtered = _filter_agent_response(content)
                    if filtered != content:
                        logger.info(
                            f"[响应过滤生效] 原始 {len(content)} 字符 → 过滤后 {len(filtered)} 字符"
                        )
                    return filtered
            raise RuntimeError("Agent 返回为空")

        except Exception as e:
            logger.error(f"[非流式调用最终失败] 用户={user_id}: {e}")
            if config.Config.ENABLE_FALLBACK:
                return random.choice(self._fallback_responses)
            return "抱歉，我没有生成回复"

    # ==================== 同步流式封装 ====================

    def stream_chat_sync(self, user_id: str, user_message: str):
        result = ""
        for token in self.stream_chat(user_id, user_message):
            result += token
        return result

    # ==================== 对话管理 ====================

    def get_history(self, user_id: str):
        return self.memory.get_history(user_id)

    def reset_conversation(self, user_id: str):
        logger.info(f"重置用户 {user_id} 的对话历史")
        config = self.memory.get_config(user_id)
        self.agent.update_state(config, {"messages": []})


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    agent = InterviewerAgent()

    # 1. 非流式对话
    print("【1. chat() 非流式】")
    reply = agent.chat("test_001", "你好，我是来面试Python后端的，有3年经验")
    print(f"面试官: {reply[:100]}...")

    # 2. 同步流式
    print("\n【2. stream_chat_sync() 流式】")
    reply = agent.stream_chat_sync("test_001", "我熟悉Django和FastAPI")
    print(f"面试官: {reply[:100]}...")

    # 3. 查历史
    print(f"\n【3. 历史消息数】{len(agent.get_history('test_001'))}")

    # 4. 重置
    print("\n【4. 重置对话】")
    agent.reset_conversation("test_001")
    print(f"重置后消息数: {len(agent.get_history('test_001'))}")

    print("\n全部测试完成！")