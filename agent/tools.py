import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import urllib.parse
import logging
import functools
import time
import contextvars
from optparse import Option
from pydantic import Field
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from langchain_classic.schema import document
from langchain_core.tools import Tool, BaseTool
from sqlalchemy.engine import default
from sqlalchemy.ext.asyncio import result
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import config

logger = logging.getLogger(__name__)

# 当前会话 ID，由 agent_core 在每次对话前设置，供工具按 session 过滤
_current_session = contextvars.ContextVar('current_session', default='')


def tool_retry(tool_name: str):
    """工具内部重试装饰器：工具调用失败时自动重试，超过次数后降级"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            max_retries = config.Config.MAX_RETRIES
            base_delay = config.Config.RETRY_BASE_DELAY
            last_exc = None

            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** (attempt - 1))
                        logger.warning(
                            f"[{tool_name} 工具重试 {attempt}/{max_retries}] "
                            f"{type(e).__name__}: {e} | 等待 {delay:.1f}s"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"[{tool_name} 工具重试耗尽] "
                            f"{type(e).__name__}: {e}"
                        )
            raise last_exc
        return wrapper
    return decorator


#==========================计算器工具==========================
class CalculatorTool(BaseTool):
    name:str= "simple_calculator"
    description:str= (
            "当求职者问及薪资计算、涨薪比例、税后收入等数学问题时使用。"
            "输入应为一个数学表达式字符串，例如 '8000*1.1'、'10000*0.8'。"
            "返回计算结果。"
)

    def _run(self, query:str)->str:
        try:
            # ✅ 安全检查：只允许数字和基本运算符
            allowed_chars = set("0123456789.+-*/()%^ ")

            #检测非法的字符
            for char in query:
                if char not in allowed_chars:
                    return f"输入格式错误：包含非法字符 {char}"

            #限制表达式长度
            if len(query) > 100:
                return "输入格式错误：表达式长度超过100个字符"

            #执行计算
            result = eval(query, {"__builtins__": {}})
            if isinstance(result, float) and result==int(result):
                result=int(result)
            if isinstance(result, float):
                result=round(result,4)
            return f"计算结果：{result}"

        except SyntaxError:
            return f"错误：表达式语法错误：{query}"
        except ZeroDivisionError:
            return "错误：除数不能为零"
        except Exception as e:
            return f"计算失败：{e}"

    async def _arun(self,query:str):
        return self._run(query)


#=========================岗位爬虫工具==========================
class JobScraperTool(BaseTool):
    name:str= "job_info_scraper"
    description:str= (
        "当求职者需要了解真实岗位信息、薪资行情、岗位要求时使用。"
        "输入为搜索关键词，如 'Python 后端 深圳'。"
        "返回职位信息，包含职位名称、职位描述、职位链接、职位发布时间、职位薪资、职位城市、职位公司名称、职位公司规模、职位公司行业、职位公司地址、职位公司描述。"
    )

    @tool_retry("job_info_scraper")
    def _run_with_retry(self, keywords: str) -> str:
        """带重试的核心爬取逻辑，失败由装饰器捕获"""
        options = Options()
        # options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        options.binary_location = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        # 设置页面加载超时
        options.add_argument(f"--timeout={config.Config.TOOL_TIMEOUT * 1000}")
        driver = None

        try:
            driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=options,
            )
            driver.set_page_load_timeout(config.Config.TOOL_TIMEOUT)
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            encoded = urllib.parse.quote(keywords)  # URL编码，处理中文、空格、防止链接失效
            url = f"https://www.liepin.com/zhaopin/?key={encoded}"
            logger.info(f"[岗位爬虫] 开始爬取: {url}")
            driver.get(url)
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CLASS_NAME, "job-card-pc-container"))
            )
            time.sleep(1.8)
            cards = driver.find_elements(By.CLASS_NAME, "job-card-pc-container")[:5]
            if not cards:
                logger.warning("[岗位爬虫] 页面加载成功但未找到岗位卡片")
                return _mock_jobs(keywords)


            # 移除 display:none，让卡片可见
            driver.execute_script("""
                document.querySelectorAll('.job-card-pc-container').forEach(c => {
                    c.style.display = 'block';
                });
            """)
            time.sleep(0.5)

            lines = [f"搜索关键词：{keywords}", ""]
            for i, card in enumerate(cards, 1):
                title = _safe_extract(card, [
                    (By.CSS_SELECTOR, "[class*='job-title']"),
                    (By.CSS_SELECTOR, "a[href*='job'] [class*='ellipsis']"),
                    (By.TAG_NAME, "h3"),
                ], "未知")
                salary = _safe_extract(card, [
                    (By.CSS_SELECTOR, "[class*='E8PWS']"),
                    (By.CSS_SELECTOR, "span[class*='k']"),
                ], "薪资面议")
                company = _safe_extract(card, [
                    (By.CSS_SELECTOR, "[class*='K6Y1c']"),
                    (By.CSS_SELECTOR, "[class*='company']"),
                ], "未知公司")

                lines.append(f"{i}. {title} | {salary} | {company}")

            logger.info(f"[岗位爬虫] 成功爬取 {len(cards)} 条岗位信息")
            return "\n".join(lines)
        finally:
            if driver is not None:
                driver.quit()

    def _run(self, keywords: str) -> str:
        """对外入口：重试失败后降级返回模拟数据"""
        try:
            return self._run_with_retry(keywords)
        except Exception as e:
            logger.error(f"[岗位爬虫] 所有重试均失败，降级返回模拟数据: {e}")
            return _mock_jobs(keywords)

    async def _arun(self, keywords: str) -> str:
        return self._run(keywords)

# =========================RAG简历检索工具==========================
class RAGResumeTool(BaseTool):
    name: str = "resume_knowledge_base"
    description: str = (
        "当需要查询求职者简历中的具体信息时使用，例如："
        "求职者的技能、工作经历、项目经验、教育背景、联系方式等。"
        "输入应为自然语言查询，如 '求职者有哪些Python相关技能'、"
        "'求职者的上一份工作是什么'。"
        "返回简历中相关的文本片段。"
    )

    def _run(self, query: str) -> str:
        from agent.rag import get_rag_engine

        engine = get_rag_engine()
        session_id = _current_session.get()
        try:
            filter_dict = {"session_id": session_id} if session_id else None
            docs = engine.retrieve(query, filter=filter_dict)
            if not docs:
                return "简历知识库中未找到相关信息，可能是简历尚未上传或内容不相关。"

            lines = ["根据简历知识库检索到以下相关信息：", ""]
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "未知")
                content = doc.page_content.strip()[:300]
                lines.append(f"【片段{i}】（来源：{source}）")
                lines.append(content)
                lines.append("")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[RAG检索失败] {e}")
            return f"简历检索失败：{e}"

    async def _arun(self, query: str) -> str:
        return self._run(query)


def _safe_extract(element, selectors: list, fallback: str) -> str:
    """逐个尝试多个选择器，返回第一个匹配到的文本"""
    for by, value in selectors:
        try:
            el = element.find_element(by, value)
            text = el.get_attribute("textContent").strip()
            if text:
                return text
        except Exception:
            continue
    return fallback

def _mock_jobs(keywords: str) -> str:
    return (
        f"[模拟数据 - 搜索关键词: {keywords}]\n"
        "1. Python后端开发工程师 | 25k-40k | 某科技有限公司 | 深圳\n"
        "2. 高级Python开发 | 30k-50k | 某某互联网公司 | 北京\n"
        "3. Python数据分析师 | 20k-35k | 数据科技公司 | 上海\n"
        "4. Python全栈工程师 | 22k-38k | 创业科技公司 | 杭州\n"
        "5. Python测试开发 | 18k-30k | 某金融科技公司 | 深圳"
    )
# ==================== 工具实例（供 Agent 注册使用） ====================

ALL_TOOLS = [
    CalculatorTool(),
    JobScraperTool(),
    RAGResumeTool(),
]


if __name__=="__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    calc=CalculatorTool()
    print(calc._run("8000*1.1"))

    job_scraper=JobScraperTool()
    print(job_scraper._run("GO 后端 北京"))