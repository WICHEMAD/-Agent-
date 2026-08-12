# 面试官 Agent · 产品文档

> 本文档既是产品设计说明，也是 Claude 开发参考。产品需求和架构决策记录在此，后续迭代始终保持一致。

---

## 一、产品定位

### 一句话描述

**AI 模拟技术面试官** — 求职者上传简历，与 LLM 进行真实的模拟面试，同步支持岗位行情查询和薪资计算。

### 解决什么问题

| 痛点 | 方案 |
|---|---|
| 求职者面试前缺乏实战练习 | LLM 扮演面试官，基于真实简历追问 |
| 不知道自己技术深度够不够 | 追问策略设计：模糊回答→澄清，有深度→深挖细节 |
| 不了解市场行情 | 内嵌猎聘爬虫，实时查询岗位薪资 |
| 不会算薪资涨幅 | 内嵌安全计算器 |

### 目标用户

- 技术求职者（面试模拟 + 行情了解）
- 也可扩展为 HR 端（出题 + 评估候选人）

---

## 二、核心功能

### 2.1 模拟面试（主线）

**流程：** 创建会话 → 上传简历（可选）→ LLM 发起第一个问题 → 求职者回答 → 追问/切换话题 → 循环

**追问策略**（定义在 `agent/prompt.py`）：
- 回答模糊 → 指出并要求澄清
- 回答有深度 → 肯定后追问他提到的具体细节
- 回答有误 → 纠正误解，追问相关知识点
- 话题挖够 → 一句话收尾，切换新话题

**深度标准（任一达成即可切换话题）：**
- 求职者说出了技术决策及原因（"因为 YY 选择了 XX"）
- 提到了踩过的坑和解决方案
- 能对比不同方案的优劣

### 2.2 简历 RAG

上传 PDF/DOCX/TXT/MD → 分块(默认 500 字/块，重叠 50 字) → Ollama Embedding → Chroma 向量存储。面试时自动检索简历片段注入 LLM 上下文，无需 LLM 主动调用工具。

**约束：** 按 `session_id` 隔离，每个会话的简历互不可见。

### 2.3 岗位行情爬虫

Selenium 无头爬取猎聘网前 5 条岗位。当用户消息匹配到市场查询模式（如"深圳Python后端薪资多少"）时，LLM 调用 `job_info_scraper` 工具。

**降级策略：** 爬虫失败重试 3 次 → 返回模拟数据，标注 `[模拟数据]`。

### 2.4 薪资计算器

安全 eval，只允许数字和基本运算符 `+-*/()%^`，禁止 `__builtins__`。表达式最长 100 字符。

### 2.5 对话记忆

- LangGraph SQLite checkpoint 持久化
- 每个 session_id 独立 thread_id
- 消息数 > 20 条自动 LLM 摘要压缩（保留最近 4 条）
- 摘要写入 checkpointer，后续对话基于摘要 + 最近消息

---

## 三、对话架构

### 消息流

```
POST /chat/{session_id}  {"message": "用户输入"}
        │
        ▼
InterviewerAgent.stream_chat(session_id, message)
        │
        ├── 对话超限？→ LLM 摘要压缩
        ├── 设置 _current_session (供 RAGResumeTool 按 session 过滤)
        ├── 首次对话？
        │     ├── 有简历 → RAG 检索 → 注入简历上下文 + "基于简历提问"
        │     └── 无简历 → 注入 "了解求职者背景" 指令
        ├── 非首次？
        │     └── 注入 "面试追问模式 + 工具调用规则"
        │
        ▼
self.agent.stream(messages, config=cfg, stream_mode="messages")
        │
        ▼
_filter_agent_response(raw)
        ├── 1. 检测 "求职者：" → 截断（防自导自演）
        ├── 2. 去掉 "面试官：" 前缀
        ├── 3. 删除废话行（_JUNK_LINES）
        ├── 4. 工具类响应（无问号多段落）→ 透传
        ├── 5. 多问句 → 只取第一个
        │
        ▼
SSE 流式输出（16 字/块）→ 前端逐字渲染
```

### 关键设计决策

**为什么简历不依赖 LLM 调用工具，而是程序化预取？**
LLM 不一定会主动调用 `resume_knowledge_base` 工具，尤其小模型。程序化预取保证简历信息一定被注入上下文。

**为什么要响应过滤？**
LLM 倾向生成完整对话，会编造求职者回答（"求职者：我熟悉 Django..."）。`stop` 参数 + `_filter_agent_response` 双重拦截。

**为什么非首次对话要注入额外指令？**
如果不重复强调"只输出面试官的话"，LLM 在多轮对话后会逐渐忘记角色约束，开始自导自演。

---

## 四、代码地图

```
interviewer_agent/
│
├── config.py                    # 全局配置中心
│   ├── DEFAULT_MODEL_SETTINGS   #   默认模型设置
│   ├── _load_model_settings()   #   从 JSON 加载（失败用默认值）
│   ├── save_model_settings()    #   持久化
│   └── class Config             #   所有配置项，支持环境变量覆盖
│
├── agent/
│   ├── agent_core.py            # 核心：InterviewerAgent
│   │   ├── _is_market_query()   #   市场行情意图检测（正则）
│   │   ├── _filter_agent_response()  # 响应过滤（核心逻辑）
│   │   ├── _call_with_retry()   #   指数退避重试
│   │   ├── _build_resume_context()   # 简历 RAG 预取
│   │   ├── InterviewerAgent.stream_chat()   # 流式对话
│   │   └── InterviewerAgent.chat()         # 非流式对话
│   │
│   ├── prompt.py                # System Prompt
│   │   └── SYSTEM_PROMPT        #   面试官角色 + 追问策略 + 工具规则
│   │
│   ├── tools.py                 # 三个工具
│   │   ├── CalculatorTool       #   薪资计算
│   │   ├── JobScraperTool       #   猎聘爬虫
│   │   ├── RAGResumeTool        #   简历检索
│   │   ├── _safe_extract()      #   多选择器安全提取
│   │   ├── _mock_jobs()         #   爬虫降级模拟数据
│   │   └── ALL_TOOLS            #   工具注册列表
│   │
│   ├── memory.py                # 对话记忆
│   │   ├── SummaryMemory.get_config()        # 按 user_id 生成 LangGraph config
│   │   ├── SummaryMemory.get_history()       # 查询对话历史
│   │   ├── SummaryMemory.needs_summary()     # >20 条触发
│   │   └── SummaryMemory.summarize_history() # LLM 摘要压缩
│   │
│   └── rag.py                   # RAG 引擎
│       ├── RAGEngine._load_document()       # 支持 pdf/docx/txt/md
│       ├── RAGEngine.process_document()     # 加载→分块→向量化→存储
│       ├── RAGEngine.retrieve()             # 语义检索（支持 metadata 过滤）
│       ├── RAGEngine.delete_by_session()    # 按 session 删除向量
│       └── get_rag_engine()                 # 全局单例
│
├── api/
│   └── main.py                  # FastAPI 路由
│       ├── /sessions            #   会话 CRUD
│       ├── /chat/{id}           #   流式对话 (SSE)
│       ├── /chat/{id}/history   #   对话历史
│       ├── /chat/{id}/reset     #   重置对话
│       ├── /upload/{id}         #   简历上传/查询/删除
│       ├── /settings            #   模型设置读写
│       └── /                    #   重定向到 /static/index.html
│
├── static/
│   └── index.html               # 前端 SPA（原生 JS，无框架）
│       ├── 侧边栏                #   会话列表、新建/切换/删除
│       ├── 聊天区                #   消息渲染、流式接收、简历上传
│       └── 设置弹窗              #   模型提供商、名称、API Key、Base URL
│
└── data/                        # 运行时数据（Git 忽略）
    ├── sessions.json            #   会话元数据
    ├── model_settings.json      #   模型配置（含 API Key）
    ├── checkpoints.db           #   SQLite 对话检查点
    ├── uploads/                 #   用户上传简历
    └── vector_store/            #   Chroma 向量库
```

---

## 五、关键数据流

### 会话生命周期

```
创建会话 → POST /sessions → 生成 8 位 UUID → 写入 sessions.json
       ↓
选择会话 → GET /chat/{id}/history → 从 checkpoints.db 加载消息 → 渲染
       ↓
发送消息 → POST /chat/{id} 流式 → Agent 处理 → SSE 返回 → 更新 sessions.json
       ↓
删除会话 → DELETE /sessions/{id} → 清除 sessions.json 条目
```

### 简历生命周期

```
上传 → POST /upload/{id} → 保存文件到 uploads/ → RAG 处理 → 写入 Chroma
                                                      ↓
                                                  更新 sessions.json
查询 → GET /upload/{id} → 读取 sessions.json 中 resume 字段
删除 → DELETE /upload/{id} → 清除 Chroma 向量 → 清除 sessions.json
```

### 模型切换

```
前端弹窗 → POST /settings → save_model_settings(json)
                                ↓
                          config.reload_settings()  (更新 Config 类属性)
                                ↓
                          rebuild_agent()           (重建 InterviewerAgent 实例)
```

---

## 六、配置体系

### 配置来源优先级

```
环境变量 > data/model_settings.json > DEFAULT_MODEL_SETTINGS
```

### 所有配置项

| 配置项 | 默认值 | 作用域 |
|---|---|---|
| `MODEL_PROVIDER` | `ollama` | ollama / openai_compatible |
| `CHAT_MODEL` | `deepseek-r1:7b` | 对话模型名 |
| `MODEL_API_KEY` | — | OpenAI 兼容模式 API Key |
| `MODEL_BASE_URL` | `http://localhost:11434` | API 服务地址 |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 专用地址 |
| `EMBEDDING_MODEL` | `qwen3-embedding:0.6b` | RAG Embedding 模型 |
| `CHUNK_SIZE` | `500` | 文本分块字符数 |
| `CHUNK_OVERLAP` | `50` | 分块重叠字符数 |
| `RETRIEVAL_K` | `5` | RAG 检索返回数 |
| `MAX_RETRIES` | `3` | 重试次数 |
| `RETRY_BASE_DELAY` | `1.0` | 重试基础间隔（秒） |
| `TOOL_TIMEOUT` | `30` | 工具调用超时（秒） |
| `ENABLE_FALLBACK` | `True` | 失败后降级兜底 |

### 新增配置项检查清单

1. `DEFAULT_MODEL_SETTINGS` dict
2. `Config` 类属性 + 环境变量读取
3. `data/model_settings.json` 对应的默认值（首次启动自动生成）
4. `api/main.py` 的 `ModelSettingsRequest` 和 `/settings` 接口
5. `static/index.html` 设置弹窗表单项

---

## 七、开发约定

### 新增工具

```python
# 1. agent/tools.py — 创建工具类
class NewTool(BaseTool):
    name: str = "tool_name"
    description: str = "何时使用、输入格式、返回值"

    def _run(self, query: str) -> str: ...
    async def _arun(self, query: str) -> str: ...

# 2. agent/tools.py — 注册
ALL_TOOLS = [CalculatorTool(), JobScraperTool(), RAGResumeTool(), NewTool()]

# 3. agent/prompt.py — 告知 LLM
# 在 SYSTEM_PROMPT 的 "工具使用规则" 段添加说明

# 4. agent/agent_core.py — 如需要程序化触发
# 在 _is_market_query() 或 user_message 前缀中加入逻辑
```

### 新增 API

```python
# 1. api/main.py — 定义 Pydantic Model
class NewRequest(BaseModel): ...

# 2. api/main.py — 添加路由
@app.post("/new-endpoint")
async def new_endpoint(req: NewRequest): ...

# 3. static/index.html — 前端对接
# 在 <script> 中添加 fetch 调用和 UI 更新逻辑
```

### 调整面试策略

| 改什么 | 改哪里 |
|---|---|
| 面试官语气、追问规则 | `agent/prompt.py` → `SYSTEM_PROMPT` |
| 首次对话注入的上下文 | `agent/agent_core.py` → `_build_resume_context()` + 消息前缀 |
| 非首次对话注入的指令 | `agent/agent_core.py` → `stream_chat()` / `chat()` 中非首次分支 |
| 响应过滤规则 | `agent/agent_core.py` → `_JUNK_LINES`、`_filter_agent_response()` |
| 市场行情触发关键词 | `agent/agent_core.py` → `_MARKET_QUERY_PATTERNS` |

### LLM 调用双模式

```python
# Ollama 本地
ChatOllama(model=..., temperature=..., base_url=..., stop=[...])

# OpenAI 兼容
ChatOpenAI(model=..., temperature=..., openai_api_key=..., openai_api_base=...)

# 通过 config.Config.MODEL_PROVIDER 判断
# 切换时 rebuild_agent() 重新创建 InterviewerAgent 实例
```

---

## 八、风险与注意事项

1. **`data/model_settings.json` 含 API Key** — 已在 `.gitignore` 中，永远不要提交
2. **爬虫依赖 Chrome 浏览器** — `JobScraperTool` 的 `binary_location` 硬编码了 Windows Chrome 路径，非 Windows 需修改
3. **Embedding 模型必须预装** — `ollama pull qwen3-embedding:0.6b`，否则 RAG 全部报错
4. **LLM 可能不支持 tool calling** — 小模型（如 1.5B）可能无法正确调用工具，需要切换到支持 function calling 的模型
5. **`_current_session` 是 ContextVar** — 仅在同一次 `stream_chat()` 调用周期内有效，异步场景注意上下文传递
6. **SQLite checkpoint 不上传 Git** — 每个环境独立生成，`data/checkpoints.db` 已在 `.gitignore`
7. **Chroma 向量库是二进制文件** — `data/vector_store/` 已在 `.gitignore`，不同机器独立构建

---

## 九、未来可扩展方向

| 方向 | 说明 |
|---|---|
| 多语言支持 | 目前仅中文，可扩展英文面试 |
| 面试评估报告 | 每轮面试后自动生成评分和弱项分析 |
| 语音输入输出 | 对接 TTS/STT 实现语音面试 |
| 更多岗位平台 | 爬虫扩展 BOSS 直聘、拉勾等 |
| HR 端 | 反向场景：HR 发布岗位，AI 模拟求职者 |
| 面试题库 | 预设各技术栈的经典面试题模板 |
| 多模态简历 | 支持图片简历 OCR 识别 |
| 面试回放 | 保存完整对话记录，支持回看和分析 |
