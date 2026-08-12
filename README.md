# 面试官 Agent

AI 驱动的技术面试官应用。上传简历后，由 LLM 扮演专业面试官进行模拟面试；支持岗位行情查询、薪资计算、简历语义检索等功能。

## 功能清单

| 功能 | 说明 |
|---|---|
| 🤖 AI 模拟面试 | LLM 扮演技术面试官，基于简历内容提问，深挖技术细节 |
| 📄 简历上传 + RAG | 上传 PDF/DOCX/TXT/MD 简历，自动分块→向量化→语义检索，面试时注入上下文 |
| 🔍 岗位行情爬虫 | Selenium 爬取猎聘真实岗位信息（职位、薪资、公司） |
| 🧮 薪资计算器 | 安全 eval 计算涨薪、税后等数学表达式 |
| 💬 流式对话 | SSE 流式输出面试官回复 |
| 🧠 对话记忆 | SQLite checkpoint 持久化对话，超过 20 条自动 LLM 摘要压缩 |
| 🔄 多会话管理 | 创建/切换/删除面试会话，对话完全隔离 |
| ⚙ 多模型切换 | 支持 Ollama 本地模型 / OpenAI 兼容 API，前端设置热切换 |
| 🛡 响应过滤 | 防止 LLM 自导自演（编造求职者回答），提取纯面试官输出 |
| ♻ 重试+降级 | Agent 调用、工具调用、爬虫三层重试，失败后降级兜底回复 |

## 技术栈

| 层 | 技术 |
|---|---|
| 后端框架 | FastAPI + Uvicorn |
| AI 框架 | LangChain + LangGraph |
| 模型 | Ollama（本地）/ OpenAI 兼容 API |
| 向量库 | ChromaDB |
| Embedding | Ollama Embedding（qwen3-embedding:0.6b） |
| 爬虫 | Selenium + ChromeDriver（猎聘） |
| 存储 | SQLite（对话检查点）、JSON（会话元数据） |
| 前端 | 原生 HTML/CSS/JS（SPA） |

## 项目结构

```
interviewer_agent/
├── config.py              # 全局配置，支持持久化热切换
├── requirements.txt       # Python 依赖
├── .gitignore
├── agent/
│   ├── agent_core.py      # InterviewerAgent：LLM 调用、流式对话、响应过滤
│   ├── prompt.py           # System Prompt：面试官角色、追问策略
│   ├── tools.py            # CalculatorTool / JobScraperTool / RAGResumeTool
│   ├── memory.py           # SummaryMemory：多用户隔离、自动摘要
│   └── rag.py              # RAGEngine：文档解析→分块→向量化→检索
├── api/
│   └── main.py             # FastAPI：会话 / 对话 / 简历 / 设置 REST 接口
├── static/
│   └── index.html          # 前端 SPA（侧边栏 + 聊天区 + 设置弹窗）
└── data/                   # 运行时数据（不上传 Git）
    ├── sessions.json       # 会话元数据
    ├── model_settings.json # 模型设置（含 API Key）
    ├── uploads/            # 用户上传简历文件
    └── vector_store/       # Chroma 向量库
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 Ollama（本地模式）

```bash
# 拉取对话模型
ollama pull deepseek-r1:7b

# 拉取 Embedding 模型（RAG 必需）
ollama pull qwen3-embedding:0.6b
```

### 3. 启动服务

```bash
python -m api.main
```

访问 `http://127.0.0.1:8000`

### 切换模型

点击页面右上角 ⚙ 按钮，支持：
- **Ollama 本地**：默认 `deepseek-r1:7b`，填入本地 Ollama 地址
- **OpenAI 兼容 API**：支持 DeepSeek、GPT-4o 等，填入 API Key 和 Base URL

## 对话流

```
用户输入 → FastAPI → InterviewerAgent.stream_chat()
    ├── 简历预取（RAG 检索 → 注入上下文）
    ├── LLM 调用（LangChain Agent + 工具）
    ├── 响应过滤（防自导自演 → 提取问句）
    └── SSE 流式返回 → 前端渲染
```

## 工具说明

| 工具名称 | 触发时机 | 功能 |
|---|---|---|
| `simple_calculator` | 薪资计算、涨薪比例 | 安全 eval 数学表达式 |
| `job_info_scraper` | 询问岗位行情 | Selenium 爬取猎聘前 5 条岗位 |
| `resume_knowledge_base` | 查询简历细节 | 从 Chroma 向量库语义检索 |

## 记忆机制

- 每个会话独立 `thread_id`，对话通过 LangGraph SQLite 持久化
- 消息数 > 20 条时自动触发 LLM 摘要压缩（保留最近 4 条原文）
- 摘要写入 checkpointer，后续对话基于摘要 + 最近消息

## 配置说明

所有配置见 `config.py`，支持环境变量覆盖：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MODEL_PROVIDER` | `ollama` | 模型提供商 |
| `CHAT_MODEL` | `deepseek-r1:7b` | 对话模型名 |
| `MODEL_API_KEY` | — | API Key（OpenAI 兼容模式） |
| `MODEL_BASE_URL` | `http://localhost:11434` | API 地址 |
| `EMBEDDING_MODEL` | `qwen3-embedding:0.6b` | Embedding 模型 |
| `CHUNK_SIZE` | `500` | RAG 分块大小 |
| `MAX_RETRIES` | `3` | Agent 调用最大重试次数 |
