# CLAUDE.md — 面试官 Agent

## 项目概述

AI 技术面试官，FastAPI + LangChain + Ollama/OpenAI 双模式。上传简历后 LLM 扮演面试官提问，附带岗位爬虫、薪资计算、RAG 简历检索等工具。

## 关键文件

| 文件 | 职责 | 注意 |
|---|---|---|
| `config.py` | 全局配置，从 `data/model_settings.json` 加载，支持 `reload_settings()` 运行时热切换 | 新增配置项需同时在 `DEFAULT_MODEL_SETTINGS`、`Config` 类、`/settings` API、前端弹窗四处同步 |
| `agent/agent_core.py` | `InterviewerAgent`：LLM 初始化、流式/非流式对话、简历预取、响应过滤 | 核心类，修改时注意 `stream_chat()` 和 `chat()` 两套逻辑需保持一致 |
| `agent/prompt.py` | System Prompt 字符串 | 面试官角色约束、追问策略、工具使用规则都在这里 |
| `agent/tools.py` | `CalculatorTool`、`JobScraperTool`、`RAGResumeTool` | 新工具在此添加，然后在 `ALL_TOOLS` 列表注册 |
| `agent/memory.py` | `SummaryMemory`：LangGraph SQLite checkpointer，>20 条触发摘要 | 依赖 `data/checkpoints.db`，不上传 Git |
| `agent/rag.py` | `RAGEngine`：文档加载→分块→Chroma 向量化→检索 | 全局单例 `get_rag_engine()` |
| `api/main.py` | FastAPI 路由：会话 CRUD、流式对话、简历上传/删除、模型设置 | `/chat/{id}` 返回 `StreamingResponse` |
| `static/index.html` | 前端 SPA，纯原生 JS | 无框架依赖 |

## 架构约定

### 模型双模式

```python
# provider="ollama" → ChatOllama
# provider="openai_compatible" → ChatOpenAI
# 通过 config.Config.MODEL_PROVIDER 判断，config.reload_settings() 切换
```

### 工具注册

新工具继承 `BaseTool`，定义 `name`、`description`、`_run`、`_arun`，加入 `ALL_TOOLS` 列表。Agent 通过 `create_agent(tools=tools.ALL_TOOLS)` 自动获得。

### 响应过滤策略

`_filter_agent_response()` 处理 LLM 输出：
1. 检测"求职者："标记 → 截断（防自导自演）
2. 去掉"面试官："前缀
3. 删除废话行（`_JUNK_LINES` 正则）
4. 工具类响应（无问号多段落）→ 透传
5. 多问句 → 只取第一个

### 简历上下文注入

首次对话时 `_build_resume_context()` 程序化检索 RAG（不依赖 LLM 调用工具），将简历片段注入 `user_message` 前缀。按 `session_id` 过滤向量库。

### 对话隔离

每个会话通过 `thread_id = user_id`（此处即 `session_id`）隔离，`SummaryMemory.get_config(user_id)` 生成 LangGraph config。

### 数据目录

- `data/sessions.json` — 会话元数据（不上传 Git）
- `data/model_settings.json` — 模型配置（含 API Key，不上传 Git）
- `data/checkpoints.db` — SQLite 对话检查点（不上传 Git）
- `data/uploads/` — 上传简历（不上传 Git）
- `data/vector_store/` — Chroma 向量库（不上传 Git）

## 常见任务模式

### 新增工具
1. 在 `agent/tools.py` 创建继承 `BaseTool` 的类
2. 实现 `_run` 和 `_arun`
3. 加入 `ALL_TOOLS`
4. 在 `agent/prompt.py` 的 System Prompt 中告知 LLM 工具用途和触发条件

### 新增 API 接口
1. 在 `api/main.py` 添加路由
2. 如需新的请求体，定义 Pydantic model
3. 前端在 `static/index.html` 的 `<script>` 中对接

### 修改面试行为
1. 修改 `agent/prompt.py` 的 `SYSTEM_PROMPT` 调整策略
2. 修改 `agent/agent_core.py` 的 `_build_resume_context` / 消息前缀调整输入
3. 修改 `agent/agent_core.py` 的 `_filter_agent_response` 调整输出

### 切换模型
- 前端设置弹窗 → `POST /settings` → `save_model_settings()` + `config.reload_settings()` + `rebuild_agent()`
- 前端 `onProviderChange()` 控制表单字段显隐

## 注意事项

- `.gitignore` 已排除所有运行时数据和编译缓存，提交前确认 `git status` 干净
- `data/model_settings.json` 含 API Key，绝对不要提交
- `job_info_scraper` 依赖 Chrome 浏览器和 ChromeDriver，非 Windows 环境需调整 `binary_location`
- Embedding 模型需提前 `ollama pull qwen3-embedding:0.6b`，否则 RAG 报错
- `_current_session` 是 `contextvars.ContextVar`，`stream_chat()` 中设置供 `RAGResumeTool` 读取
- LLM 的 `stop` 参数设置为 `["求职者：", "求职者说：", "求职者回答"]` 辅助防自导自演
