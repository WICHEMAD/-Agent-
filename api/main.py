# api/main.py
import uuid
import json
import logging
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
# ==================== 简历上传 ====================

from fastapi import UploadFile, File
from agent.rag import get_rag_engine

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
import config
from config import save_model_settings
from agent.agent_core import InterviewerAgent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="面试官 Agent API")

# ==================== 静态文件 ====================
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ==================== 会话存储 ====================
SESSIONS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sessions.json")
os.makedirs(os.path.dirname(SESSIONS_FILE), exist_ok=True)

def load_sessions():
    try:
        with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_sessions(sessions):
    with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)

# ==================== Agent 单例 ====================
_agent: InterviewerAgent = None

def get_agent() -> InterviewerAgent:
    global _agent
    if _agent is None:
        _agent = InterviewerAgent()
    return _agent


def rebuild_agent():
    """重建 Agent 实例（模型设置变更后调用）"""
    global _agent
    logger.info("[模型切换] 重建 Agent 实例...")
    _agent = InterviewerAgent()
    return _agent

# ==================== 请求模型 ====================
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)

class CreateSessionRequest(BaseModel):
    title: str = Field(default="新面试")

# ==================== 会话管理 ====================

@app.get("/sessions")
async def list_sessions():
    return load_sessions()

@app.post("/sessions")
async def create_session(req: CreateSessionRequest):
    sessions = load_sessions()
    session = {
        "id": str(uuid.uuid4())[:8],
        "title": req.title,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "message_count": 0,
    }
    sessions.insert(0, session)
    save_sessions(sessions)
    return session

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    sessions = [s for s in load_sessions() if s["id"] != session_id]
    save_sessions(sessions)
    return {"status": "ok"}

# ==================== 对话接口 ====================

@app.post("/chat/{session_id}", response_class=StreamingResponse)
async def chat(session_id: str, request: ChatRequest):
    if not session_id or len(session_id) > 50:
        raise HTTPException(status_code=400, detail="session_id 不合法")

    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message 不能为空")

    agent = get_agent()

    # 更新会话消息计数和标题
    history = agent.get_history(session_id)
    if len(history) == 0:
        sessions = load_sessions()
        for s in sessions:
            if s["id"] == session_id:
                s["title"] = message[:20]
                break
        save_sessions(sessions)

    logger.info(f"[请求] session_id={session_id}, message={message[:50]}...")

    def generate():
        try:
            for token in agent.stream_chat(session_id, message):
                yield token
            # 更新消息计数
            sessions = load_sessions()
            for s in sessions:
                if s["id"] == session_id:
                    s["message_count"] = len(agent.get_history(session_id))
                    break
            save_sessions(sessions)
        except Exception as e:
            logger.error(f"[异常] session_id={session_id}: {e}", exc_info=True)
            yield "系统暂不可用，请稍后再试"

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")

@app.get("/chat/{session_id}/history")
async def get_history(session_id: str):
    agent = get_agent()
    history = agent.get_history(session_id)
    return {
        "messages": [
            {"role": "user" if m.type == "human" else "agent", "content": m.content}
            for m in history
            if hasattr(m, "type") and hasattr(m, "content")
        ]
    }

@app.post("/chat/{session_id}/reset")
async def reset_chat(session_id: str):
    agent = get_agent()
    agent.reset_conversation(session_id)
    return {"status": "ok"}

@app.get("/upload/{session_id}")
async def get_resume_info(session_id: str):
    """查询指定会话的简历上传状态"""
    sessions = load_sessions()
    for s in sessions:
        if s["id"] == session_id:
            resume = s.get("resume")
            if resume:
                return {"uploaded": True, **resume}
            return {"uploaded": False}
    return {"uploaded": False}


@app.post("/upload/{session_id}")
async def upload_resume(session_id: str, file: UploadFile = File(...)):
    """上传简历文档，自动解析、向量化并存入知识库"""
    # 校验文件类型
    allowed_ext = {".pdf", ".docx", ".txt", ".md"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_ext:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 {ext}，支持：{', '.join(allowed_ext)}",
        )

    # 保存到本地
    safe_filename = f"{session_id}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    logger.info(f"[上传] session={session_id}, file={file.filename}, size={len(content)} bytes")

    # 送入 RAG 引擎处理
    try:
        engine = get_rag_engine()
        chunk_count = engine.process_document(
            file_path,
            metadata={"session_id": session_id, "filename": file.filename},
        )
        # 同步更新会话中的简历信息
        sessions = load_sessions()
        for s in sessions:
            if s["id"] == session_id:
                s["resume"] = {
                    "filename": file.filename,
                    "size": len(content),
                    "chunks": chunk_count,
                    "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
                break
        save_sessions(sessions)

        return {
            "status": "ok",
            "filename": file.filename,
            "chunks": chunk_count,
            "message": f"简历已成功解析，共 {chunk_count} 个文本块存入知识库",
        }
    except Exception as e:
        logger.error(f"[RAG处理失败] {e}", exc_info=True)
        # 清理临时文件
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"简历处理失败：{e}")


@app.delete("/upload/{session_id}")
async def delete_resume(session_id: str):
    """删除指定会话的简历向量"""
    engine = get_rag_engine()
    engine.delete_by_session(session_id)
    # 同步清除会话中的简历信息
    sessions = load_sessions()
    for s in sessions:
        if s["id"] == session_id:
            s.pop("resume", None)
            break
    save_sessions(sessions)
    return {"status": "ok", "message": "简历已从知识库中删除"}

# ==================== 模型设置接口 ====================

class ModelSettingsRequest(BaseModel):
    provider: str = Field(default="ollama")          # ollama | openai_compatible
    model_name: str = Field(default="deepseek-r1:7b")
    api_key: str = Field(default="")
    base_url: str = Field(default="http://localhost:11434")


@app.get("/settings")
async def get_settings():
    """获取当前模型设置"""
    config.Config.reload_settings()
    return {
        "provider": config.Config.MODEL_PROVIDER,
        "model_name": config.Config.CHAT_MODEL,
        "api_key": config.Config.MODEL_API_KEY,
        "base_url": config.Config.MODEL_BASE_URL,
    }


@app.post("/settings")
async def update_settings(settings: ModelSettingsRequest):
    """更新模型设置并立即生效"""
    data = {
        "provider": settings.provider,
        "model_name": settings.model_name,
        "api_key": settings.api_key,
        "base_url": settings.base_url,
    }
    save_model_settings(data)
    config.Config.reload_settings()
    rebuild_agent()
    logger.info(f"[设置更新] provider={settings.provider}, model={settings.model_name}")
    return {"status": "ok", **data}


# ==================== 基础路由 ====================

@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

@app.get("/favicon.ico")
async def favicon():
    return StreamingResponse(iter([]), status_code=204)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)