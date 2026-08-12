import os
import json
from logging import config

# ==================== 模型设置持久化 ====================

_MODEL_SETTINGS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "model_settings.json"
)

DEFAULT_MODEL_SETTINGS = {
    "provider": "ollama",          # ollama | openai_compatible
    "model_name": "deepseek-r1:7b",
    "api_key": "",
    "base_url": "http://localhost:11434",
}


def _load_model_settings() -> dict:
    """从持久化文件加载模型设置，不存在则返回默认值"""
    try:
        os.makedirs(os.path.dirname(_MODEL_SETTINGS_FILE), exist_ok=True)
        with open(_MODEL_SETTINGS_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
            # 合并默认值，确保所有 key 都存在
            merged = {**DEFAULT_MODEL_SETTINGS, **saved}
            return merged
    except (FileNotFoundError, json.JSONDecodeError):
        return {**DEFAULT_MODEL_SETTINGS}


def save_model_settings(settings: dict) -> None:
    """持久化模型设置"""
    os.makedirs(os.path.dirname(_MODEL_SETTINGS_FILE), exist_ok=True)
    with open(_MODEL_SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)


# 加载当前设置
_model_settings = _load_model_settings()


class Config:
    # ==================== 模型配置 ====================
    # 模型提供商：ollama | openai_compatible
    MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", _model_settings["provider"])
    # 模型名称
    CHAT_MODEL = os.getenv("CHAT_MODEL", _model_settings["model_name"])
    # API 密钥（openai_compatible 时使用）
    MODEL_API_KEY = os.getenv("MODEL_API_KEY", _model_settings["api_key"])
    # API 地址
    MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", _model_settings["base_url"])
    # Ollama 地址（provider=ollama 时使用）
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # ==================== 重试配置 ====================
    # Agent 调用最大重试次数（工具调用失败/LLM异常时）
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    # 重试基础等待时间（秒），会指数递增：1s, 2s, 4s...
    RETRY_BASE_DELAY = float(os.getenv("RETRY_BASE_DELAY", "1.0"))
    # 工具调用超时时间（秒）
    TOOL_TIMEOUT = int(os.getenv("TOOL_TIMEOUT", "30"))
    # 失败后是否启用降级（返回模拟/兜底回复）
    ENABLE_FALLBACK = True
    # ==================== RAG 配置 ====================
    # Embedding 模型（需先在 Ollama 中 pull：ollama pull nomic-embed-text）
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    # 向量数据库持久化目录
    VECTOR_STORE_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "vector_store"
    )
    # 文本分块大小（字符数）
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    # 分块重叠大小（字符数）
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    # 检索返回的最大文档数
    RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))

    @classmethod
    def reload_settings(cls):
        """从持久化文件重新加载设置（运行时动态切换模型）"""
        settings = _load_model_settings()
        cls.MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", settings["provider"])
        cls.CHAT_MODEL = os.getenv("CHAT_MODEL", settings["model_name"])
        cls.MODEL_API_KEY = os.getenv("MODEL_API_KEY", settings["api_key"])
        cls.MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", settings["base_url"])


config = Config()