# agent/rag.py
"""
RAG 引擎：文档解析 → 文本切分 → 向量化 → 存储 → 检索

使用方式：
    engine = RAGEngine()
    engine.process_document("resume.pdf", metadata={"session_id": "abc123"})
    docs = engine.retrieve("Python 后端经验", filter={"session_id": "abc123"})
"""
import os
import logging
from typing import List, Optional, Dict

from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config import config

logger = logging.getLogger(__name__)


class RAGEngine:
    """RAG 引擎：管理文档的解析、向量化存储与语义检索"""

    def __init__(
        self,
        embedding_model: str = None,
        vector_store_dir: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        embedding_model = embedding_model or config.EMBEDDING_MODEL
        vector_store_dir = vector_store_dir or config.VECTOR_STORE_DIR
        chunk_size = chunk_size or config.CHUNK_SIZE
        chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

        # 确保向量库目录存在
        os.makedirs(vector_store_dir, exist_ok=True)

        self.vector_store_dir = vector_store_dir
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434"),
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            length_function=len,
        )
        self._vector_store: Optional[Chroma] = None

    # ==================== 向量库懒加载 ====================

    def _get_vector_store(self) -> Chroma:
        if self._vector_store is None:
            self._vector_store = Chroma(
                persist_directory=self.vector_store_dir,
                embedding_function=self.embeddings,
            )
        return self._vector_store

    # ==================== 文档处理 ====================

    def _load_document(self, file_path: str) -> List[Document]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".docx":
            try:
                from langchain_community.document_loaders import Docx2txtLoader
                loader = Docx2txtLoader(file_path)
            except ImportError:
                raise ImportError("解析 .docx 需要安装 docx2txt：pip install docx2txt")
        elif ext in (".txt", ".md", ".py", ".java", ".go"):
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError(f"不支持的文件格式：{ext}，支持的格式：pdf, docx, txt, md")

        return loader.load()

    def process_document(
        self, file_path: str, metadata: Optional[Dict] = None
    ) -> int:
        """处理单个文档：加载 → 切分 → 向量化 → 存储

        Args:
            file_path: 文档路径
            metadata: 附加元数据（如 session_id, filename）

        Returns:
            存入的文本块数量
        """
        metadata = metadata or {}
        metadata["source"] = os.path.basename(file_path)

        # 1. 加载
        logger.info(f"[RAG] 加载文档: {file_path}")
        docs = self._load_document(file_path)

        # 2. 切分
        chunks = self.text_splitter.split_documents(docs)
        logger.info(f"[RAG] 切分为 {len(chunks)} 个文本块")

        # 3. 注入元数据
        for i, chunk in enumerate(chunks):
            chunk.metadata.update(metadata)
            chunk.metadata["chunk_index"] = i

        # 4. 向量化 + 存储
        vector_store = self._get_vector_store()
        vector_store.add_documents(chunks)

        logger.info(f"[RAG] 成功存入 {len(chunks)} 个向量")
        return len(chunks)

    # ==================== 检索 ====================

    def retrieve(
        self, query: str, k: int = None, filter: Optional[Dict] = None
    ) -> List[Document]:
        """语义检索

        Args:
            query: 查询文本
            k: 返回文档数
            filter: 元数据过滤条件（如 {"session_id": "abc123"}）

        Returns:
            相关文档列表（按相似度降序）
        """
        k = k or config.RETRIEVAL_K
        vector_store = self._get_vector_store()

        if filter:
            docs = vector_store.similarity_search(query, k=k, filter=filter)
        else:
            docs = vector_store.similarity_search(query, k=k)

        logger.info(f"[RAG] 检索 '{query[:50]}...' → 返回 {len(docs)} 个结果")
        return docs

    # ==================== 删除 ====================

    def delete_by_session(self, session_id: str):
        """删除指定会话的所有文档向量"""
        vector_store = self._get_vector_store()
        collection = vector_store._collection
        results = collection.get(where={"session_id": session_id})
        ids_to_delete = results.get("ids", [])
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            logger.info(f"[RAG] 删除会话 {session_id} 的 {len(ids_to_delete)} 个向量")


# ==================== 全局单例 ====================

_rag_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine