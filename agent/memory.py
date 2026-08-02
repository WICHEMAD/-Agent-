import os
import sqlite3
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from config import config
MAX_MESSAGE_COUNT = 20

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DB_DIR, exist_ok=True)

conn = sqlite3.connect(os.path.join(DB_DIR, "checkpoints.db"), check_same_thread=False)
checkpointer = SqliteSaver(conn)

class SummaryMemory:
    """管理对话记忆：多用户隔离、自动摘要、历史查询"""
    def __init__(self, llm):
        self.llm=llm
        self.checkpointer=checkpointer

    @staticmethod
    def get_config(user_id: str)-> dict:
        """为每个用户生成独立的thread_id,实现对话隔离"""
        return {"configurable": {"thread_id": user_id, "checkpoint_ns": ""}}

    def get_history(self, user_id: str) -> list:
        config = self.get_config(user_id)
        state = self.checkpointer.get(config)
        if state and "channel_values" in state:
            return state["channel_values"].get("messages", [])
        if state and "messages" in state:
            return state["messages"]
        return []
    def needs_summary(self, user_id: str)->bool:
        """判断指定用户是否需要触发自动摘要"""
        history=self.get_history(user_id)
        return len(history)>=MAX_MESSAGE_COUNT

    def summarize_history(self, user_id: str):
        """使用LLM对指定用户对话历史进行摘要"""
        messages=self.get_history(user_id)
        if not messages:
            return

        #保留最近4条消息，其余压缩为摘要
        keep_count=4
        if len(messages)<=keep_count:
            return

        #压缩为摘要
        to_summarize=messages[:-keep_count] #拿到除了最新4条消息以外的所有消息
        recent=messages[-keep_count:] #拿到最新4条消息

        #构建摘要prompt
        history_text="\n".join(
            f"{'用户' if m.type == 'human' else '面试官'}: {m.content}"
            for m in to_summarize
            if hasattr(m, "type") and hasattr(m, "content")
        )

        #摘要
        summary_response=self.llm.invoke([
            SystemMessage(content="请用一段简洁的中文总结以下面试对话的关键信息，包括求职者的技能、经验、期望薪资等。"),
            HumanMessage(content=history_text)
        ])


        #摘要和最近的消息
        config=self.get_config(user_id)
        self.checkpointer.put(
            config,
            {
                **empty_checkpoint(),
                "messages":[
                    SystemMessage(content=f"[对话摘要]{summary_response.content}"),
                    *recent,
                ]
            },
            metadata={},
            new_versions={},
        )

if __name__ == "__main__":

    llm = ChatOllama(model=config.CHAT_MODEL, temperature=0)
    mem = SummaryMemory(llm=llm)

    # 1. 写入几条消息
    cfg = mem.get_config("user_1")
    mem.checkpointer.put(cfg, {**empty_checkpoint(), "messages": [
        HumanMessage(content="你好，我面试Python"),
        AIMessage(content="你好，请介绍一下你的经验"),
        HumanMessage(content="我有3年后端经验"),
    ]}, metadata={}, new_versions={})

    # 2. 查历史
    print("历史条数:", len(mem.get_history("user_1")))

    # 3. 需要摘要吗？
    print("需要摘要:", mem.needs_summary("user_1"))