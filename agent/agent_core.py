# agent/agent_core.py
import logging
import asyncio
from typing import AsyncGenerator

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

import config
from agent import tools
from agent import prompt
from agent.memory import SummaryMemory

logger = logging.getLogger(__name__)


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

        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.memory = SummaryMemory(llm=self.llm)
        self.agent = create_agent(
            model=self.llm,
            tools=tools,
            checkpointer=self.memory.checkpointer,
            system_prompt=system_prompt,
        )

    # ==================== 流式对话 ====================

    def stream_chat(self, user_id: str, user_message: str):
        config = self.memory.get_config(user_id)

        if self.memory.needs_summary(user_id):
            logger.info(f"用户 {user_id} 对话过长，触发自动摘要")
            self.memory.summarize_history(user_id)

        for chunk in self.agent.stream(
            {"messages": [HumanMessage(content=user_message)]},
            config=config,
            stream_mode="messages",
        ):
            if isinstance(chunk, tuple):
                msg, _ = chunk
                if hasattr(msg, "content") and msg.content:
                    yield msg.content

    # ==================== 非流式对话 ====================

    def chat(self, user_id: str, user_message: str) -> str:
        config = self.memory.get_config(user_id)

        if self.memory.needs_summary(user_id):
            logger.info(f"用户 {user_id} 对话过长，触发自动摘要")
            self.memory.summarize_history(user_id)

        result = self.agent.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=config,
        )

        messages = result.get("messages", [])
        if messages:
            last_msg = messages[-1]
            return last_msg.content if hasattr(last_msg, "content") else str(last_msg)
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