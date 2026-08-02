import os
from logging import config


class Config:
    CHAT_MODEL = os.getenv("CHAT_MODEL", "deepseek-r1:7b")

config=Config()
