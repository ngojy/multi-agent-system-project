"""
Part 1 — Basic LLM call with Ollama
====================================
Goal: Get a response from qwen3.6 via LangChain.
No agents, no graphs. Just a plain LLM call.
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from utils import spinner, Timer
from dotenv import load_dotenv
import os

# Load model from .env file
load_dotenv()
model = os.getenv("MODEL", "qwen3.6:35b-a3b")

# ── Setup

llm = ChatOllama(
    model=model,
    temperature=0,
)

# ── Single message

print("=== Single message ===")
with spinner("Calling LLM"):
    response = llm.invoke([HumanMessage(content="What is LangGraph in one sentence?")])
print(response.content)

# ── System + human message

print("\n=== System + human message ===")
messages = [
    SystemMessage(content="You are a concise assistant. Reply in bullet points only."),
    HumanMessage(content="What are the 3 main components of a multi-agent system?"),
]
with spinner("Calling LLM"):
    response = llm.invoke(messages)
print(response.content)

# ── Streaming (see tokens arrive one by one)

print("\n=== Streaming ===")
timer = Timer()
print("  ", end="", flush=True)
for chunk in llm.stream([HumanMessage(content="Count from 1 to 5 slowly.")]):
    print(chunk.content, end="", flush=True)
print()