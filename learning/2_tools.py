"""
Part 2 — Tools
==============
Goal: Attach tools to the LLM so it can decide to call them.
The LLM returns a tool_call instead of plain text when appropriate.
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from utils import spinner, Timer
from dotenv import load_dotenv
import os

# Load model from .env file
load_dotenv()
model = os.getenv("MODEL", "qwen3.6:35b-a3b")

# ── Define tools

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    return a * b

@tool
def to_uppercase(text: str) -> str:
    """Convert a string to uppercase."""
    return text.upper()

@tool
def word_count(text: str) -> int:
    """Count how many words are in a string."""
    return len(text.split())

# ── Bind tools to the LLM

llm = ChatOllama(
    model=model,
    temperature=0,
)

llm_with_tools = llm.bind_tools([multiply, to_uppercase, word_count])

# ── Test: question that should trigger a tool call

print("=== Tool call test ===")
with spinner("Testing tool call"):
    response = llm_with_tools.invoke([HumanMessage(content="What is 42 multiplied by 7?")])

if response.tool_calls:
    print(f"Tool called : {response.tool_calls[0]['name']}")
    print(f"Arguments   : {response.tool_calls[0]['args']}")

    # Manually execute the tool to see the result
    result = multiply.invoke(response.tool_calls[0]["args"])
    print(f"Result      : {result}")
else:
    print("No tool call made. Response:", response.content)

# ── Test: question that should NOT trigger a tool call

print("\n=== No tool call test ===")
with spinner("Testing no tool call"):
    response = llm_with_tools.invoke([HumanMessage(content="What is the capital of France?")])

if response.tool_calls:
    print(f"Tool called: {response.tool_calls[0]['name']}")
else:
    print("No tool call — plain response:")
    print(response.content)