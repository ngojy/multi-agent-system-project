"""
Part 3 — Single Node LangGraph
===============================
Goal: Build the smallest possible LangGraph.
Understand how state flows through a node and how messages accumulate.
"""

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Annotated
from typing_extensions import TypedDict
import operator
from utils import spinner
from dotenv import load_dotenv
import os

# Load model from .env file
load_dotenv()
model = os.getenv("MODEL", "qwen3.6:35b-a3b")

# ── State
# operator.add means messages ACCUMULATE across nodes instead of overwriting.
# This is the key pattern used throughout the full system.

class State(TypedDict):
    messages: Annotated[list, operator.add]

# ── LLM

llm = ChatOllama(
    model=model,
    temperature=0,
    extra_body={"think": False},  # disable thinking mode for speed
)

# ── Node

def assistant_node(state: State) -> State:
    """A single node that calls the LLM and appends its reply to messages."""
    # Grab just the human messages to pass to the LLM
    user_messages = [m for m in state["messages"] if m["role"] == "user"]

    lc_messages = [
        SystemMessage(content="You are a helpful assistant. Be concise."),
        HumanMessage(content=user_messages[-1]["content"]),
    ]

    response = llm.invoke(lc_messages)

    # Return only the NEW messages — operator.add merges them automatically
    return {
        "messages": [{"role": "assistant", "content": response.content}]
    }

# ── Build graph

graph = StateGraph(State)
graph.add_node("assistant", assistant_node)
graph.set_entry_point("assistant")
graph.add_edge("assistant", END)
app = graph.compile()

# ── Run

print("=== Single node graph ===")
initial_state = {
    "messages": [{"role": "user", "content": "What is a LangGraph node in one sentence?"}]
}
with spinner("Graph running..."):
    result = app.invoke(initial_state)

print(f"Total messages in state: {len(result['messages'])}")
for msg in result["messages"]:
    print(f"  [{msg['role']}] {msg['content']}")