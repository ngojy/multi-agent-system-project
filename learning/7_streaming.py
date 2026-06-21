"""
Part 7 — Streaming
===================
Goal: See each agent's output as it happens instead of waiting
for the whole graph to finish. This is how main.py works in the
full multi-agent system.

Three streaming modes are shown:
  1. step-by-step  — one dict per node, after the node finishes
  2. token-by-token — LLM tokens as they stream in (astream_events)
  3. combined       — node name + streamed tokens together
"""

import asyncio
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Annotated
from typing_extensions import TypedDict
import operator
from utils import spinner, Timer
from dotenv import load_dotenv
import os

# Load model from .env file
load_dotenv()
model = os.getenv("MODEL", "qwen3.6:35b-a3b")

# ── State

class State(TypedDict):
    task: str
    research: str
    answer: str
    messages: Annotated[list, operator.add]

# ── LLM

llm = ChatOllama(
    model=model,
    temperature=0,
    extra_body={"think": False},
)

# ── Nodes

def researcher_node(state: State) -> State:
    response = llm.invoke([
        SystemMessage(content="You are a researcher. Summarise the key facts in 3 bullet points."),
        HumanMessage(content=state["task"]),
    ])
    return {
        **state,
        "research": response.content,
        "messages": [{"role": "researcher", "content": response.content}],
    }


def answerer_node(state: State) -> State:
    response = llm.invoke([
        SystemMessage(content="You are a helpful assistant. Use the research to answer clearly."),
        HumanMessage(content=f"Question: {state['task']}\n\nResearch:\n{state['research']}"),
    ])
    return {
        **state,
        "answer": response.content,
        "messages": [{"role": "answerer", "content": response.content}],
    }

# ── Build graph

graph = StateGraph(State)
graph.add_node("researcher", researcher_node)
graph.add_node("answerer", answerer_node)
graph.set_entry_point("researcher")
graph.add_edge("researcher", "answerer")
graph.add_edge("answerer", END)
app = graph.compile()

initial_state = {
    "task": "What is the difference between supervised and unsupervised learning?",
    "research": "",
    "answer": "",
    "messages": [],
}

# ── Mode 1: Step-by-step streaming
# app.stream() yields one dict per node after it finishes.
# Each dict is {node_name: node_state}.

print("=" * 60)
print("MODE 1: Step-by-step (one update per node)")
print("=" * 60)

for step in app.stream(initial_state):
    node_name, node_state = next(iter(step.items()))
    msgs = node_state.get("messages", [])
    if msgs:
        latest = msgs[-1]
        print(f"\n[{node_name.upper()}]")
        print(latest["content"][:300] + "..." if len(latest["content"]) > 300 else latest["content"])

# ── Mode 2: Token-by-token streaming
# astream_events() yields individual LLM tokens as they are generated.
# This requires async — we wrap it in asyncio.run().

print("\n\n" + "=" * 60)
print("MODE 2: Token-by-token streaming")
print("=" * 60)

async def stream_tokens():
    current_node = ""
    async for event in app.astream_events(initial_state, version="v2"):
        kind = event["event"]
        # Track which node we're in
        if kind == "on_chain_start" and event.get("name") in ("researcher", "answerer"):
            current_node = event["name"]
            print(f"\n[{current_node.upper()}] ", end="", flush=True)
        # Print tokens as they arrive
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                print(chunk.content, end="", flush=True)
    print()  # newline at end

asyncio.run(stream_tokens())

# ── Mode 3: Combined — node label + streamed output
# Practical version: show which node is running, then its full output.

print("\n\n" + "=" * 60)
print("MODE 3: Combined (as used in main.py)")
print("=" * 60)

final_state = None
for step in app.stream(initial_state):
    node_name, node_state = next(iter(step.items()))
    final_state = node_state
    msgs = node_state.get("messages", [])
    for msg in msgs:
        role = msg.get("role", node_name)
        content = msg.get("content", "")
        print(f"\n[{role.upper()}]\n{content}")

print(f"\n{'='*60}")
print("FINAL ANSWER:")
print("=" * 60)
print(final_state.get("answer", ""))