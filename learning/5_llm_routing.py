"""
Part 5 — LLM-Powered Routing
==============================
Goal: Replace the hardcoded keyword rule from Part 4 with an
actual LLM decision. This is the real orchestrator pattern used
in the full multi-agent system.

The LLM reads the task and replies with ONE word: researcher, coder, or writer.
"""

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from typing_extensions import TypedDict
from utils import spinner, Timer
from dotenv import load_dotenv
import os

# Load model from .env file
load_dotenv()
model = os.getenv("MODEL", "qwen3.6:35b-a3b")

# ── State

class State(TypedDict):
    input: str
    route: str
    result: str

# ── LLMs
# Router uses think=False for fast, strict one-word replies.
# Specialists use think=True so they reason before answering.

router_llm = ChatOllama(
    model=model,
    temperature=0,
    extra_body={"think": False},   # fast, no reasoning needed for routing
)

specialist_llm = ChatOllama(
    model=model,
    temperature=0,
    extra_body={"think": True},    # think before answering
)

# ── Router prompt

ROUTER_SYSTEM = """You are a router in a multi-agent system.
Read the task and reply with EXACTLY one word — nothing else, no punctuation.

Choose from:
  researcher  → task needs information gathering or web knowledge
  coder       → task needs Python code written or explained
  writer      → task needs prose, explanation, or documentation

One word only."""

# ── Nodes

def router_node(state: State) -> State:
    """Use the LLM to decide which specialist to call."""
    response = router_llm.invoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=f"Task: {state['input']}"),
    ])

    route = response.content.strip().lower()

    # Validate — if LLM returns something unexpected, default to writer
    valid = {"researcher", "coder", "writer"}
    if route not in valid:
        print(f"[router] unexpected route '{route}', defaulting to writer")
        route = "writer"

    print(f"[router] LLM decided → {route}")
    return {**state, "route": route}


def researcher_node(state: State) -> State:
    """Specialist: gathers and summarises information."""
    print("[researcher] researching...")
    response = specialist_llm.invoke([
        SystemMessage(content="You are a research analyst. Summarise key facts clearly and concisely."),
        HumanMessage(content=state["input"]),
    ])
    return {**state, "result": response.content}


def coder_node(state: State) -> State:
    """Specialist: writes Python code."""
    print("[coder] writing code...")
    response = specialist_llm.invoke([
        SystemMessage(content="You are a Python expert. Write clean, well-commented code."),
        HumanMessage(content=state["input"]),
    ])
    return {**state, "result": response.content}


def writer_node(state: State) -> State:
    """Specialist: writes prose or documentation."""
    print("[writer] writing...")
    response = specialist_llm.invoke([
        SystemMessage(content="You are a clear, concise technical writer."),
        HumanMessage(content=state["input"]),
    ])
    return {**state, "result": response.content}


def pick_route(state: State) -> str:
    return state["route"]


# ── Build graph

graph = StateGraph(State)
graph.add_node("router", router_node)
graph.add_node("researcher", researcher_node)
graph.add_node("coder", coder_node)
graph.add_node("writer", writer_node)

graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",
    pick_route,
    {
        "researcher": "researcher",
        "coder": "coder",
        "writer": "writer",
    },
)

graph.add_edge("researcher", END)
graph.add_edge("coder", END)
graph.add_edge("writer", END)

app = graph.compile()

# ── Run

test_cases = [
    "What is the difference between TCP and UDP?",
    "Write a Python function that checks if a number is prime",
    "Explain what gradient descent is in simple terms",
]

for task in test_cases:
    print(f"\n{'='*60}")
    print(f"Input : {task}")
    print(f"{'='*60}")
    with spinner(f"Running graph"):
        result = app.invoke({"input": task, "route": "", "result": ""})
    print(f"Route  : {result['route']}")
    print(f"Result :\n{result['result']}")