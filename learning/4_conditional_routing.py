"""
Part 4 — Conditional Routing (rule-based)
==========================================
Goal: Build a graph with two specialist nodes and a router that
picks between them using a simple rule (no LLM needed yet).

Flow:
    router → coder   (if "code" in input)
    router → writer  (otherwise)
"""

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from typing_extensions import TypedDict
from utils import spinner
from dotenv import load_dotenv
import os

# Load model from .env file
load_dotenv()
model = os.getenv("MODEL", "qwen3.6:35b-a3b")

# ── State

class State(TypedDict):
    input: str      # the user's request
    route: str      # set by router: "coder" or "writer"
    result: str     # set by the specialist

# ── LLM

llm = ChatOllama(
    model=model,
    temperature=0,
    extra_body={"think": False},
)

# ── Nodes

def router_node(state: State) -> State:
    """Decide which specialist to call based on a simple keyword rule."""
    route = "coder" if "code" in state["input"].lower() else "writer"
    print(f"[router] routing to → {route}")
    return {**state, "route": route}


def coder_node(state: State) -> State:
    """Specialist: writes code."""
    print("[coder] generating code...")
    response = llm.invoke([
        SystemMessage(content="You are an expert Python developer. Write clean, commented code only."),
        HumanMessage(content=state["input"]),
    ])
    return {**state, "result": response.content}


def writer_node(state: State) -> State:
    """Specialist: writes prose."""
    print("[writer] generating prose...")
    response = llm.invoke([
        SystemMessage(content="You are a clear, concise technical writer."),
        HumanMessage(content=state["input"]),
    ])
    return {**state, "result": response.content}


# ── Routing function
# This function is called AFTER router_node runs.
# It reads state and returns the name of the next node as a string.

def pick_route(state: State) -> str:
    return state["route"]


# ── Build graph

graph = StateGraph(State)
graph.add_node("router", router_node)
graph.add_node("coder", coder_node)
graph.add_node("writer", writer_node)

graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",        # from this node
    pick_route,      # call this function to decide
    {
        "coder": "coder",    # if pick_route returns "coder" → go to coder node
        "writer": "writer",  # if pick_route returns "writer" → go to writer node
    },
)

graph.add_edge("coder", END)
graph.add_edge("writer", END)

app = graph.compile()

# ── Run both routes

test_cases = [
    "Write me Python code to reverse a string",
    "Explain what a neural network is",
]

for task in test_cases:
    print(f"\n{'='*60}")
    print(f"Input: {task}")
    print(f"{'='*60}")
    with spinner(f"Running graph"):
        result = app.invoke({"input": task, "route": "", "result": ""})
    print(f"Route taken : {result['route']}")
    print(f"Result      :\n{result['result']}")