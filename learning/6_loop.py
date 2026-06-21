"""
Part 6 — The Agent Loop
========================
Goal: Make agents loop — do work, check quality, repeat if needed.
This is what makes agents powerful vs a single LLM call.

Flow:
    worker → critic → (loop back to worker if quality is poor)
                    → (END if quality is good or max iterations hit)
"""

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
    draft: str
    critique: str
    iteration: int                          # tracks how many loops we've done
    messages: Annotated[list, operator.add] # full history accumulates

MAX_ITERATIONS = 3  # safety cap — never loop more than this

# ── LLMs

worker_llm = ChatOllama(
    model="qwen3.6:35b-a3b",
    temperature=0.3,                        # slight creativity for writing
    extra_body={"think": True},
)

critic_llm = ChatOllama(
    model="qwen3.6:35b-a3b",
    temperature=0,
    extra_body={"think": False},            # critic just checks, no creativity
)

# ── Prompts

WORKER_SYSTEM = """You are a technical writer. Write a clear, accurate explanation.
If a critique is provided, incorporate the feedback and improve your answer."""

CRITIC_SYSTEM = """You are a strict quality reviewer.
Read the draft and decide if it is good enough.

If it is good: reply with exactly: APPROVED
If it needs improvement: reply with a short critique (2-3 sentences max) explaining what to fix.

Be strict but fair. Only approve when the explanation is clear and complete."""

# ── Nodes

def worker_node(state: State) -> State:
    """Write or revise the draft based on the task and any critique."""
    iteration = state.get("iteration", 0)
    print(f"[worker] iteration {iteration + 1}")

    content = f"Task: {state['task']}"
    if state.get("critique") and state["critique"] != "APPROVED":
        content += f"\n\nPrevious draft:\n{state['draft']}"
        content += f"\n\nCritique to address:\n{state['critique']}"

    with spinner(f"Worker writing (iteration {iteration + 1})"):
        response = worker_llm.invoke([
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(content=content),
        ])

    draft = response.content.strip()
    print(f"[worker] draft written ({len(draft.split())} words)")

    return {
        **state,
        "draft": draft,
        "iteration": iteration + 1,
        "messages": [{"role": "worker", "content": f"Draft v{iteration + 1} written."}],
    }


def critic_node(state: State) -> State:
    """Review the draft and either approve it or return a critique."""
    with spinner("Critic reviewing draft..."):
        print(f"[critic] reviewing draft...")
        response = critic_llm.invoke([
            SystemMessage(content=CRITIC_SYSTEM),
            HumanMessage(content=f"Task: {state['task']}\n\nDraft:\n{state['draft']}"),
        ])

    critique = response.content.strip()
    approved = critique.upper() == "APPROVED"
    print(f"[critic] {'✓ APPROVED' if approved else '✗ needs revision'}")

    return {
        **state,
        "critique": critique,
        "messages": [{"role": "critic", "content": critique}],
    }


# ── Loop decision

def should_continue(state: State) -> str:
    """After the critic runs, decide: loop back to worker or finish?"""
    if state["critique"].upper() == "APPROVED":
        print("[loop] approved — finishing")
        return END

    if state.get("iteration", 0) >= MAX_ITERATIONS:
        print(f"[loop] hit max iterations ({MAX_ITERATIONS}) — finishing anyway")
        return END

    print("[loop] looping back to worker")
    return "worker"


# ── Build graph

graph = StateGraph(State)
graph.add_node("worker", worker_node)
graph.add_node("critic", critic_node)

graph.set_entry_point("worker")
graph.add_edge("worker", "critic")

graph.add_conditional_edges(
    "critic",
    should_continue,
    {"worker": "worker", END: END},
)

app = graph.compile()

# ── Run

print("=" * 60)
task = "Explain how the backpropagation algorithm works in neural networks"
print(f"Task: {task}")
print("=" * 60)
total_timer = Timer()
result = app.invoke({
    "task": task,
    "draft": "",
    "critique": "",
    "iteration": 0,
    "messages": [],
})
total_timer.print_elapsed(f"\nCompleted in {result['iteration']} iteration(s)")
print(f"\n{'='*60}")
print("FINAL DRAFT:")
print("=" * 60)
print(result["draft"])
print(f"\nFinal critique: {result['critique']}")