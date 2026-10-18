from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from llm import get_llm
from tools import TOOLS

SYSTEM_PROMPT = """
You are a planning module for an assistant with these tools:
{tools}
 
Write a short numbered plan (max 5 steps) for answering the user's latest request.
Say which tool each step needs, or 'no tool'. Do not execute anything. 
"""

AGENT_PROMPT = """You are a helpful assistant that solves tasks step by step.
Follow this plan, adapting it if tool results require:
 
{plan}
 
Rules:
- Use search_knowledge_base for company/internal questions and ground answers in what it returns.
- If the knowledge base doesn't contain the answer, say so; don't invent facts.
- Use calculator for any arithmetic. Don't do math in your head.
- Call tools one after another as needed, then give a clear final answer."""


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    plan: str

def build_agent():
    llm = get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)
    tool_desc = "\n".join(f"- {t.name}: {t.description.splitlines()[0]}" for t in TOOLS)

    def planner(state: State):
        msgs = [SystemMessage(SYSTEM_PROMPT.format(tools=tool_desc))] + state["messages"]
        return {"plan": llm.invoke(msgs).content}

    def agent(state: State):
        msgs = [SystemMessage(AGENT_PROMPT.format(plan=state["plan"]))] + state["messages"]
        return {"messages": [llm_with_tools.invoke(msgs)]}

    g = StateGraph(State)
    g.add_node("planner", planner)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "planner")
    g.add_edge("planner", "agent")
    g.add_conditional_edges("agent", tools_condition) 
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=MemorySaver())