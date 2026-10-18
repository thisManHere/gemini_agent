import ast
import operator
from datetime import datetime
from pathlib import Path

import requests
from langchain_core.tools import tool

from rag import get_vectorstore
 
OUTPUT_DIR = Path("outputs")
_store = None

def _retriever():
    global _store
    if _store is None:
        _store = get_vectorstore()
    return _store

@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base (company docs) for relevant passages.
    Use this for any question about company policies, products or procedures."""
    results = _retriever().similarity_search(query, k=4)
    if not results:
        return "No relevant documents found."
    return "\n\n".join(f"[{d.metadata.get('source')}]\n{d.page_content}" for d in results)

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg}

def eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left = eval(node.left)
        right =  eval(node.right)

        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](eval(node.operand))
    raise ValueError("unsupported expression")

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression, e.g. '(1200 * 0.15) + 40'. Supports + - * / ** %."""
    try:
        return str(eval(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"Calculator error: {e}"

@tool
def get_current_time() -> str:
    """Get the current local date and time."""
    return datetime.now().strftime("%A, %d %B %Y, %H:%M")

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city (live data from Open-Meteo, no key needed)."""
    try:
        geo = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                           params={"name": city, "count": 1}, timeout=10).json()
        if not geo.get("results"):
            return f"Could not find a city called '{city}'."
        place = geo["results"][0]
        wx = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": place["latitude"], "longitude": place["longitude"],
            "current": "temperature_2m,wind_speed_10m,precipitation"}, timeout=10).json()["current"]
        return (f"{place['name']}, {place.get('country', '')}: {wx['temperature_2m']}°C, "
                f"wind {wx['wind_speed_10m']} km/h, precipitation {wx['precipitation']} mm")
    except Exception as e:
        return f"Weather lookup failed: {e}"

@tool
def save_report(title: str, content: str) -> str:
    """Save a finished report/summary as a markdown file. Use as the final step of
    a workflow when the user wants a written deliverable."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_ " else "" for c in title).strip().replace(" ", "_")
    path = OUTPUT_DIR / f"{datetime.now():%Y%m%d_%H%M}_{safe or 'report'}.md"
    path.write_text(f"# {title}\n\n{content}\n", encoding="utf-8")
    return f"Report saved to {path}"
 
 
TOOLS = [search_knowledge_base, calculator, get_current_time, get_weather, save_report]
 