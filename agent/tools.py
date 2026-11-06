"""Tools the agent can call: document search (RAG), calculator, weather."""
from __future__ import annotations

import ast
import logging
import math
import operator
from typing import Callable, Sequence

import requests
from langchain_chroma import Chroma
from langchain_core.tools import tool

from . import config

log = logging.getLogger(__name__)

# ----------------------------------------------------------------- calculator
_BIN = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"sqrt": math.sqrt, "log": math.log, "log10": math.log10, "sin": math.sin,
          "cos": math.cos, "tan": math.tan, "abs": abs, "round": round}
_CONSTS = {"pi": math.pi, "e": math.e}


def safe_eval(expr: str) -> float:
    """Evaluate arithmetic with an AST whitelist (never uses eval())."""

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                raise ValueError("Exponent too large.")
            return _BIN[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in _FUNCS and not node.keywords):
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError("Unsupported expression.")

    return ev(ast.parse(expr.strip(), mode="eval"))


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression exactly, e.g. '(1250 * 0.18) / 12' or 'sqrt(2) ** 3'.
    Supports + - * / // % **, parentheses, pi, e, sqrt, log, log10, sin, cos, tan, abs, round.
    Use this for any arithmetic instead of computing in your head."""
    try:
        return str(safe_eval(expression))
    except ZeroDivisionError:
        return "Error: division by zero."
    except (ValueError, SyntaxError, TypeError, OverflowError) as e:
        return f"Error: could not evaluate '{expression}' ({e})."


# -------------------------------------------------------------------- weather
_WMO = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog",
        48: "rime fog", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
        61: "light rain", 63: "rain", 65: "heavy rain", 71: "light snow", 73: "snow",
        75: "heavy snow", 80: "rain showers", 81: "heavy rain showers", 95: "thunderstorm"}


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city (free Open-Meteo API, no key needed).
    Input is a city name such as 'Kochi' or 'Paris'."""
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1}, timeout=8,
        )
        geo.raise_for_status()
        results = geo.json().get("results")
        if not results:
            return f"Error: could not find a place called '{city}'."
        place = results[0]
        wx = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": place["latitude"], "longitude": place["longitude"],
                    "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                               "wind_speed_10m,weather_code"},
            timeout=8,
        )
        wx.raise_for_status()
        cur = wx.json()["current"]
        desc = _WMO.get(cur.get("weather_code"), "unknown conditions")
        return (f"{place['name']}, {place.get('country', '')}: {cur['temperature_2m']}°C "
                f"(feels like {cur['apparent_temperature']}°C), {desc}, "
                f"humidity {cur['relative_humidity_2m']}%, wind {cur['wind_speed_10m']} km/h.")
    except requests.Timeout:
        return "Error: the weather service timed out. Try again shortly."
    except (requests.RequestException, KeyError, ValueError) as e:
        log.warning("weather lookup failed: %s", e)
        return "Error: the weather service is unavailable right now."


# ------------------------------------------------------------- document search
def make_search_tool(store: Chroma, get_selected_docs: Callable[[], Sequence[str]]):
    """Build the RAG tool. A factory + closure lets the UI control which documents
    are searched without the LLM having to know about document ids."""

    @tool
    def search_document(query: str) -> str:
        """Search the uploaded PDF document(s) for passages relevant to the query.
        Use this for ANY question about the content of the uploaded documents.
        Write a focused, self-contained query (rephrase follow-ups so they make sense alone)."""
        try:
            selected = list(get_selected_docs())
            if not selected:
                return "No document is available. Ask the user to upload a PDF first."
            flt = {"doc_id": selected[0]} if len(selected) == 1 else {"doc_id": {"$in": selected}}
            hits = store.similarity_search(query, k=config.TOP_K, filter=flt)
            if not hits:
                return "No relevant passages found in the document."
            return "\n\n".join(
                f"[{d.metadata['filename']}, p.{d.metadata['page']}]\n{d.page_content}" for d in hits
            )
        except Exception as e:
            log.exception("search_document failed")
            return f"Error: document search failed ({type(e).__name__}). Try rephrasing or retry."

    return search_document
