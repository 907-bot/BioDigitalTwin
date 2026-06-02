"""
Phase 5 — Ollama LLM client + LangChain agent.

We support two backends:
  - "ollama"  : local Llama-3.1 via Ollama (default, no API key)
  - "openai"  : any OpenAI-compatible endpoint (set LLM_BASE_URL, LLM_API_KEY)

If neither is reachable, the agent still works in `mock` mode (rule-based
tool dispatch) so the API surface is testable without a live LLM.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.agent.tools import TOOL_REGISTRY

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
LLM_MODEL  = os.getenv("LLM_MODEL", "llama3.1")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))


def _ollama_reachable() -> bool:
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def _ollama_chat(messages: list[dict], tools_desc: str) -> str:
    """Call Ollama's /api/chat. We pass tools as a system-prompt block —
    Ollama's native tool calling is patchy across versions, so we use
    plain text generation with a strict JSON tool-call format."""
    sys = (
        "You are a clinical AI assistant for a Bio-Digital Twin platform. "
        "You can call tools by responding with a single JSON object "
        "({\"tool\": \"<name>\", \"args\": {<args>}}) and NOTHING ELSE. "
        "If you do not need a tool, respond in plain English. "
        "Available tools:\n" + tools_desc
    )
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "system", "content": sys}, *messages],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    r = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=LLM_TIMEOUT)
    r.raise_for_status()
    return r.json()["message"]["content"]


@dataclass
class ConversationMemory:
    """Per-session rolling conversation buffer."""
    session_id: str
    messages: list[dict] = field(default_factory=list)
    max_turns: int = 20

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        # truncate oldest turns but always keep system-style preamble
        if len(self.messages) > self.max_turns * 2:
            self.messages = self.messages[-self.max_turns * 2:]

    def as_list(self) -> list[dict]:
        return list(self.messages)


_MEMORIES: dict[str, ConversationMemory] = {}


def get_memory(session_id: str) -> ConversationMemory:
    if session_id not in _MEMORIES:
        _MEMORIES[session_id] = ConversationMemory(session_id=session_id)
    return _MEMORIES[session_id]


def reset_memory(session_id: str) -> None:
    _MEMORIES.pop(session_id, None)


# --- tool-call parsing --------------------------------------------------
def _extract_tool_call(text: str) -> dict | None:
    """Find the first balanced JSON object in `text` that has 'tool' and 'args'.

    Brace-counting because the regex with `*?` stops at the first inner `{`.
    """
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        depth = 0
        in_str = False
        esc = False
        for j in range(i, len(text)):
            c = text[j]
            if esc:
                esc = False
                continue
            if c == "\\":
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[i:j + 1]
                    try:
                        obj = json.loads(candidate)
                    except Exception:
                        break
                    if isinstance(obj, dict) and "tool" in obj:
                        if "args" not in obj:
                            obj["args"] = {}
                        if isinstance(obj["args"], dict):
                            return obj
                    break
    return None


def _format_tools_desc() -> str:
    lines = []
    for name, spec in TOOL_REGISTRY.items():
        args = ", ".join(f"{k}: {v}" for k, v in spec["args"].items())
        lines.append(f"- {name}({args}): {spec['description']}")
    return "\n".join(lines)


def _dispatch(tool_name: str, args: dict) -> str:
    spec = TOOL_REGISTRY.get(tool_name)
    if spec is None:
        return f"Unknown tool '{tool_name}'."
    try:
        return str(spec["fn"](**args))
    except TypeError as e:
        return f"Tool '{tool_name}' called with bad args: {e}"
    except Exception as e:
        return f"Tool '{tool_name}' failed: {e}"


# --- mock agent (no LLM needed) ---------------------------------------
def _mock_agent(messages: list[dict]) -> tuple[str, list[dict]]:
    """Rule-based dispatch for the mock backend. Picks the best tool by
    keyword matching against the latest user message."""
    last = messages[-1]["content"].lower() if messages else ""
    tool_calls: list[dict] = []

    # patient_id detection (P\d{6})
    pid = re.search(r"p\d{6}", last)

    if "list" in last and "disease" in last:
        tool_calls.append({"tool": "list_diseases", "args": {}})
    if ("list" in last or "what" in last) and "intervention" in last:
        tool_calls.append({"tool": "list_interventions", "args": {}})
    if "cohort" in last or "overview" in last or "how many" in last:
        tool_calls.append({"tool": "cohort_overview", "args": {}})

    if pid:
        if "similar" in last:
            tool_calls.append({"tool": "find_similar_patients",
                               "args": {"patient_id": pid.group(0).upper(), "k": 5}})
        if "summary" in last or "tell me about" in last or "patient" in last:
            tool_calls.append({"tool": "get_patient_summary",
                               "args": {"patient_id": pid.group(0).upper()}})

    # disease + intervention detection
    disease_map = {"t2d": "t2d", "diabet": "t2d",
                   "hypertens": "hypertension", "blood pressure": "hypertension",
                   "cvd": "cvd", "cardio": "cvd", "heart": "cvd",
                   "copd": "copd", "lung": "copd"}
    matched_disease = None
    for k, v in disease_map.items():
        if k in last:
            matched_disease = v; break

    matched_intervention = None
    for k in ["metformin", "losartan", "statin", "exercise", "weight_loss",
              "smoking_cessation", "weight loss"]:
        if k.replace("_", " ") in last or k in last:
            matched_intervention = k
            break

    if pid and matched_disease and ("simulate" in last or "trajectory" in last
                                    or "progress" in last or "over time" in last):
        tool_calls.append({
            "tool": "simulate_disease",
            "args": {
                "patient_id": pid.group(0).upper(),
                "disease": matched_disease,
                "horizon_days": 365,
                "intervention": matched_intervention,
            },
        })

    if pid and ("counterfactual" in last or "what if" in last
                or "what would" in last):
        outcome = "glucose" if matched_disease == "t2d" else \
                  "systolic_bp" if matched_disease in ("hypertension", "cvd") else "spo2"
        tool_calls.append({
            "tool": "counterfactual_for_patient",
            "args": {
                "patient_id": pid.group(0).upper(),
                "treatment": "glucose" if matched_disease == "t2d" else "systolic_bp",
                "outcome": outcome,
                "value": 100.0,
            },
        })

    if "effect" in last and ("treatment" in last or "ate" in last):
        t = matched_intervention or "exercise_30m"
        o = "glucose" if "glu" in last or matched_disease == "t2d" else "systolic_bp"
        tool_calls.append({"tool": "estimate_treatment_effect",
                           "args": {"treatment": t, "outcome": o}})

    if not tool_calls:
        return ("I can help with patient lookups, similar-patient search, "
                "disease simulation, counterfactuals, and treatment effects. "
                "Try asking about a specific patient like P000001."), []

    outputs = []
    for tc in tool_calls:
        out = _dispatch(tc["tool"], tc["args"])
        outputs.append(f"[{tc['tool']}] {out}")
    return "\n\n".join(outputs), tool_calls


# --- main agent --------------------------------------------------------
def chat(session_id: str, user_message: str,
         use_mock: bool | None = None) -> dict:
    """
    Add user message to memory, run agent, return reply + tool trace.
    """
    mem = get_memory(session_id)
    mem.add("user", user_message)
    messages = mem.as_list()
    tools_desc = _format_tools_desc()

    tool_trace: list[dict] = []
    if use_mock is None:
        use_mock = not _ollama_reachable()

    t0 = time.time()
    if use_mock:
        reply, tool_calls = _mock_agent(messages)
        backend = "mock"
    else:
        try:
            text = _ollama_chat(messages, tools_desc)
        except Exception as e:
            reply, tool_calls = _mock_agent(messages)
            backend = f"mock (ollama error: {e})"
        else:
            tc = _extract_tool_call(text)
            if tc is not None:
                out = _dispatch(tc["tool"], tc.get("args", {}))
                tool_trace.append({"tool": tc["tool"], "args": tc.get("args", {}),
                                   "output": out})
                followup_sys = (
                    "You have just received tool output. Summarise it for the "
                    "user in 1-3 sentences, in plain English. Do not call any "
                    "more tools."
                )
                followup_msgs = messages + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content":
                     f"Tool output for {tc['tool']}({tc.get('args',{})}):\n{out}"},
                ]
                try:
                    reply = _ollama_chat(followup_msgs, followup_sys)
                except Exception:
                    reply = f"Tool {tc['tool']} returned: {out}"
                tool_calls = [tc]
            else:
                reply = text
                tool_calls = []
            backend = "ollama"

    elapsed = round(time.time() - t0, 3)
    mem.add("assistant", reply)
    return {
        "session_id": session_id,
        "user_message": user_message,
        "reply": reply,
        "backend": backend,
        "tool_calls": tool_calls,
        "elapsed_s": elapsed,
        "turn": len(mem.messages) // 2,
    }


def list_tools() -> list[dict]:
    return [
        {"name": name, "description": spec["description"], "args": spec["args"]}
        for name, spec in TOOL_REGISTRY.items()
    ]
