"""LangGraph graph definition for the Shitstorm-Simulation agent.

Dieses File wird von LangGraph Server / LangGraph Cloud geladen.
Die exportierte Variable `graph` ist der ausführbare Graph.
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Literal, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.types import interrupt
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Optional: LangSmith / LangChain Tracing deaktivieren, damit es auch ohne
# gültigen LANGSMITH_API_KEY keine 403-Fehler gibt.
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGSMITH_TRACING", "false")


class ShitstormState(TypedDict):
    """Gesamter Zustand der Shitstorm-Simulation.

    Dieser State wird zwischen den Nodes hin- und hergereicht und am Ende
    als Ergebnis zurückgegeben. Alle Keys müssen JSON-serialisierbar sein.
    """

    platform: str
    cause: str
    company_name: str
    round: int
    history: List[Dict[str, Any]]
    last_company_response: str
    last_community_comments: List[str]
    politeness_score: float
    responsibility_score: float
    reaction_score: float
    intensity: float
    status: Literal["running", "user_won", "user_lost"]
    summary: str


def _safe_load_json(text: str):
    """Versuche, robust JSON aus einem LLM-Output zu laden."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                return None
        # Evtl. reine Liste ohne geschweifte Klammern
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                return None
        return None


def _make_llm() -> ChatOpenAI:
    """Erzeuge das LLM für die Simulation.

    Das Modell kann über die Umgebungsvariable OPENAI_MODEL überschrieben werden.
    """
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    # OPENAI_API_KEY kommt ebenfalls aus der Umgebung (.env im Projekt).
    return ChatOpenAI(model=model_name, temperature=0.3)


def community_round(state: ShitstormState, llm: ChatOpenAI) -> ShitstormState:
    """Generiert Community-Kommentare für die aktuelle Runde."""
    state["round"] += 1
    round_num = state["round"]

    platform = state["platform"]
    cause = state["cause"]
    company_name = state["company_name"]
    intensity = state["intensity"]
    last_answer = state["last_company_response"]
    reaction_score = state["reaction_score"]

    if round_num == 1:
        situation_desc = "Dies sind die ersten Reaktionen der Community auf den Auslöser."
    else:
        if reaction_score >= 65:
            situation_desc = (
                "Die letzte Antwort des Unternehmens wurde überwiegend positiv aufgenommen."
            )
        elif reaction_score >= 45:
            situation_desc = (
                "Die letzte Antwort des Unternehmens war gemischt und hat die Lage nur leicht beruhigt."
            )
        else:
            situation_desc = (
                "Die letzte Antwort des Unternehmens kam schlecht an und hat die Community eher verärgert."
            )

    system_msg = SystemMessage(
        content=(
            "Du simulierst eine Kommentarspalte in einem Social-Media-Shitstorm.\n"
            "Schreibe auf Deutsch, im typischen Ton der jeweiligen Plattform.\n"
            "Erzeuge realistische, aber nicht beleidigende Kommentare.\n"
            "Antwort NUR als JSON-Liste von Strings, z.B.:\n"
            '["Kommentar 1", "Kommentar 2", "..."]\n'
            "Kein zusätzlicher Text, keine Erklärungen."
        )
    )

    human_msg = HumanMessage(
        content=(
            f"Plattform: {platform}\n"
            f"Unternehmen: {company_name}\n"
            f"Ursache des Shitstorms: {cause}\n"
            f"Aktuelle Shitstorm-Intensität (0-100): {intensity}\n"
            f"Runde: {round_num}\n"
            f"Situation: {situation_desc}\n"
            f"Letzte Antwort des Unternehmens:\n{last_answer or '(noch keine)'}\n\n"
            "Generiere 3 bis 6 kurze Kommentare der Community. "
            "Mische sachliche Kritik und emotionale Reaktionen. "
            "Es darf hart, aber nicht beleidigend oder diskriminierend sein."
        )
    )

    result = llm.invoke([system_msg, human_msg])
    comments = _safe_load_json(result.content) or []

    if not isinstance(comments, list) or not comments:
        # Fallback: Zeilenweise interpretieren
        lines = [ln.strip("- ").strip() for ln in result.content.splitlines() if ln.strip()]
        comments = lines[:5] or [
            "Ich bin echt sauer über diese Situation.",
            "So kann ein Unternehmen nicht mit seinen Kund:innen umgehen.",
        ]

    comments = [str(c) for c in comments]

    state["last_community_comments"] = comments
    for c in comments:
        state["history"].append(
            {
                "actor": "community",
                "round": round_num,
                "content": c,
            }
        )

    return state


def company_response_node(state: ShitstormState) -> ShitstormState:
    """Human-in-the-loop Node für die Unternehmensantwort.

    Statt input() in der CLI benutzen wir LangGraph `interrupt`, damit
    LangGraph Server / Cloud die Ausführung pausieren und von außen
    mit einer Antwort wieder aufgenommen werden kann.

    Erwartet wird, dass beim Resume ein String oder ein Dict wie
    {"text": "..."} zurückkommt.
    """
    prompt_payload: Dict[str, Any] = {
        "type": "company_response_request",
        "round": state["round"],
        "platform": state["platform"],
        "cause": state["cause"],
        "company_name": state["company_name"],
        "intensity": state["intensity"],
        "last_community_comments": state["last_community_comments"],
        "hint": (
            "Formuliere eine öffentliche Antwort des Unternehmens. "
            "Sei höflich, übernimm Verantwortung und biete konkrete Schritte an."
        ),
    }

    resume_value = interrupt(prompt_payload)

    if isinstance(resume_value, dict):
        answer = str(resume_value.get("text", "")).strip()
    else:
        answer = str(resume_value or "").strip()

    if not answer:
        # Keine neue Antwort – Status unverändert lassen
        return state

    state["last_company_response"] = answer
    state["history"].append(
        {
            "actor": "company",
            "round": state["round"],
            "content": answer,
        }
    )
    return state


def llm_evaluate(state: ShitstormState, llm: ChatOpenAI) -> ShitstormState:
    """Bewertet die Unternehmensantwort nach Höflichkeit und Verantwortung."""
    platform = state["platform"]
    cause = state["cause"]
    company_name = state["company_name"]
    answer = state["last_company_response"]

    system_msg = SystemMessage(
        content=(
            "Du bist ein professioneller Coach für Krisenkommunikation in sozialen Medien.\n"
            "Bewerte Antworten von Unternehmen in Shitstorms nach klaren Kriterien.\n"
            "Du antwortest ausschließlich als JSON-Objekt und gibst keine Erklärungen außerhalb von JSON."
        )
    )

    human_msg = HumanMessage(
        content=(
            "Bewerte die folgende Antwort eines Unternehmens in einem Shitstorm.\n\n"
            f"Plattform: {platform}\n"
            f"Unternehmen: {company_name}\n"
            f"Ursache des Shitstorms: {cause}\n\n"
            f"Antwort des Unternehmens:\n\"\"\"{answer}\"\"\"\n\n"
            "Kriterien (0–100):\n"
            "- politeness: Höflichkeit, respektvoller Ton, Empathie, keine Angriffe.\n"
            "- responsibility: Verantwortungsübernahme, klare Benennung eigener Fehler, Entschuldigung, glaubwürdige Maßnahmen.\n"
            "- overall: Gesamteindruck, wie gut die Antwort zur Deeskalation beiträgt.\n\n"
            "Gib deine Antwort NUR als gültiges JSON im folgenden Format zurück:\n"
            "{\n"
            '  \"politeness\": <Zahl 0-100>,\n'
            '  \"responsibility\": <Zahl 0-100>,\n'
            '  \"overall\": <Zahl 0-100>,\n'
            '  \"feedback\": \"Kurzes Feedback für die Nutzerin / den Nutzer\"\n'
            "}\n"
        )
    )

    result = llm.invoke([system_msg, human_msg])
    data = _safe_load_json(result.content) or {}

    def clamp_score(v: Any, default: float = 50.0) -> float:
        try:
            x = float(v)
        except (TypeError, ValueError):
            x = default
        return max(0.0, min(100.0, x))

    politeness = clamp_score(data.get("politeness"))
    responsibility = clamp_score(data.get("responsibility"))
    overall = clamp_score(data.get("overall", (politeness + responsibility) / 2.0))

    feedback = data.get("feedback") or "Keine detaillierte Rückmeldung verfügbar."

    state["politeness_score"] = politeness
    state["responsibility_score"] = responsibility
    state["reaction_score"] = overall

    state["history"].append(
        {
            "actor": "coach",
            "round": state["round"],
            "content": (
                f"Politeness={politeness:.1f}, Responsibility={responsibility:.1f}, "
                f"Overall={overall:.1f}. Feedback: {feedback}"
            ),
        }
    )

    return state


def update_intensity(state: ShitstormState) -> ShitstormState:
    """Aktualisiert die Shitstorm-Intensität basierend auf dem Gesamtscore."""
    prev = state["intensity"]
    score = state["reaction_score"]

    if score >= 80:
        delta = -20.0
    elif score >= 65:
        delta = -12.0
    elif score >= 50:
        delta = -5.0
    elif score >= 35:
        delta = +5.0
    elif score >= 20:
        delta = +12.0
    else:
        delta = +20.0

    new_intensity = max(0.0, min(100.0, prev + delta))
    state["intensity"] = new_intensity

    if new_intensity < 10.0:
        state["status"] = "user_won"
    elif new_intensity > 90.0:
        state["status"] = "user_lost"
    else:
        state["status"] = "running"

    state["history"].append(
        {
            "actor": "system",
            "round": state["round"],
            "content": f"Intensität von {prev:.1f} auf {new_intensity:.1f} geändert (Score={score:.1f}).",
        }
    )

    return state


def route_after_update(state: ShitstormState) -> str:
    """Bestimmt, ob weiter simuliert oder beendet wird."""
    if state["status"] == "running":
        return "continue"
    return "end"


def summarize(state: ShitstormState, llm: ChatOpenAI) -> ShitstormState:
    """Erzeugt eine kurze Zusammenfassung des Verlaufs."""
    outcome = {
        "user_won": "Der Shitstorm ist weitgehend abgeklungen.",
        "user_lost": "Der Shitstorm ist außer Kontrolle geraten.",
        "running": "Die Simulation wurde vorzeitig beendet.",
    }[state["status"]]

    history_lines: List[str] = []
    for h in state["history"]:
        actor = h.get("actor", "?")
        rnd = h.get("round", 0)
        content = h.get("content", "")
        if len(content) > 400:
            content = content[:400] + " [...]"
        history_lines.append(f"Runde {rnd} - {actor}: {content}")

    history_text = "\n".join(history_lines[-60:])

    system_msg = SystemMessage(
        content=(
            "Du bist ein Coach für Krisenkommunikation und sollst eine Trainingssimulation auswerten.\n"
            "Fasse den Verlauf des Shitstorms und das Verhalten des Users zusammen.\n"
            "Gib konkrete Lernpunkte und Verbesserungsvorschläge.\n"
            "Antwort auf Deutsch, in 2–4 kurzen Absätzen."
        )
    )

    human_msg = HumanMessage(
        content=(
            f"Ausgangssituation: Shitstorm auf {state['platform']} wegen \"{state['cause']}\".\n"
            f"Unternehmen: {state['company_name']}\n"
            f"Endgültige Shitstorm-Intensität: {state['intensity']:.1f} / 100\n"
            f"Ergebnis: {outcome}\n\n"
            "Ausschnitte aus dem Verlauf:\n"
            f"{history_text}"
        )
    )

    result = llm.invoke([system_msg, human_msg])
    summary = result.content.strip()
    state["summary"] = summary
    return state


def build_graph():
    """Erzeugt den ausführbaren LangGraph-Workflow für die Simulation."""
    llm = _make_llm()

    def community_node(state: ShitstormState) -> ShitstormState:
        return community_round(state, llm)

    def evaluate_node(state: ShitstormState) -> ShitstormState:
        return llm_evaluate(state, llm)

    def summarize_node(state: ShitstormState) -> ShitstormState:
        return summarize(state, llm)

    workflow = StateGraph(ShitstormState)

    workflow.add_node("community_round", community_node)
    workflow.add_node("company_response", company_response_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("update_intensity", update_intensity)
    workflow.add_node("summarize", summarize_node)

    workflow.set_entry_point("community_round")
    workflow.add_edge("community_round", "company_response")
    workflow.add_edge("company_response", "evaluate")
    workflow.add_edge("evaluate", "update_intensity")
    workflow.add_conditional_edges(
        "update_intensity",
        route_after_update,
        {
            "continue": "community_round",
            "end": "summarize",
        },
    )
    workflow.set_finish_point("summarize")

    return workflow.compile(name="Shitstorm-Simulation")


# Diese Variable wird von LangGraph Server / Cloud geladen
graph = build_graph()

__all__ = ["graph", "ShitstormState"]
