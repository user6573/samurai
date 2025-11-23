import os
import json
import random
from typing import List, Optional, Literal, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt


# ------------------------------------------------------------
# Modell / LLM
# ------------------------------------------------------------

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0.7,
)


# ------------------------------------------------------------
# State-Definition
# ------------------------------------------------------------

class ShitstormState(TypedDict, total=False):
    platform: str
    cause: str
    brand: str

    round: int
    intensity: int  # 0–100

    last_company_response: Optional[str]
    politeness_score: Optional[float]
    responsibility_score: Optional[float]
    last_quality_score: Optional[float]

    comments: List[str]          # aktuelle Community-Kommentare (mit Namen/Handles im Text)
    history: List[str]           # Log der Simulation

    status: Literal["running", "user_won", "user_lost", "open"]
    summary: str                 # finale Zusammenfassung


# ------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------

def _clamp_intensity(value: float) -> int:
    return max(0, min(100, int(round(value))))


def _split_lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


# ------------------------------------------------------------
# Nodes
# ------------------------------------------------------------

def init_shitstorm(state: ShitstormState) -> ShitstormState:
    """Startzustand: Plattform, Ursache, Marke + erste Community-Kommentare."""
    platform = state.get("platform") or "Instagram"
    cause = state.get("cause") or "Produktfehler"
    brand = state.get("brand") or "Beispiel GmbH"
    intensity = state.get("intensity", 65)

    sys = SystemMessage(
        content=(
            "Du bist eine wütende Social-Media-Community. "
            "Du schreibst authentische Kommentare im Stil der gewählten Plattform. "
            "Jeder Kommentar soll so aussehen: 'Name (@handle): kurzer Kommentar'. "
            "Keine Aufzählungszeichen, keine zusätzlichen Erklärungen."
        )
    )

    user = HumanMessage(
        content=(
            f"Plattform: {platform}\n"
            f"Unternehmen: {brand}\n"
            f"Ursache des Shitstorms: {cause}\n\n"
            "Erzeuge 5 kurze, emotionale Kommentare der Community.\n"
            "Format: eine Zeile pro Kommentar wie 'Name (@handle): Kommentar'."
        )
    )

    resp = llm.invoke([sys, user])
    comments = _split_lines(resp.content)

    history = state.get("history") or []
    history.append(
        f"Runde 1 gestartet. Plattform: {platform}, Ursache: {cause}, "
        f"Start-Intensität: {intensity}/100."
    )
    history.append("Erste Community-Kommentare generiert.")

    return {
        "platform": platform,
        "cause": cause,
        "brand": brand,
        "round": 1,
        "intensity": intensity,
        "comments": comments,
        "history": history,
        "status": "running",
    }


def wait_for_company_response(state: ShitstormState) -> ShitstormState:
    """
    Human-in-the-loop: Hier pausiert die Ausführung mit `interrupt`
    und wartet auf den Social-Media-Post des Unternehmens.
    """
    round_no = state.get("round", 1)
    intensity = state.get("intensity", 50)
    comments = state.get("comments", [])

    preview_comments = "\n".join(comments[:5]) or "(noch keine Kommentare)"

    prompt_text = (
        f"Runde {round_no} – aktuelle Shitstorm-Intensität: {intensity}/100.\n\n"
        "Aktuelle Community-Kommentare (Auszug):\n"
        f"{preview_comments}\n\n"
        "Schreibe jetzt deinen Social-Media-Post als Unternehmen, "
        "um den Shitstorm abzuschwächen oder zu beenden."
    )

    # In LangGraph Studio / per API bekommst du dieses Objekt zurück.
    # Zum Fortsetzen musst du den Text (die Unternehmensantwort) als resume-Wert schicken.
    response_from_human = interrupt(
        {
            "type": "company_response",
            "round": round_no,
            "intensity": intensity,
            "message": prompt_text,
        }
    )

    # Beim Resumen wird das der Rückgabewert von interrupt()
    if isinstance(response_from_human, dict) and "text" in response_from_human:
        company_response = str(response_from_human["text"])
    else:
        company_response = str(response_from_human)

    history = state.get("history") or []
    history.append(f"Runde {round_no}: Unternehmensreaktion gepostet:\n{company_response}")

    return {
        "last_company_response": company_response,
        "history": history,
    }


def evaluate_response(state: ShitstormState) -> ShitstormState:
    """Bewertet die Antwort anhand von folgenden Kriterien(0–100): 
    Authentisch
    Professionell
    Verifiziert und transparent
    Positiv & lösungsorintiert
    Ganzheitlich & einheitlich
    ."""
    company_response = state.get("last_company_response") or ""
    platform = state.get("platform", "Social Media")
    cause = state.get("cause", "")
    comments_preview = "\n".join(state.get("comments", [])[:5])

    sys = SystemMessage(
        content=(
            "Du bist Expert:in für Krisenkommunikation und bewertest Antworten von Unternehmen "
            "auf Shitstorms. Antworte NUR mit gültigem JSON."
        )
    )
    user = HumanMessage(
        content=(
            f"Bewerte die folgende Antwort eines Unternehmens auf einen Shitstorm.\n\n"
            f"Plattform: {platform}\n"
            f"Ursache des Shitstorms: {cause}\n\n"
            f"Aktuelle Community-Kommentare (Auszug):\n{comments_preview}\n\n"
            "Antwort des Unternehmens:\n"
            f"\"\"\"{company_response}\"\"\"\n\n"
            "Bewerte auf einer Skala von 0 bis 100:\n"
            "- politeness: Wie höflich/respektvoll/ruhig ist die Antwort?\n"
            "- responsibility: Wie klar übernimmt das Unternehmen Verantwortung, "
            "zeigt Einsicht und bietet Lösungen?\n\n"
            "Gib die Antwort ausschließlich als JSON in diesem Format zurück:\n"
            "{\n"
            '  "politeness": <Zahl 0-100>,\n'
            '  "responsibility": <Zahl 0-100>,\n'
            '  "short_feedback": "<sehr kurze deutsche Text-Zusammenfassung>"\n'
            "}\n"
        )
    )

    resp = llm.invoke([sys, user])
    text = resp.content

    try:
        data = json.loads(text)
        politeness = float(data.get("politeness", 0))
        responsibility = float(data.get("responsibility", 0))
        feedback = str(data.get("short_feedback", "")).strip()
    except Exception:
        politeness = 0.0
        responsibility = 0.0
        feedback = "Fehler beim Parsen der Bewertung – Scores auf 0 gesetzt."

    quality = (politeness + responsibility) / 2.0

    history = state.get("history") or []
    history.append(
        f"Bewertung Runde {state.get('round', 1)} – "
        f"Höflichkeit: {politeness:.1f}, Verantwortung: {responsibility:.1f}, "
        f"Gesamtscore: {quality:.1f}. Feedback: {feedback}"
    )

    return {
        "politeness_score": politeness,
        "responsibility_score": responsibility,
        "last_quality_score": quality,
        "history": history,
    }


def update_intensity(state: ShitstormState) -> ShitstormState:
    """Passt die Shitstorm-Intensität basierend auf dem Qualitätsscore an."""
    intensity = state.get("intensity", 50)
    quality = state.get("last_quality_score", 50.0) or 50.0

    if quality >= 75:
        delta = random.randint(-28, -18)
        verdict = "Die Community reagiert überwiegend positiv – der Shitstorm flaut deutlich ab."
    elif quality >= 60:
        delta = random.randint(-18, -8)
        verdict = "Die Reaktion kommt gut an – der Shitstorm nimmt spürbar ab."
    elif quality >= 40:
        delta = random.randint(-5, 5)
        verdict = "Die Reaktion polarisiert – einige sind zufrieden, andere nicht."
    elif quality >= 25:
        delta = random.randint(5, 15)
        verdict = "Viele sind unzufrieden – der Shitstorm legt wieder zu."
    else:
        delta = random.randint(15, 25)
        verdict = "Die Antwort gießt Öl ins Feuer – der Shitstorm eskaliert deutlich."

    new_intensity = _clamp_intensity(intensity + delta)

    history = state.get("history") or []
    history.append(
        f"Shitstorm-Intensität verändert sich um {delta:+d} Punkte auf {new_intensity}/100. {verdict}"
    )

    return {
        "intensity": new_intensity,
        "history": history,
    }


def route_after_update(state: ShitstormState) -> Literal["community_round", "summarize"]:
    """Router: weiter simuliern oder beenden?"""
    intensity = state.get("intensity", 50)
    round_no = state.get("round", 1)

    # Harte Grenze auf Rundenanzahl, damit nichts unendlich läuft
    if round_no >= 10:
        return "summarize"

    if intensity < 10 or intensity > 90:
        return "summarize"

    return "community_round"


def community_round(state: ShitstormState) -> ShitstormState:
    """Neue Community-Kommentare auf Basis der letzten Antwort + Intensität generieren."""
    round_no = state.get("round", 1) + 1
    platform = state.get("platform", "Social Media")
    cause = state.get("cause", "")
    brand = state.get("brand", "das Unternehmen")
    intensity = state.get("intensity", 50)
    quality = state.get("last_quality_score", 50.0) or 50.0
    last_response = state.get("last_company_response") or ""

    if quality >= 70:
        tone = "freundlicher und konstruktiver"
    elif quality >= 45:
        tone = "gemischt – teils konstruktiv, teils kritisch"
    else:
        tone = "härter, zynischer und wütender"

    sys = SystemMessage(
        content=(
            "Du bist die Social-Media-Community und reagierst auf die Krisenkommunikation "
            "eines Unternehmens. Du schreibst authentische Kommentare mit Namen und Handles.\n"
            "Format: 'Name (@handle): Kommentar', eine Zeile pro Kommentar. "
            "Keine Aufzählungszeichen, keine extra Erklärungen."
        )
    )

    user = HumanMessage(
        content=(
            f"Plattform: {platform}\n"
            f"Unternehmen: {brand}\n"
            f"Ursache des Shitstorms: {cause}\n"
            f"Aktuelle Shitstorm-Intensität: {intensity}/100\n\n"
            f"Letzte Unternehmensreaktion:\n\"\"\"{last_response}\"\"\"\n\n"
            f"Erzeuge 4 neue Community-Kommentare, die {tone} sind.\n"
            "Format: eine Zeile pro Kommentar wie 'Name (@handle): Kommentar'."
        )
    )

    resp = llm.invoke([sys, user])
    comments = _split_lines(resp.content)

    history = state.get("history") or []
    history.append(
        f"Neue Community-Kommentare für Runde {round_no} generiert ({len(comments)} Stück)."
    )

    return {
        "round": round_no,
        "comments": comments,
        "history": history,
        "status": "running",
    }


def summarize(state: ShitstormState) -> ShitstormState:
    """Abschluss: Gewinner/Verlierer bestimmen + kurze Zusammenfassung generieren."""
    intensity = state.get("intensity", 0)
    platform = state.get("platform", "Social Media")
    cause = state.get("cause", "")
    brand = state.get("brand", "das Unternehmen")
    rounds = state.get("round", 1)
    history = state.get("history", [])

    if intensity < 10:
        status: Literal["user_won", "user_lost", "open"] = "user_won"
        outcome = "Der Shitstorm ist weitgehend abgeflaut – starke Performance."
    elif intensity > 90:
        status = "user_lost"
        outcome = "Der Shitstorm ist außer Kontrolle geraten."
    else:
        status = "open"
        outcome = "Die Situation ist nicht eindeutig, aber die Übung wird hier beendet."

    log_excerpt = "\n".join(history[-8:])

    sys = SystemMessage(
        content=(
            "Du bist Trainer:in für Krisenkommunikation und fasst die Übung kurz zusammen."
        )
    )
    user = HumanMessage(
        content=(
            "Fasse die folgende Shitstorm-Simulation in 3–5 Sätzen zusammen.\n"
            "Gehe darauf ein:\n"
            "- wie sich die Intensität entwickelt hat,\n"
            "- was an den Antworten gut / schlecht war,\n"
            "- welche Learnings die Person mitnehmen sollte.\n\n"
            f"Meta-Infos:\n"
            f"- Plattform: {platform}\n"
            f"- Unternehmen: {brand}\n"
            f"- Ursache: {cause}\n"
            f"- Finale Intensität: {intensity}/100\n"
            f"- Runden: {rounds}\n"
            f"- Ergebnis: {outcome}\n\n"
            f"Verlauf (Auszug):\n\"\"\"{log_excerpt}\"\"\""
        )
    )

    resp = llm.invoke([sys, user])
    summary_text = resp.content.strip()

    history.append(f"Finale Zusammenfassung: {summary_text}")

    return {
        "status": status,
        "summary": summary_text,
        "history": history,
    }


# ------------------------------------------------------------
# Graph bauen & kompilieren
# ------------------------------------------------------------

builder = StateGraph(ShitstormState)

builder.add_node("init_shitstorm", init_shitstorm)
builder.add_node("wait_for_response", wait_for_company_response)
builder.add_node("evaluate_response", evaluate_response)
builder.add_node("update_intensity", update_intensity)
builder.add_node("community_round", community_round)
builder.add_node("summarize", summarize)

builder.add_edge(START, "init_shitstorm")
builder.add_edge("init_shitstorm", "wait_for_response")
builder.add_edge("wait_for_response", "evaluate_response")
builder.add_edge("evaluate_response", "update_intensity")

builder.add_conditional_edges(
    "update_intensity",
    route_after_update,
    {
        "community_round": "community_round",
        "summarize": "summarize",
    },
)

builder.add_edge("community_round", "wait_for_response")
builder.add_edge("summarize", END)

# Für Interrupts wird ein Checkpointer benötigt.
memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)
