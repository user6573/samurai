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


# Marker, der aus x.html bei Reaktionszeit-Timeout als "Antwort" geschickt wird
TIMEOUT_MARKER_PREFIX = "[AUTOMATISCHE COMMUNITY-REAKTION"


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

    # Erkennen, ob wir im X/Twitter-Interface sind
    is_x = False
    if isinstance(platform, str):
        pl = platform.lower()
        is_x = pl.startswith("x") or "twitter" in pl

    # Beschreibung der Situation im Verlauf
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

    # --- Timeout-Analyse: wurde die Ausführung wegen abgelaufener Reaktionszeit fortgesetzt? ---
    company_events = [
        h for h in state.get("history", []) if h.get("actor") == "company"
    ]

    last_is_timeout = False
    if company_events:
        last_content = str(company_events[-1].get("content", ""))
        last_is_timeout = last_content.startswith(TIMEOUT_MARKER_PREFIX)

    had_real_company_response_before = any(
        isinstance(ev.get("content"), str)
        and not str(ev.get("content")).startswith(TIMEOUT_MARKER_PREFIX)
        for ev in company_events[:-1]
    )

    if last_is_timeout and not had_real_company_response_before:
        # Noch nie eine echte Unternehmensantwort -> Community beschwert sich über Schweigen
        timeout_mode: Literal["no_response", "after_response", "none"] = "no_response"
    elif last_is_timeout and had_real_company_response_before:
        # Es gab schon mind. eine echte Antwort -> Community fragt sich, ob das alles war
        timeout_mode = "after_response"
    else:
        timeout_mode = "none"

    # Welcher Post hängt direkt über den Kommentaren?
    if timeout_mode == "no_response":
        target_post_type = (
            "Beschwerde-Post einer Nutzerin / eines Nutzers "
            "(das Unternehmen hat bisher nicht öffentlich reagiert)"
        )
        target_post_text = cause
    elif timeout_mode == "after_response":
        # Letzte echte Unternehmensantwort aus der History suchen
        real_company_answer = None
        for ev in reversed(company_events):
            txt = str(ev.get("content", ""))
            if not txt.startswith(TIMEOUT_MARKER_PREFIX):
                real_company_answer = txt
                break

        target_post_type = (
            "öffentliche Antwort des Unternehmens "
            "(Community wartet auf weitere konkrete Reaktionen)"
        )
        target_post_text = real_company_answer or last_answer or cause
    else:
        # Normalfall ohne Timeout-Fokus
        if round_num == 1:
            target_post_type = "Beschwerde-Post einer Nutzerin / eines Nutzers"
            target_post_text = cause
        else:
            target_post_type = "öffentliche Antwort des Unternehmens"
            target_post_text = last_answer or cause

    # Kurzer Auszug aus der bisherigen History, damit die Replies konsistenter werden
    recent_events: List[str] = []
    for h in state.get("history", [])[-6:]:
        actor = h.get("actor", "?")
        content = str(h.get("content", ""))
        if len(content) > 160:
            content = content[:160] + "…"
        recent_events.append(f"- {actor}: {content}")
    recent_text = "\n".join(recent_events) if recent_events else "noch keine relevanten Einträge"

    # Grund-Prompt
    system_content = (
        "Du simulierst eine Kommentarspalte in einem Social-Media-Shitstorm.\n"
        "Schreibe auf Deutsch, im typischen Ton der jeweiligen Plattform.\n"
        "Erzeuge realistische wenn nötig beleidigende Kommentare.\n"
        "Du schreibst NUR Community-Kommentare, NIEMALS die Antwort des Unternehmens.\n"
        "Jeder Kommentar ist eine einzelne, eigenständige Antwort (kein Dialog, keine langen Threads).\n"
        "Antwort NUR als JSON-Liste von Strings, z.B.:\n"
        '  [\"Kommentar 1\", \"Kommentar 2\", \"...\"]\n'
        "Kein zusätzlicher Text, keine Erklärungen, keine JSON-Objekte.\n"
        "Achte auf Varianz: Mindestens eine starke Kritik, eine sachlich-konstruktive Stimme "
        "SEHR WICHTIG:\n"
        "- Die Kommentare stehen direkt unter EINEM konkreten Post.\n"
        "- Der Hauptpunkt jedes Kommentars muss sich klar auf GENAU diesen Post beziehen "
        "(Inhalt, Ton, Versprechen oder Lücken dieses Posts).\n"
        "- Schreibe KEINE völlig allgemeinen Aussagen über das Unternehmen, sondern reagiere "
        "auf das, was in diesem Post steht oder NICHT steht.\n"
    )

    # Spezieller Stil, wenn wir im X-Interface sind
    if is_x:
        system_content += (
            "\nSpezifisch für die Plattform X/Twitter:\n"
            "- Du simulierst die „Antworten“-Sektion unter einem Post.\n"
            "- Schreibe kurze, pointierte Kommentare (max. ca. 200 Zeichen).\n"
            "- Ton: wie typische X-Replies – direkt, emotional, manchmal sarkastisch, gerne auch ironisch.\n"
            "- Du kannst gelegentlich Emojis oder Ironie nutzen, aber übertreibe nicht.\n"
            "- Keine @Handles oder Namen im Kommentartext, die UI zeigt Namen/Handles separat.\n"
            "- Keine Hashtags-Spam, maximal 0–2 Hashtags pro Kommentar.\n"
            "- Jeder Kommentar soll deutlich machen, dass er sich auf GENAU diesen Post bezieht "
                "\nWICHTIG:\n"
            "- Erzeuge Kommentare, die NICHT identisch oder fast identisch mit früheren Kommentaren sind.\n"
            "- Keine Wiederholungen, keine wiederverwendeten Formulierungen.\n"
            "- Jeder Kommentar MUSS neu und einzigartig klingen.\n"
            "- Formuliere jedes Mal neue Kritik, neue Perspektiven oder neue Nuancen.\n"
            "(z.B. durch Formulierungen wie „diese Antwort“, „das hier“, „euer Statement oben“ usw.).\n"
        )

    system_msg = SystemMessage(content=system_content)

    # Timeout-spezifische Zusatzinstruktionen
    if timeout_mode == "no_response":
        extra_timeout_instr = (
            "\nZusätzlicher Fokus: Die Community ist frustriert, dass das Unternehmen bisher GAR NICHT "
            "öffentlich reagiert hat. Die Kommentare kritisieren vor allem:\n"
            "- Schweigen und Nicht-Reagieren\n"
            "- zu lange Reaktionszeiten\n"
            "- das Gefühl, ignoriert zu werden\n"
            "Trotzdem: keine Beleidigungen, keine Diskriminierung.\n"
        )
    elif timeout_mode == "after_response":
        extra_timeout_instr = (
            "\nZusätzlicher Fokus: Die Community fragt sich, ob das wirklich alles war. "
            "Die Kommentare kritisieren vor allem:\n"
            "- dass nach die vorherige Antwort zu wenig war\n"
            "Formuliere das kritisch, gerne auch zugespitzt, aber ohne Beleidigungen oder Diskriminierung.\n"
        )
    else:
        extra_timeout_instr = ""

    human_msg = HumanMessage(
        content=(
            f"Plattform: {platform}\n"
            f"Unternehmen: {company_name}\n"
            f"Ursache des Shitstorms: {cause}\n"
            f"Aktuelle Shitstorm-Intensität (0-100): {intensity}\n"
            f"Runde: {round_num}\n"
            f"Situation: {situation_desc}\n\n"
            "Der folgende Post steht direkt über der Kommentarspalte, die du simulierst:\n"
            f"Art des Posts: {target_post_type}\n"
            f"Post-Inhalt:\n\"\"\"{target_post_text}\"\"\"\n\n"
            "Relevante Ausschnitte aus dem bisherigen Verlauf:\n"
            f"{recent_text}\n"
            f"{extra_timeout_instr}\n"
            "Generiere 6 kurze Kommentare der Community, die sich klar und hauptsächlich "
            "auf diesen einen Post beziehen. Mische:\n"
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
        entry: Dict[str, Any] = {
            "actor": "community",
            "round": round_num,
            "content": c,
        }
        # Meta-Flag, damit man im Verlauf später erkennen kann,
        # dass diese Kommentare als X-Replies gedacht waren.
        if is_x:
            entry["section"] = "x_replies"
        state["history"].append(entry)

    return state


def company_response_node(state: ShitstormState) -> ShitstormState:
    """Human-in-the-loop Node für die Unternehmensantwort."""
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
    """Bewertet die Unternehmensantwort anhand der definierten Kriterien."""
    platform = state["platform"]
    cause = state["cause"]
    company_name = state["company_name"]
    answer = state["last_company_response"]

    system_msg = SystemMessage(
        content=(
            "Du bist ein professioneller Coach für Krisenkommunikation in sozialen Medien.\n"
            "Bewerte Antworten von Unternehmen in Shitstorms NUR anhand der folgenden Kriterien:\n"
            "1) Präzise & vollständig: Text mit so viel Inhalt wie notwendig, aber nicht mehr; "
            "   möglichst kein Spielraum zur Interpretation.\n"
            "2) Was ist passiert? – Klar erklärt, was konkret vorgefallen ist.\n"
            "3) Statement / Entschuldigung – Klares Statement und ggf. ehrliche Entschuldigung.\n"
            "4) Wer sagt das? – Absender / Verantwortliche Person oder Funktion ist eindeutig.\n"
            "5) Lösung – Es ist klar, was das Unternehmen als Lösung / nächste Schritte anbietet.\n"
            "7) Authentisch – Wirkt ehrlich, nicht wie reine PR-Floskel.\n"
            "8) Professionell – Ton und Struktur sind respektvoll, klar und angemessen.\n"
            "9) Verifiziert & transparent – Offenheit über Fakten, Status, Prüfungen, Zahlen, Hintergründe.\n"
            "10) Positiv & lösungsorientiert – Fokus auf Lösungen und Verbesserung statt Abwehr.\n"
            "11) Ganzheitlich & einheitlich – Die Antwort ist in sich stimmig, widerspricht sich nicht "
            "    und adressiert die wichtigsten Punkte der Kritik.\n\n"
            "Du antwortest ausschließlich als JSON-Objekt, ohne zusätzlichen Text."
        )
    )

    human_msg = HumanMessage(
        content=(
            "Bewerte die folgende Antwort eines Unternehmens in einem Shitstorm.\n\n"
            f"Plattform: {platform}\n"
            f"Unternehmen: {company_name}\n"
            f"Ursache des Shitstorms: {cause}\n\n"
            f"Antwort des Unternehmens:\n\"\"\"{answer}\"\"\"\n\n"
            "Bewerte, inwiefern die Antwort die oben genannten Kriterien erfüllt.\n"
            "Gib deine Antwort NUR als gültiges JSON im folgenden Format zurück:\n"
            "{\n"
            '  \"overall\": <Zahl 0-100>,\n'
            '  \"criteria\": {\n'
            '    \"praezise_ohne_spielraum\": true/false,\n'
            '    \"klar_was_passiert\": true/false,\n'
            '    \"statement_oder_entschuldigung\": true/false,\n'
            '    \"wer_sagt_das\": true/false,\n'
            '    \"loesung_angeboten\": true/false,\n'
            '    \"schnell\": true/false,\n'
            '    \"authentisch\": true/false,\n'
            '    \"professionell\": true/false,\n'
            '    \"verifiziert_transparent\": true/false,\n'
            '    \"positiv_loesungsorientiert\": true/false,\n'
            '    \"ganzheitlich_einheitlich\": true/false\n'
            "  },\n"
            '  \"all_criteria_met\": true/false,\n'
            '  \"missing_criteria\": [\"...\"],\n'
            '  \"feedback\": \"Kurzes, konkretes Feedback dazu, welche Kriterien gut erfüllt sind '
            "und welche noch verbessert werden sollten.\"\n"
            "}\n"
        )
    )

    result = llm.invoke([system_msg, human_msg])
    data = _safe_load_json(result.content) or {}

    criteria: Dict[str, Any] = data.get("criteria") or {}
    criteria_total = len(criteria) if isinstance(criteria, dict) and criteria else 11

    fulfilled_count = 0
    if isinstance(criteria, dict):
        fulfilled_count = sum(1 for v in criteria.values() if bool(v))

    all_criteria_met = bool(data.get("all_criteria_met"))
    if criteria and "all_criteria_met" not in data:
        all_criteria_met = all(bool(v) for v in criteria.values())

    missing_criteria = data.get("missing_criteria") or []
    if not isinstance(missing_criteria, list):
        missing_criteria = [str(missing_criteria)]

    # Score, den wir für UI/Debug verwenden: %-Anteil erfüllter Kriterien
    if criteria_total > 0:
        reaction_score = 100.0 * fulfilled_count / criteria_total
    else:
        try:
            reaction_score = float(data.get("overall", 50.0))
        except (TypeError, ValueError):
            reaction_score = 50.0

    feedback = data.get("feedback") or "Keine detaillierte Rückmeldung verfügbar."

    # Alte Felder behalten wir für Kompatibilität, nutzen sie aber als Proxy für die Kriterienerfüllung
    state["politeness_score"] = reaction_score
    state["responsibility_score"] = reaction_score
    state["reaction_score"] = reaction_score

    # Neue interne Felder für die Intensitätslogik
    state["criteria_all_met"] = all_criteria_met
    state["criteria_missing"] = missing_criteria
    state["criteria_fulfilled_count"] = fulfilled_count
    state["criteria_total"] = criteria_total

    missing_text = ", ".join(missing_criteria) if missing_criteria else "keine (alle erfüllt)"

    state["history"].append(
        {
            "actor": "coach",
            "round": state["round"],
            "content": (
                f"Kriterien erfüllt: {fulfilled_count}/{criteria_total}. "
                f"Fehlende Kriterien: {missing_text}. "
                f"Feedback: {feedback}"
            ),
        }
    )

    return state


def update_intensity(state: ShitstormState) -> ShitstormState:
    """Aktualisiert die Shitstorm-Intensität basierend auf den Kriterien.

    Logik:
    - Wenn ALLE Kriterien erfüllt sind -> Intensität stark senken, so dass man praktisch gewinnt.
    - Wenn NICHT alle Kriterien erfüllt sind -> Intensität um 10–50 Punkte erhöhen,
      abhängig davon, wie viele Kriterien fehlen.
    """
    prev = state["intensity"]

    all_met = bool(state.get("criteria_all_met"))
    criteria_total = int(state.get("criteria_total") or 0)
    fulfilled = int(state.get("criteria_fulfilled_count") or 0)

    if all_met and criteria_total > 0 and fulfilled >= criteria_total:
        # Perfekte Antwort: Shitstorm bricht fast komplett ab
        new_intensity = min(prev, 5.0)  # max. 5/100
        delta = new_intensity - prev
    else:
        # Nicht alle Kriterien erfüllt -> Shitstorm verschärft sich um +10 bis +50
        if criteria_total > 0:
            missing = max(0, criteria_total - fulfilled)
            missing_ratio = missing / criteria_total
        else:
            missing_ratio = 1.0

        # Lineare Skalierung: 10 + (0..1)*40 -> 10..50
        delta = 10.0 + missing_ratio * 40.0
        new_intensity = max(0.0, min(100.0, prev + delta))

    state["intensity"] = new_intensity

    if new_intensity < 10.0:
        state["status"] = "user_won"
    elif new_intensity > 90.0:
        state["status"] = "user_lost"
    else:
        state["status"] = "running"

    criteria_total_safe = criteria_total if criteria_total > 0 else 0
    state["history"].append(
        {
            "actor": "system",
            "round": state["round"],
            "content": (
                f"Intensität von {prev:.1f} auf {new_intensity:.1f} geändert "
                f"(Delta={delta:+.1f}, Kriterien erfüllt: {fulfilled}/{criteria_total_safe}, "
                f"alle_criteria_ertefüllt={all_met})."
            ),
        }
    )

    return state


def route_after_update(state: ShitstormState) -> str:
    """Bestimmt, ob weiter simuliert oder beendet wird."""
    if state["status"] == "running":
        return "continue"
    return "end"


def summarize(state: ShitstormState, llm: ChatOpenAI) -> ShitstormState:
    """Erzeugt eine kurze Zusammenfassung des Verlaufs auf Basis der Kriterien."""
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
            "Bewerte insbesondere, wie gut die Antworten des Unternehmens folgende Kriterien erfüllt haben:\n"
            "- präzise & eindeutig (kein unnötiger Ballast, wenig Spielraum für Interpretation)\n"
            "- klar erklärt: Was ist passiert?\n"
            "- Statement / Entschuldigung\n"
            "- klare Absender-Rolle (wer spricht?)\n"
            "- konkrete Lösung / nächste Schritte\n"
            "- schnell, authentisch, professionell\n"
            "- verifiziert & transparent\n"
            "- positiv & lösungsorientiert\n"
            "- ganzheitlich & einheitlich\n\n"
            "Fasse den Verlauf des Shitstorms und das Verhalten des Users zusammen.\n"
            "Gib konkrete Lernpunkte und Verbesserungsvorschläge, strukturiert an diesen Kriterien.\n"
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
