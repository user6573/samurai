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

# Marker, den dein Frontend bei Timeout als "Antwort" setzt
TIMEOUT_MARKER_PREFIX = "[AUTOMATISCHE COMMUNITY-REAKTION"


class ShitstormState(TypedDict):
    """Gesamter Zustand der Shitstorm-Simulation.

    Alle Keys müssen JSON-serialisierbar sein.
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
    # Dynamische Felder (werden zur Laufzeit ergänzt):
    # criteria_all_met: bool
    # criteria_missing: List[str]
    # criteria_fulfilled_count: int
    # criteria_total: int


# --------------------------------------------------------------------------- #
# Hilfsfunktionen
# --------------------------------------------------------------------------- #

def _safe_load_json(text: str):
    """Versuche, robust JSON aus einem LLM-Output zu laden."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Objekt
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Liste
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


def _make_llm() -> ChatOpenAI:
    """Erzeuge das LLM für die Simulation.

    Das Modell kann über die Umgebungsvariable OPENAI_MODEL überschrieben werden.
    """
    model_name = os.getenv("OPENAI_MODEL", "gpt-5.1")
    return ChatOpenAI(model=model_name, temperature=0.3)


# --------------------------------------------------------------------------- #
# Community-Runde – harte, kritische, anti-duplizierte Kommentare
# --------------------------------------------------------------------------- #
def community_round(state: ShitstormState, llm: ChatOpenAI) -> ShitstormState:
    """Generiert Community-Kommentare für die aktuelle Runde.

    Die Härte / Gemeinheit der Kommentare hängt von der letzten
    Intensitätsänderung (Δ) ab, die im update_intensity-Node gesetzt wird.
    """

    state["round"] += 1
    round_num = state["round"]

    platform = state["platform"]
    cause = state["cause"]
    company_name = state["company_name"]

    # Start-Intensität für Runde 1 immer auf 50 setzen
    prev_intensity = float(state.get("intensity", 50.0))
    if round_num == 1:
        state["intensity"] = 50.0
    intensity = float(state.get("intensity", prev_intensity))

    last_answer = state.get("last_company_response", "")
    reaction_score = float(state.get("reaction_score", 50.0))

    # Δ-Intensität aus der letzten Runde (durch den letzten Post verursacht)
    last_delta = float(state.get("last_intensity_delta", 0.0))

    # Erkennen, ob X/Twitter-Interface
    is_x = False
    if isinstance(platform, str):
        pl = platform.lower()
        is_x = pl.startswith("x") or "twitter" in pl

    # Beschreibung der Situation im Verlauf (für das LLM)
    if round_num == 1:
        situation_desc = (
            "Dies sind die ersten Reaktionen der Community auf den Auslöser. "
            "Es gibt noch keine offizielle Antwort des Unternehmens."
        )
    else:
        if last_delta <= -15:
            situation_desc = (
                "Die letzte Antwort des Unternehmens hat den Shitstorm deutlich beruhigt. "
                "Viele nehmen wahr, dass konkrete Verantwortung übernommen wurde und sich wirklich etwas bewegt."
            )
        elif last_delta <= -5:
            situation_desc = (
                "Die letzte Antwort des Unternehmens hat die Lage spürbar entspannt. "
                "Die Community erkennt Fortschritte und ernsthafte Bemühungen an."
            )
        elif last_delta < 5:
            situation_desc = (
                "Die letzte Antwort des Unternehmens hat kaum etwas an der Situation verändert. "
                "Teile der Community sind unzufrieden, andere abwartend."
            )
        elif last_delta < 15:
            situation_desc = (
                "Die letzte Antwort des Unternehmens hat den Shitstorm eher verschärft. "
                "Viele sind noch unzufriedener und sehen zu wenig echte Veränderung."
            )
        else:
            situation_desc = (
                "Die letzte Antwort des Unternehmens hat die Lage stark eskalieren lassen. "
                "Die Community reagiert sehr wütend und extrem misstrauisch."
            )

    # Härte-Level aus Δ ableiten
    if round_num == 1:
        severity = "initial"
    elif last_delta <= -15:
        severity = "strong_decrease"
    elif last_delta <= -5:
        severity = "mild_decrease"
    elif last_delta < 5:
        severity = "neutral"
    elif last_delta < 15:
        severity = "mild_increase"
    else:
        severity = "strong_increase"

    # Ton / Mischung je nach Schweregrad
    if severity == "initial":
        tone_instruction = (
            "Die Community reagiert zum ersten Mal auf den Auslöser. "
            "Die Kommentare sind deutlich kritisch, wütend und fordernd."
        )
        comment_mix_hint = (
            "Erzeuge überwiegend harte, kritische Kommentare, die deutlich machen, "
            f"dass {company_name} mit der Kampagne / dem Vorfall eine Grenze überschritten hat."
        )
        positive_mode = False
    elif severity == "strong_decrease":
        tone_instruction = (
            "Die letzte Antwort und die Maßnahmen von "
            f"{company_name} haben den Shitstorm deutlich beruhigt. "
            "Viele in der Community empfinden das Statement als klare, längst fällige Klarstellung "
            "und sehen eine echte Bereitschaft zur Veränderung."
        )
        comment_mix_hint = (
            "Erzeuge überwiegend leicht positive, dankbare und zukunftsorientierte Kommentare. "
            f"Die Leute bedanken sich explizit für das Statement bzw. die Klarstellung von {company_name}, "
            "loben die konkreten Schritte und blicken vorsichtig positiv in die Zukunft "
            "(z.B. Hoffnung, dass es jetzt wirklich besser wird). "
            "Formuliere KEINE neuen Vorwürfe und stelle NICHT grundsätzlich infrage, "
            f"ob {company_name} es ernst meint."
        )
        positive_mode = True
    elif severity == "mild_decrease":
        tone_instruction = (
            "Die Community erkennt Fortschritte an und nimmt das Statement von "
            f"{company_name} überwiegend positiv wahr, auch wenn noch nicht alles perfekt ist."
        )
        comment_mix_hint = (
            "Erzeuge überwiegend konstruktive Kommentare, die sich für die Klarstellung und die ersten Schritte "
            f"von {company_name} bedanken. Einige Kommentare dürfen freundlich darauf hinweisen, "
            "dass bestimmte Punkte weiter beobachtet oder nachgehalten werden sollten. "
            "Die Grundstimmung ist: ‚Danke, guter Anfang, bitte dranbleiben.‘"
        )
        positive_mode = True
    elif severity == "neutral":
        tone_instruction = (
            "Die letzte Antwort hat kaum etwas verändert. Die Stimmung ist gemischt."
        )
        comment_mix_hint = (
            "Erzeuge eine Mischung aus nüchtern-kritischen, skeptischen und wenigen neutralen Kommentaren. "
            "Es gibt weder klare Entspannung noch massive Verschärfung."
        )
        positive_mode = False
    elif severity == "mild_increase":
        tone_instruction = (
            "Die Community ist eher noch kritischer geworden. Die Antwort wirkt vielen zu schwach."
        )
        comment_mix_hint = (
            "Erzeuge überwiegend harte, kritische Kommentare, die deutlich machen, "
            f"dass {company_name} noch zu wenig Verantwortung übernimmt oder zu vage bleibt."
        )
        positive_mode = False
    else:  # strong_increase
        tone_instruction = (
            "Die letzte Antwort hat großen Frust ausgelöst. Die Community fühlt sich nicht ernst genommen."
        )
        comment_mix_hint = (
            "Erzeuge sehr scharfe, deutlich ablehnende Kommentare, die großes Misstrauen "
            f"gegenüber {company_name} ausdrücken. Die Formulierungen dürfen sehr hart und sarkastisch sein, "
            "aber ohne Gewaltaufrufe und ohne diskriminierende oder hasserfüllte Sprache."
        )
        positive_mode = False

    # --- Timeout-Analyse -----------------------------------------------------
    company_events = [h for h in state.get("history", []) if h.get("actor") == "company"]

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
        timeout_mode: Literal["no_response", "after_response", "none"] = "no_response"
    elif last_is_timeout and had_real_company_response_before:
        timeout_mode = "after_response"
    else:
        timeout_mode = "none"

    # --- Welcher Post hängt direkt über den Kommentaren? ---------------------
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
        if round_num == 1:
            target_post_type = "Beschwerde-Post einer Nutzerin / eines Nutzers"
            target_post_text = cause
        else:
            target_post_type = "öffentliche Antwort des Unternehmens"
            target_post_text = last_answer or cause

    # --- Kurzer Auszug aus der bisherigen History ---------------------------
    recent_events: List[str] = []
    for h in state.get("history", [])[-6:]:
        actor = h.get("actor", "?")
        content = str(h.get("content", ""))
        if len(content) > 160:
            content = content[:160] + "…"
        recent_events.append(f"- {actor}: {content}")
    recent_text = "\n".join(recent_events) if recent_events else "noch keine relevanten Einträge"

    # --- Bisherige Community-Kommentare (für Anti-Duplikation) --------------
    previous_replies = [
        str(h.get("content", "")).strip()
        for h in state.get("history", [])
        if h.get("actor") == "community"
    ]
    previous_replies_text = (
        "\n".join(f"- {c}" for c in previous_replies[-25:])
        if previous_replies else "Keine bisherigen Community-Kommentare."
    )

    # --- System-Prompt mit dynamischem Ton -----------------------------------
    if positive_mode:
        system_tone_header = (
            "Du simulierst eine Kommentarspalte in einem Social-Media-Shitstorm.\n"
            "Schreibe auf Deutsch.\n"
            "In dieser Phase hat die letzte Antwort des Unternehmens viel zur Deeskalation beigetragen.\n"
            "Die Kommentare sind überwiegend dankbar, erleichtert und konstruktiv. "
            "Es gibt KEINE neuen Vorwürfe und keine grundsätzlichen Angriffe mehr.\n"
        )
    else:
        system_tone_header = (
            "Du simulierst eine Kommentarspalte in einem Social-Media-Shitstorm.\n"
            "Schreibe auf Deutsch.\n"
            "Der Ton kann kritisch, frustriert, verärgert und sehr hart sein. "
            "Du darfst scharfe Kritik, Sarkasmus und überzogene Formulierungen nutzen, "
            "aber keine Gewaltaufrufe und keine diskriminierende oder hasserfüllte Sprache.\n"
        )

    system_content = (
        system_tone_header
        + f"{tone_instruction}\n"
        "Du schreibst NUR Community-Kommentare, NIEMALS die Antwort des Unternehmens.\n"
        "Jeder Kommentar ist eine einzelne, eigenständige Antwort (kein Dialog, keine langen Threads).\n"
        "Alle Kommentare beziehen sich klar auf den EINEN Post direkt darüber (Inhalt, Ton, Lücken).\n"
        "Antwort NUR als JSON-Liste von Strings, z.B.:\n"
        '  [\"Kommentar 1\", \"Kommentar 2\", \"...\"]\n'
        "Kein zusätzlicher Text, keine Erklärungen, keine JSON-Objekte.\n"
        "WICHTIG:\n"
        "- Erzeuge Kommentare, die NICHT identisch oder fast identisch mit früheren "
        "Community-Kommentaren sind.\n"
        "- Keine Wiederholungen, keine fast gleichen Formulierungen.\n"
        "- Jeder Kommentar muss neu, frisch und eindeutig formuliert sein.\n"
    )

    if is_x:
        system_content += (
            "\nSpezifisch für die Plattform X/Twitter:\n"
            "- Du simulierst die „Antworten“-Sektion unter einem Post.\n"
            "- Schreibe kurze, pointierte, sehr direkte Kommentare (max. ca. 200 Zeichen).\n"
            "- Stil je nach Lage: von stark kritisch bis zunehmend konstruktiv, "
            "abhängig von der beschriebenen Intensitätsänderung.\n"
            "- Keine @Handles oder Namen im Kommentartext, die UI zeigt Namen/Handles separat.\n"
            "- Keine Hashtag-Spam, maximal 0–2 Hashtags pro Kommentar.\n"
        )

    # Timeout-spezifische Zusatzinstruktionen
    if timeout_mode == "no_response":
        extra_timeout_instr = (
            "\nZusätzlicher Fokus: Das Unternehmen hat TROTZ Shitstorm noch nicht öffentlich reagiert.\n"
            "- Kritisiere explizit das Schweigen, die fehlende Reaktion und das Ignorieren der Community.\n"
            "- Stelle infrage, wie ernst das Unternehmen die Situation wirklich nimmt.\n"
        )
    elif timeout_mode == "after_response":
        extra_timeout_instr = (
            "\nZusätzlicher Fokus: Es gab bereits eine Unternehmensantwort, aber seitdem kommt nichts mehr.\n"
            "- Kommentare im Stil von: 'Das kann nicht alles sein', "
            "'Da fehlen noch konkrete Schritte', 'Das reicht so noch nicht'.\n"
        )
    else:
        extra_timeout_instr = ""

    # Zusatzregel: immer Firmenname, keine Pronomen
    system_msg = SystemMessage(
        content=system_content
        + f"\nZusatzregel:\n"
          f"- Wenn in den Kommentaren über das Unternehmen gesprochen wird, verwende IMMER den exakten Namen „{company_name}“.\n"
          f"- Verwende KEINE allgemeinen Pronomen wie „ihr“, „euch“, „denen“, „die Firma“, „dieser Laden“ oder ähnliche Umschreibungen.\n"
          f"- Formuliere Kritik oder Zustimmung direkt mit dem Namen „{company_name}“.\n"
    )

    # Unterschiedliche Emotion-Hints je nach Modus
    if positive_mode:
        emotional_hint = (
            "Generiere genau 6 kurze Kommentare der Community, die sich ausdrücklich für das Statement, "
            f"die Klarstellung oder die konkreten Schritte von {company_name} bedanken oder sie als wichtigen "
            "Schritt anerkennen. Die Kommentare blicken vorsichtig positiv in die Zukunft und drücken Erleichterung "
            "oder Hoffnung aus. Formuliere KEINE neuen Vorwürfe, "
            "keine zynischen Untertöne und keine Andeutungen, dass 'eh nichts passieren wird'."
        )
    else:
        emotional_hint = (
            "Generiere genau 6 kurze Kommentare der Community, die – je nach Lage – kritisch bis stark ablehnend "
            f"gegenüber {company_name} sein können. Du darfst Frust, Enttäuschung und Wut ausdrücken, "
            "die Formulierungen können sehr hart und sarkastisch sein, "
            "solange sie keine Gewaltaufrufe oder diskriminierende / hasserfüllte Sprache enthalten."
        )

    human_msg = HumanMessage(
        content=(
            f"Plattform: {platform}\n"
            f"Unternehmen: {company_name}\n"
            f"Ursache des Shitstorms: {cause}\n"
            f"Aktuelle Shitstorm-Intensität (0-100): {intensity}\n"
            f"Runde: {round_num}\n"
            f"Letzte Intensitätsänderung (Δ): {last_delta:+.1f} Punkte\n"
            f"Situation: {situation_desc}\n\n"
            "Der folgende Post steht direkt über der Kommentarspalte, die du simulierst:\n"
            f"Art des Posts: {target_post_type}\n"
            f"Post-Inhalt:\n\"\"\"{target_post_text}\"\"\"\n\n"
            "Relevante Ausschnitte aus dem bisherigen Verlauf:\n"
            f"{recent_text}\n\n"
            "Alle bisherigen Community-Kommentare (nicht wiederholen!):\n"
            f"{previous_replies_text}\n"
            f"{extra_timeout_instr}\n\n"
            f"{comment_mix_hint}\n"
            f"{emotional_hint}\n"
            "Jeder Kommentar muss sich klar auf den obenstehenden Post beziehen.\n"
            "Gib deine Antwort NUR als JSON-Liste von Strings zurück."
        )
    )

    result = llm.invoke([system_msg, human_msg])
    comments = _safe_load_json(result.content) or []

    if not isinstance(comments, list) or not comments:
        comments = [
            f"Das klingt alles sehr vage – {company_name} geht keinem Punkt wirklich klar nach.",
            f"{company_name} sagt viel, beantwortet aber kaum eine der wichtigsten Fragen.",
        ]

    # --- HARTE DEDUPLIKATION: keine doppelten Kommentare, weder in dieser Runde
    #     noch im bisherigen Verlauf -----------------------------------------
    normalized_existing = {c.strip() for c in previous_replies if c.strip()}
    unique_round: List[str] = []
    seen_round: set[str] = set()

    for raw in comments:
        c = str(raw).strip()
        if not c:
            continue
        if c in normalized_existing:
            continue  # schon früher in der History
        if c in seen_round:
            continue  # doppelt in dieser Runde
        seen_round.add(c)
        unique_round.append(c)

    # Fallback, falls wirklich alles wegdedupliziert wurde
    if not unique_round:
        if positive_mode:
            unique_round = [
                f"Ich bin ehrlich erleichtert – {company_name} klingt dieses Mal wirklich konkret.",
                f"Respekt, {company_name}. So ein klares Statement hätte ich nicht erwartet.",
            ]
        else:
            unique_round = [
                f"{company_name} redet immer noch um den heißen Brei herum.",
                f"Für mich wirkt das alles noch nicht glaubwürdig, {company_name}.",
            ]

    comments = unique_round

    state["last_community_comments"] = comments
    for c in comments:
        entry: Dict[str, Any] = {
            "actor": "community",
            "round": round_num,
            "content": c,
        }
        if is_x:
            entry["section"] = "x_replies"
        state["history"].append(entry)

    return state




# --------------------------------------------------------------------------- #
# Human-in-the-loop Node für Unternehmensantwort
# --------------------------------------------------------------------------- #

def company_response_node(state: ShitstormState) -> ShitstormState:
    """Human-in-the-loop Node für die Unternehmensantwort (via interrupt)."""
    prompt_payload: Dict[str, Any] = {
        "type": "company_response_request",
        "round": state["round"],
        "platform": state["platform"],
        "cause": state["cause"],
        "company_name": state["company_name"],
        "intensity": state["intensity"],
        "last_community_comments": state["last_community_comments"],
        "hint": (
            "Formuliere eine öffentliche Antwort von "
            f"{state['company_name']}. "
            "Sei höflich, übernimm Verantwortung und biete konkrete Schritte an."
        ),
    }

    resume_value = interrupt(prompt_payload)

    if isinstance(resume_value, dict):
        answer = str(resume_value.get("text", "")).strip()
    else:
        answer = str(resume_value or "").strip()

    if not answer:
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


# --------------------------------------------------------------------------- #
# LLM-Evaluation nach neuen 4 Kriterien
# --------------------------------------------------------------------------- #

def llm_evaluate(state: ShitstormState, llm: ChatOpenAI) -> ShitstormState:
    """Bewertet die Unternehmensantwort anhand von 4 Kern-Kriterien."""
    platform = state["platform"]
    cause = state["cause"]
    company_name = state["company_name"]
    answer = state["last_company_response"]

    system_msg = SystemMessage(
        content=(
            "Du bist ein professioneller Coach für Krisenkommunikation in sozialen Medien.\n"
            "Bewerte Antworten von Unternehmen in Shitstorms NUR anhand dieser vier Kriterien:\n"
            "1) Authentisch – wirkt ehrlich, menschlich.\n"
            "2) Professionell – Ton und Struktur sind respektvoll, klar, verständlich und angemessen.\n"
            "3) Positiv & lösungsorientiert – Fokus auf konkrete Lösungen, nächste Schritte und Verbesserung,\n"
            "   nicht auf Abwehr, Relativierung oder Ausreden.\n"
            "4) Ganzheitlich & einheitlich – die Antwort ist in sich stimmig, widerspricht sich nicht und\n"
            "   adressiert die wichtigsten Punkte der Kritik.\n\n"
            "Bewerte bewusst eher großzügig:\n"
            "- Ein Kriterium gilt als erfüllt (true), wenn es voll ODER überwiegend erfüllt ist,\n"
            "  auch wenn kleinere Lücken oder Schwächen vorhanden sind.\n"
            "- Setze ein Kriterium nur dann auf false, wenn es klar nicht erkennbar erfüllt ist.\n"
            "- In Zweifelsfällen entscheide dich für true.\n\n"
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
            "Bewerte, inwiefern die Antwort die vier Kriterien erfüllt.\n"
            "Gib deine Antwort NUR als gültiges JSON im folgenden Format zurück:\n"
            "{\n"
            '  \"overall\": <Zahl 0-100>,\n'
            '  \"criteria\": {\n'
            '    \"authentisch\": true/false,\n'
            '    \"professionell\": true/false,\n'
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

    # --- overall-Score robust extrahieren -----------------------------------
    try:
        overall_raw = float(data.get("overall", 0.0))
    except (TypeError, ValueError):
        overall_raw = 0.0
    overall = max(0.0, min(100.0, overall_raw))

    # --- Kriterien robust verarbeiten / fallbacken ---------------------------
    raw_criteria = data.get("criteria")
    criteria: Dict[str, bool]

    if isinstance(raw_criteria, dict) and raw_criteria:
        # Modell hat die Struktur eingehalten
        criteria = {k: bool(v) for k, v in raw_criteria.items()}
        criteria_total = len(criteria)
        fulfilled_count = sum(1 for v in criteria.values() if v)
    else:
        # Fallback: aus overall grob ableiten, wie viele Kriterien erfüllt sind
        # 4 Kriterien -> Wir mappen:
        #  - >= 80 -> 4 erfüllt
        #  - 60–79 -> 3 erfüllt
        #  - 40–59 -> 2 erfüllt
        #  - 20–39 -> 1 erfüllt
        #  - < 20  -> 0 erfüllt
        criteria_total = 4
        if overall >= 80:
            fulfilled_count = 4
        elif overall >= 60:
            fulfilled_count = 3
        elif overall >= 40:
            fulfilled_count = 2
        elif overall >= 20:
            fulfilled_count = 1
        else:
            fulfilled_count = 0

        keys = ["authentisch", "professionell", "positiv_loesungsorientiert", "ganzheitlich_einheitlich"]
        criteria = {}
        for idx, key in enumerate(keys):
            criteria[key] = fulfilled_count > idx  # 1. Kriterium als erstes "true" usw.

    # Verhältnis & fehlende Kriterien
    if criteria_total > 0:
        fulfilled_ratio = fulfilled_count / criteria_total
    else:
        fulfilled_ratio = 0.0

    missing_criteria = [k for k, v in criteria.items() if not v]
    all_criteria_met = criteria_total > 0 and fulfilled_count == criteria_total

    # Wenn overall im JSON fehlt, aus Kriterien ableiten
    if overall == 0.0 and criteria_total > 0:
        overall = fulfilled_ratio * 100.0

    feedback = data.get("feedback") or "Keine detaillierte Rückmeldung verfügbar."

    # Scores für Frontend
    state["politeness_score"] = overall
    state["responsibility_score"] = overall
    state["reaction_score"] = overall

    # Interne Kriterien-Daten
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

    Logik mit 4 Kriterien:
    - Wenn mindestens 80 % der Kriterien erfüllt sind (bei 4 Kriterien: 4/4) ->
      Intensität stark senken (praktisch gewonnen).
    - Wenn 60–79 % erfüllt sind (bei 4 Kriterien: 3/4) ->
      Intensität spürbar SENKEN (gute Reaktion, Shitstorm legt sich).
    - Wenn weniger als 60 % erfüllt sind (0–2/4) ->
      Intensität um 5–20 Punkte erhöhen (maximale Verschlechterung bleibt +20).
    - Eine „gute“ Reaktion (>=60 %) darf niemals dazu führen, dass man durch
      Intensitätserhöhung direkt verliert.
    """
    prev = float(state.get("intensity", 50.0))

    criteria_total = int(state.get("criteria_total") or 0)
    fulfilled = int(state.get("criteria_fulfilled_count") or 0)

    # Fallback: wenn irgendwas mit Kriterien schiefging, nutze reaction_score
    if criteria_total > 0:
        fulfilled_ratio = fulfilled / criteria_total
    else:
        reaction_score = float(state.get("reaction_score", 50.0))
        fulfilled_ratio = max(0.0, min(1.0, reaction_score / 100.0))

    if fulfilled_ratio >= 0.8:
        # Sehr gute Antwort: Shitstorm bricht fast komplett ab
        new_intensity = min(prev, 5.0)
        delta = new_intensity - prev
        solved = True
        good_but_not_solved = False
    elif fulfilled_ratio >= 0.6:
        # Gute Antwort: Shitstorm legt sich deutlich, aber noch nicht komplett gelöst
        delta = -10.0
        new_intensity = max(0.0, prev + delta)
        solved = False
        good_but_not_solved = True
    else:
        # Antwort ist nicht gut genug -> Shitstorm verschärft sich um 5–20 Punkte
        missing_ratio = 1.0 - fulfilled_ratio
        # Skala: 5 .. 20 (max +20 Verschlechterung)
        delta = 5.0 + missing_ratio * 15.0
        new_intensity = max(0.0, min(100.0, prev + delta))
        solved = False
        good_but_not_solved = False

    state["intensity"] = new_intensity
    # NEU: Delta für die nächste Community-Runde speichern
    state["last_intensity_delta"] = float(delta)

    # Status-Logik: bei „guten“ Reaktionen (>=60 %) niemals direkt verlieren
    if new_intensity < 10.0:
        state["status"] = "user_won"
    elif new_intensity > 90.0:
        if fulfilled_ratio >= 0.6:
            # Sicherheitsnetz: gute Reaktion darf nicht zum sofortigen „user_lost“ führen
            state["status"] = "running"
        else:
            state["status"] = "user_lost"
    else:
        state["status"] = "running"

    state["history"].append(
        {
            "actor": "system",
            "round": state["round"],
            "content": (
                f"Intensität von {prev:.1f} auf {new_intensity:.1f} geändert "
                f"(Δ={delta:+.1f}, Verhältnis erfüllter Kriterien={fulfilled_ratio:.2f})."
            ),
        }
    )

    return state




def route_after_update(state: ShitstormState) -> str:
    """Bestimmt, ob weiter simuliert oder beendet wird."""
    if state["status"] == "running":
        return "continue"
    return "end"


# --------------------------------------------------------------------------- #
# Zusammenfassung
# --------------------------------------------------------------------------- #

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
            "- authentisch\n"
            "- professionell\n"
            "- positiv & lösungsorientiert\n"
            "- ganzheitlich & einheitlich\n\n"
            "- zeitnah\n"
            "Fasse den Verlauf des Shitstorms und das Verhalten des Users zusammen.\n"
            "Gib konkrete Lernpunkte und Verbesserungsvorschläge, strukturiert an diesen Kriterien.\n"
            "Antwort auf Deutsch, in 2–4 kurzen Absätzen."

            "Gib an welche Kriterien wann erfüllt wurden\n"
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


# --------------------------------------------------------------------------- #
# Graph-Bau
# --------------------------------------------------------------------------- #

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


# Diese Variable wird von LangGraph Server / LangGraph Cloud geladen
graph = build_graph()

__all__ = ["graph", "ShitstormState"]
