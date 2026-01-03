"""LangGraph graph definition for the Shitstorm-Simulation agent (Multi-Bot mit Epilog).

Dieses File wird von LangGraph Server / LangGraph Cloud geladen.
Die exportierte Variable `graph` ist der ausführbare Graph.
"""

from __future__ import annotations

import os
import re
import json
from typing import Any, Dict, List, Literal, TypedDict

from langgraph.graph import StateGraph
from langgraph.types import interrupt
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# --------------------------------------------------------------------------- #
# Globale Settings
# --------------------------------------------------------------------------- #

# Optional: LangSmith / LangChain Tracing deaktivieren
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
    status: Literal[
        "running",
        "user_won",
        "user_lost",
        "user_won_pending_epilogue",
    ]
    summary: str
    # Dynamisch genutzte Felder:
    # last_intensity_delta: float
    # last_comment_tone: str
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

    # Objekt extrahieren
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        cand = text[start:end + 1]
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass

    # Liste extrahieren
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        cand = text[start:end + 1]
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass

    return None


def _normalize_comment(text: str) -> str:
    """Aggressive Normalisierung für Deduplikation.

    - Kleinbuchstaben
    - Mehrere Whitespaces -> 1 Space
    - Satzzeichen am Anfang/Ende weg
    """
    t = text.strip().lower()
    # Satzzeichen am Anfang/Ende entfernen
    t = re.sub(r"^[\s\.\,\!\?\-\_#\"']+|[\s\.\,\!\?\-\_#\"']+$", "", t)
    # Mehrfach-Spaces zusammenziehen
    t = re.sub(r"\s+", " ", t)
    return t


def _make_llm(env_suffix: str, default_temp: float) -> ChatOpenAI:
    """Erzeuge ein LLM mit optionalem Suffix (NEGATIVE/NEUTRAL/EVAL).

    Die max_tokens werden bewusst hoch gesetzt, damit genug „Luft“ für komplexere
    Antworten vorhanden ist.
    """
    model_name = os.getenv(f"OPENAI_MODEL_{env_suffix}") or os.getenv(
        "OPENAI_MODEL", "gpt-5.1"
    )
    return ChatOpenAI(
        model=model_name,
        temperature=default_temp,
        max_tokens=1024,
    )


def _compute_severity(round_num: int, last_delta: float) -> str:
    """Grobe Klassifizierung, wie stark sich die Lage geändert hat."""
    if round_num == 1:
        return "initial"
    if last_delta <= -15:
        return "strong_decrease"
    if last_delta <= -5:
        return "mild_decrease"
    if last_delta < 5:
        return "neutral"
    if last_delta < 15:
        return "mild_increase"
    return "strong_increase"


def _build_situation_description(round_num: int, last_delta: float) -> str:
    """Sprachliche Beschreibung der Lage für das LLM."""
    if round_num == 1:
        return (
            "Dies sind die ersten Reaktionen der Community auf den Auslöser. "
            "Es gibt noch keine offizielle Antwort des Unternehmens."
        )
    if last_delta <= -15:
        return (
            "Die letzte Antwort des Unternehmens hat den Shitstorm deutlich beruhigt. "
            "Viele nehmen wahr, dass konkrete Verantwortung übernommen wurde und sich wirklich etwas bewegt."
        )
    if last_delta <= -5:
        return (
            "Die letzte Antwort des Unternehmens hat die Lage spürbar entspannt. "
            "Die Community erkennt Fortschritte und ernsthafte Bemühungen an."
        )
    if last_delta < 5:
        return (
            "Die letzte Antwort des Unternehmens hat kaum etwas an der Situation verändert. "
            "Teile der Community sind unzufrieden, andere abwartend."
        )
    if last_delta < 15:
        return (
            "Die letzte Antwort des Unternehmens hat den Shitstorm eher verschärft. "
            "Viele sind noch unzufriedener und sehen zu wenig echte Veränderung."
        )
    return (
        "Die letzte Antwort des Unternehmens hat die Lage stark eskalieren lassen. "
        "Die Community reagiert sehr wütend und extrem misstrauisch."
    )


def _tone_config_for_severity(severity: str, company_name: str):
    """Baseline-Tonkonfiguration je nach Schweregrad."""
    if severity == "initial":
        return dict(
            positive_mode=False,
            tone_instruction=(
                "Die Community reagiert zum ersten Mal auf den Auslöser. "
                "Die Kommentare sind deutlich kritisch, wütend und fordernd."
            ),
            comment_mix_hint=(
                "Erzeuge überwiegend harte, kritische Kommentare, die deutlich machen, "
                f"dass {company_name} mit der Kampagne oder dem Vorfall eine Grenze überschritten hat."
            ),
        )
    if severity == "strong_decrease":
        return dict(
            positive_mode=True,
            tone_instruction=(
                "Die letzte Antwort und die Maßnahmen des Unternehmens haben den Shitstorm deutlich beruhigt. "
                "Viele empfinden das Statement als klare, längst fällige Klarstellung und sehen echte Veränderungsbereitschaft."
            ),
            comment_mix_hint=(
                "Erzeuge überwiegend leicht positive, dankbare und zukunftsorientierte Kommentare. "
                f"Die Leute bedanken sich explizit für das Statement bzw. die Klarstellung von {company_name}, "
                "loben konkrete Schritte und blicken vorsichtig positiv in die Zukunft."
            ),
        )
    if severity == "mild_decrease":
        return dict(
            positive_mode=True,
            tone_instruction=(
                "Die Community erkennt Fortschritte an und nimmt das Statement des Unternehmens überwiegend positiv wahr, "
                "auch wenn noch nicht alles perfekt ist."
            ),
            comment_mix_hint=(
                "Erzeuge überwiegend konstruktive Kommentare, die sich für die Klarstellung und erste Schritte "
                f"von {company_name} bedanken. Einige Kommentare dürfen freundlich darauf hinweisen, "
                "dass bestimmte Punkte weiter beobachtet werden sollten."
            ),
        )
    if severity == "neutral":
        return dict(
            positive_mode=False,
            tone_instruction=(
                "Die letzte Antwort hat kaum etwas verändert. Die Stimmung ist gemischt."
            ),
            comment_mix_hint=(
                "Erzeuge eine Mischung aus nüchtern-kritischen, skeptischen und einigen neutralen Kommentaren."
            ),
        )
    if severity == "mild_increase":
        return dict(
            positive_mode=False,
            tone_instruction=(
                "Die Community ist eher noch kritischer geworden. Die Antwort wirkt vielen zu schwach."
            ),
            comment_mix_hint=(
                "Erzeuge überwiegend harte, kritische Kommentare, die deutlich machen, "
                f"dass {company_name} zu wenig Verantwortung übernimmt oder zu vage bleibt."
            ),
        )
    return dict(
        positive_mode=False,
        tone_instruction=(
            "Die letzte Antwort hat großen Frust ausgelöst. Die Community fühlt sich nicht ernst genommen."
        ),
        comment_mix_hint=(
            "Erzeuge sehr scharfe, deutlich ablehnende Kommentare, die großes Misstrauen "
            f"gegenüber {company_name} ausdrücken. Die Formulierungen dürfen sehr hart und sarkastisch sein, "
            "aber ohne Gewaltaufrufe und ohne diskriminierende oder hasserfüllte Sprache."
        ),
    )


def _timeout_mode(state: ShitstormState) -> Literal["no_response", "after_response", "none"]:
    """Ermittelt, ob ein Timeout als letzte 'Antwort' im State steht."""
    company_events = [h for h in state.get("history", []) if h.get("actor") == "company"]
    if not company_events:
        return "none"

    last_content = str(company_events[-1].get("content", ""))
    last_is_timeout = last_content.startswith(TIMEOUT_MARKER_PREFIX)

    had_real_before = any(
        isinstance(ev.get("content"), str)
        and not str(ev.get("content")).startswith(TIMEOUT_MARKER_PREFIX)
        for ev in company_events[:-1]
    )

    if last_is_timeout and not had_real_before:
        return "no_response"
    if last_is_timeout and had_real_before:
        return "after_response"
    return "none"


def _target_post(state: ShitstormState, round_num: int, timeout_mode: str) -> Dict[str, str]:
    """Bestimmt, auf welchen Post sich die Kommentare beziehen sollen."""
    cause = state["cause"]
    last_answer = state.get("last_company_response", "")

    if timeout_mode == "no_response":
        return dict(
            post_type=(
                "Beschwerde-Post einer Nutzerin / eines Nutzers "
                "(das Unternehmen hat bisher nicht öffentlich reagiert)"
            ),
            post_text=cause,
        )

    if timeout_mode == "after_response":
        company_events = [h for h in state.get("history", []) if h.get("actor") == "company"]
        real_answer = None
        for ev in reversed(company_events):
            txt = str(ev.get("content", ""))
            if not txt.startswith(TIMEOUT_MARKER_PREFIX):
                real_answer = txt
                break
        return dict(
            post_type=(
                "öffentliche Antwort des Unternehmens "
                "(Community wartet auf weitere konkrete Reaktionen)"
            ),
            post_text=real_answer or last_answer or cause,
        )

    if round_num == 1:
        return dict(
            post_type="Beschwerde-Post einer Nutzerin / eines Nutzers",
            post_text=cause,
        )
    return dict(
        post_type="öffentliche Antwort des Unternehmens",
        post_text=last_answer or cause,
    )


def _compute_comment_mix(round_num: int, last_delta: float) -> tuple[int, int]:
    """Berechnet (n_positive, n_negative) abhängig von Δ.

    Regeln:
    - Runde 1: immer 0 positive, 6 negative (klassischer Start-Shitstorm).
    - Sehr starke Verbesserung (perfekte Nachricht, großer Δ nach unten):
        -> 6 positive, 0 negative.
    - Δ ≈ -10: 2 positive, 4 negative.
    - Δ = 0: 1 positive, 5 negative.
    - Δ > 0: 0 positive, 6 negative.
    """
    # Runde 1: Start-Shitstorm
    if round_num == 1:
        return 0, 6

    # Sehr starke Verbesserung (perfekte Nachricht, z.B. -30, -60 usw.)
    if last_delta <= -30:
        return 6, 0

    # Gute Verbesserung: -30 < Δ <= -15 -> 4 positive, 2 negative
    if last_delta <= -15:
        return 4, 2

    # Spürbare, aber nicht perfekte Verbesserung: -15 < Δ < -5 -> 2 positive, 4 negative
    if last_delta < -5:
        return 2, 4

    # Leichte Verbesserung oder gleich geblieben: -5 <= Δ <= 0 -> 1 positive, 5 negative
    if last_delta <= 0:
        return 1, 5

    # Verschlechterung: Δ > 0 -> 0 positive, 6 negative
    return 0, 6


# --------------------------------------------------------------------------- #
# Community-Runde – Multi-Bot-Kommentarerzeugung
# --------------------------------------------------------------------------- #

def community_round(
    state: ShitstormState,
    negative_llm: ChatOpenAI,
    neutral_llm: ChatOpenAI,
) -> ShitstormState:
    """Erzeugt Community-Kommentare für die aktuelle Runde.

    Multi-Bot-Logik:
    - negative_llm: harsche / eskalierende Kommentare
    - neutral_llm: neutrale bis unterstützende Kommentare
    Die genaue Mischung (positiv/negativ) hängt von last_intensity_delta (Δ) ab.
    """
    state["round"] += 1
    round_num = state["round"]

    platform = state["platform"]
    company_name = state["company_name"]

    prev_intensity = float(state.get("intensity", 50.0))
    if round_num == 1:
        state["intensity"] = 50.0
    intensity = float(state.get("intensity", prev_intensity))

    last_delta = float(state.get("last_intensity_delta", 0.0))

    # X-UI?
    is_x = isinstance(platform, str) and (
        platform.lower().startswith("x") or "twitter" in platform.lower()
    )

    severity = _compute_severity(round_num, last_delta)
    situation_desc = _build_situation_description(round_num, last_delta)
    tone_cfg = _tone_config_for_severity(severity, company_name)

    base_positive_mode: bool = tone_cfg["positive_mode"]
    tone_instruction: str = tone_cfg["tone_instruction"]
    comment_mix_hint_base: str = tone_cfg["comment_mix_hint"]

    # Kommentar-Mix anhand von Δ bestimmen
    n_positive, n_negative = _compute_comment_mix(round_num, last_delta)
    n_total = n_positive + n_negative

    # Dominanter Ton (für Kopfzeile): positiv, wenn Mehrheit positiv
    if n_positive > n_negative:
        positive_mode = True
    elif n_negative > n_positive:
        positive_mode = False
    else:
        # Gleichstand -> Basis-Modus aus der Schwere
        positive_mode = base_positive_mode

    # Timeout-Analyse & Ziel-Post
    t_mode = _timeout_mode(state)
    target = _target_post(state, round_num, t_mode)
    target_post_type = target["post_type"]
    target_post_text = target["post_text"]

    # Kurzer Auszug aus der History
    recent_events: List[str] = []
    for h in state.get("history", [])[-6:]:
        actor = h.get("actor", "?")
        content = str(h.get("content", ""))
        if len(content) > 160:
            content = content[:160] + "…"
        recent_events.append(f"- {actor}: {content}")
    recent_text = "\n".join(recent_events) if recent_events else "noch keine relevanten Einträge"

    # Bisherige Community-Kommentare (für Anti-Duplikation)
    previous_replies = [
        str(h.get("content", "")).strip()
        for h in state.get("history", [])
        if h.get("actor") == "community"
    ]
    previous_replies_text = (
        "\n".join(f"- {c}" for c in previous_replies[-25:])
        if previous_replies
        else "Keine bisherigen Community-Kommentare."
    )

    # System-Prompt: negativer vs. unterstützender Kommentar-Bot
    if positive_mode:
        system_tone_header = (
            "Du simulierst eine Kommentarspalte in einem Social-Media-Shitstorm.\n"
            "Schreibe auf Deutsch.\n"
            "Die letzte Antwort des Unternehmens hat bereits spürbar zur Deeskalation beigetragen.\n"
            "Die Kommentare enthalten je nach Vorgabe einige dankbare, unterstützende Stimmen, "
            "aber ggf. auch noch vereinzelt kritische oder skeptische Aussagen.\n"
        )
    else:
        system_tone_header = (
            "Du simulierst eine Kommentarspalte in einem Social-Media-Shitstorm.\n"
            "Schreibe auf Deutsch.\n"
            "Der Ton ist mehrheitlich hart, wütend, frustriert und spöttisch. "
            "Du darfst harte Kritik, Sarkasmus, zynische Kommentare und deutliche Beleidigungen "
            "gegen das Unternehmen und seine Entscheidungen verwenden.\n"
            "WICHTIG: Vermeide Gewaltaufrufe und verwende KEINE rassistische, sexistische oder sonst "
            "diskriminierende Sprache gegen Gruppen.\n"
        )

    # Konkrete Mix-Beschreibung ins Prompt übernehmen
    if n_positive == 6 and n_negative == 0:
        comment_mix_hint = (
            "Erzeuge ausschließlich unterstützende, dankbare Kommentare. "
            f"Die Community ist spürbar erleichtert und lobt {company_name} für das Statement."
        )
    else:
        comment_mix_hint = (
            comment_mix_hint_base
            + "\n"
            f"Erzeuge insgesamt genau {n_total} Kommentare, davon {n_negative} klar kritische/negative "
            f"und {n_positive} eher unterstützende/positive Kommentare. "
            "Die Gesamtstimmung entspricht diesem Verhältnis."
        )

    system_content = (
        system_tone_header
        + f"{tone_instruction}\n"
        "Du schreibst NUR Community-Kommentare, NIEMALS die Antwort des Unternehmens.\n"
        "Jeder Kommentar ist eine einzelne, eigenständige Antwort (kein Dialog, keine langen Threads).\n"
        "Alle Kommentare beziehen sich klar auf den EINEN Post direkt darüber (Inhalt, Ton, Lücken).\n"
        "Du darfst typische X-Stilmittel nutzen: Emojis, Großbuchstaben für Betonung, "
        "Auslassungspunkte, sehr direkte Formulierungen, kurze Hashtags (max. 1–2 pro Kommentar).\n"
        "Mische sehr kurze, knallige Antworten mit etwas längeren, aber bleibe immer unter ca. 200 Zeichen.\n"
        "Antwort NUR als JSON-Liste von Strings, z.B.:\n"
        '  [\"Kommentar 1\", \"Kommentar 2\", \"...\"]\n'
        "Kein zusätzlicher Text, keine Erklärungen, keine JSON-Objekte.\n"
        "WICHTIG:\n"
        "- Erzeuge Kommentare, die NICHT identisch oder fast identisch mit früheren "
        "Community-Kommentaren sind.\n"
        "- Keine Wiederholungen, keine fast gleichen Formulierungen.\n"
        "- Jeder Kommentar muss neu, frisch, eindeutig formuliert und lesbar sein.\n"
    )

    if is_x:
        system_content += (
            "\nSpezifisch für die Plattform X/Twitter:\n"
            "- Du simulierst die „Antworten“-Sektion unter einem Post.\n"
            "- Schreibe kurze, pointierte, sehr direkte Kommentare (max. ca. 200 Zeichen).\n"
            "- Stil je nach Lage: von stark kritisch bis zunehmend konstruktiv, "
            "abhängig von der beschriebenen Intensitätsänderung.\n"
            "- Keine @Handles oder Namen im Kommentartext, die UI zeigt Namen/Handles separat.\n"
            "- Vermeide Hashtag-Spam, maximal 0–2 Hashtags pro Kommentar.\n"
        )

    # Timeout-spezifische Zusatzinstruktionen
    if t_mode == "no_response":
        extra_timeout_instr = (
            "\nZusätzlicher Fokus: Das Unternehmen hat TROTZ Shitstorm noch nicht öffentlich reagiert.\n"
            "- Kritisiere explizit das Schweigen, die fehlende Reaktion und das Ignorieren der Community.\n"
            "- Stelle infrage, wie ernst das Unternehmen die Situation wirklich nimmt.\n"
        )
    elif t_mode == "after_response":
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

    if positive_mode:
        emotional_hint = (
            f"Generiere genau {n_total} kurze Kommentare der Community. "
            f"Davon sollen {n_negative} Kommentare eher kritisch/skeptisch formuliert sein "
            f"und {n_positive} Kommentare deutlich unterstützend und dankbar wirken. "
            f"Die unterstützenden Kommentare bedanken sich ausdrücklich bei {company_name} für das Statement, "
            "die Klarstellung oder die konkreten Schritte und blicken vorsichtig positiv in die Zukunft."
        )
    else:
        emotional_hint = (
            f"Generiere genau {n_total} kurze Kommentare der Community. "
            f"Davon sollen {n_negative} Kommentare klar kritisch bis stark ablehnend "
            f"gegenüber {company_name} sein und {n_positive} Kommentare dürfen vorsichtig anerkennen, "
            "dass einzelne Punkte richtig oder hilfreich sind. "
            "Die Formulierungen können sehr hart und sarkastisch sein, "
            "solange sie keine Gewaltaufrufe oder diskriminierende/hasserfüllte Sprache enthalten."
        )

    human_msg = HumanMessage(
        content=(
            f"Plattform: {platform}\n"
            f"Unternehmen: {company_name}\n"
            f"Ursache des Shitstorms: {state['cause']}\n"
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

    # Multi-Bot-Auswahl: negativer vs. neutral/positiver LLM
    llm_to_use = neutral_llm if positive_mode else negative_llm
    result = llm_to_use.invoke([system_msg, human_msg])
    comments = _safe_load_json(result.content) or []

    if not isinstance(comments, list) or not comments:
        if n_positive >= n_negative:
            comments = [
                f"Ich bin ehrlich erleichtert – {company_name} klingt dieses Mal wirklich konkret.",
                f"Respekt, {company_name}. So ein klares Statement hätte ich nicht erwartet.",
            ]
        else:
            comments = [
                f"Das klingt alles sehr vage – {company_name} geht keinem Punkt wirklich klar nach.",
                f"{company_name} sagt viel, beantwortet aber kaum eine der wichtigsten Fragen.",
            ]

    # --- HARTE DEDUPLIKATION über normalisierte Strings ---------------------
    normalized_existing = {
        _normalize_comment(c) for c in previous_replies if c.strip()
    }
    unique_round: List[str] = []
    seen_round: set[str] = set()

    for raw in comments:
        c = str(raw).strip()
        if not c:
            continue
        norm = _normalize_comment(c)
        if norm in normalized_existing:
            continue  # schon früher in der History
        if norm in seen_round:
            continue  # doppelt in dieser Runde
        seen_round.add(norm)
        unique_round.append(c)

    # Fallback, falls wirklich alles wegdedupliziert wurde
    if not unique_round:
        if n_positive >= n_negative:
            unique_round = [
                f"Nochmals danke an {company_name} – das wirkt auf mich dieses Mal wirklich glaubwürdig.",
                f"Für mich ist das ein wichtiger Schritt, {company_name}. Jetzt bitte konsequent dranbleiben.",
            ]
        else:
            unique_round = [
                f"{company_name} redet immer noch um den heißen Brei herum.",
                f"Für mich wirkt das alles noch nicht glaubwürdig, {company_name}.",
            ]

    # Tonflag für diese Runde (für Frontend)
    if n_positive == n_total:
        comment_tone = "supportive"
    elif n_negative == n_total:
        comment_tone = "critical"
    else:
        comment_tone = "mixed"

    state["last_comment_tone"] = comment_tone

    state["last_community_comments"] = unique_round
    for c in unique_round:
        entry: Dict[str, Any] = {
            "actor": "community",
            "round": round_num,
            "content": c,
            "tone": comment_tone,
        }
        if is_x:
            entry["section"] = "x_replies"
        state["history"].append(entry)

    return state


# --------------------------------------------------------------------------- #
# Epilog-Community-Runde nach Sieg
# --------------------------------------------------------------------------- #

def epilogue_community_round(
    state: ShitstormState,
    neutral_llm: ChatOpenAI,
) -> ShitstormState:
    """Letzte supportive Kommentarrunde nach einer sehr guten Antwort.

    Wird nur aufgerufen, wenn die Intensität bereits stark gesunken ist und der
    Shitstorm im Prinzip gewonnen ist. Erzeugt eine klare, positive
    Community-Reaktion auf die letzte Unternehmensantwort.
    """
    state["round"] += 1
    round_num = state["round"]

    platform = state["platform"]
    company_name = state["company_name"]
    cause = state["cause"]
    intensity = float(state.get("intensity", 0.0))
    last_answer = state.get("last_company_response", "")

    is_x = isinstance(platform, str) and (
        platform.lower().startswith("x") or "twitter" in platform.lower()
    )

    # Kurzer Auszug aus der History
    recent_events: List[str] = []
    for h in state.get("history", [])[-8:]:
        actor = h.get("actor", "?")
        content = str(h.get("content", ""))
        if len(content) > 200:
            content = content[:200] + "…"
        recent_events.append(f"- {actor}: {content}")
    recent_text = "\n".join(recent_events) if recent_events else "noch keine relevanten Einträge"

    # Bisherige Community-Kommentare (für Anti-Duplikation)
    previous_replies = [
        str(h.get("content", "")).strip()
        for h in state.get("history", [])
        if h.get("actor") == "community"
    ]
    previous_replies_text = (
        "\n".join(f"- {c}" for c in previous_replies[-25:])
        if previous_replies
        else "Keine bisherigen Community-Kommentare."
    )

    system_content = (
        "Du simulierst die Kommentarspalte in einem Social-Media-Shitstorm NACH einer sehr guten Antwort des Unternehmens.\n"
        "Schreibe auf Deutsch.\n"
        "Die letzte Antwort und die geplanten Maßnahmen von {company_name} haben den Shitstorm praktisch beendet.\n"
        "Die Kommentare sind überwiegend dankbar, erleichtert und konstruktiv.\n"
        "Es gibt keine neuen Vorwürfe, keine zynischen Kommentare und keine Angriffe mehr.\n"
        "Stattdessen bedanken sich die Leute ausdrücklich für die Klarstellung, Entschuldigung und die konkreten Schritte.\n"
        "Du schreibst NUR Community-Kommentare, NIEMALS die Antwort des Unternehmens.\n"
        "Antwort NUR als JSON-Liste von Strings, z.B. [\"Kommentar 1\", \"Kommentar 2\", \"...\"]\n"
        "Kein zusätzlicher Text, keine Erklärungen, keine JSON-Objekte.\n"
    ).format(company_name=company_name)

    if is_x:
        system_content += (
            "\nSpezifisch für die Plattform X/Twitter:\n"
            "- Du simulierst die Antworten direkt unter dem finalen Statement.\n"
            "- Schreibe kurze, pointierte Kommentare (max. ca. 200 Zeichen).\n"
            "- Kein Hashtag-Spam, maximal 0–2 Hashtags pro Kommentar.\n"
        )

    system_content += (
        "\nZusatzregel:\n"
        f"- Wenn in den Kommentaren über das Unternehmen gesprochen wird, verwende IMMER den exakten Namen „{company_name}“.\n"
        "- Verwende KEINE allgemeinen Pronomen wie „ihr“, „euch“, „denen“, „die Firma“, „dieser Laden“ oder ähnliche Umschreibungen.\n"
        f"- Formuliere Zustimmung, Dank und positive Erwartungen direkt mit dem Namen „{company_name}“.\n"
    )

    system_msg = SystemMessage(content=system_content)

    human_msg = HumanMessage(
        content=(
            f"Plattform: {platform}\n"
            f"Unternehmen: {company_name}\n"
            f"Ursache des ursprünglichen Shitstorms: {cause}\n"
            f"Aktuelle Shitstorm-Intensität (0-100): {intensity}\n"
            f"Runde (Epilog): {round_num}\n\n"
            "Die folgende Antwort des Unternehmens steht direkt über der Kommentarspalte, die du simulierst:\n"
            f"Antwort des Unternehmens:\n\"\"\"{last_answer}\"\"\"\n\n"
            "Relevante Ausschnitte aus dem bisherigen Verlauf:\n"
            f"{recent_text}\n\n"
            "Alle bisherigen Community-Kommentare (nicht wiederholen!):\n"
            f"{previous_replies_text}\n\n"
            "Generiere genau 6 kurze Kommentare der Community, die sich ausdrücklich für das Statement, "
            f"die Klarstellung oder die konkreten Schritte von {company_name} bedanken oder sie als wichtigen "
            "Schritt anerkennen. Die Kommentare blicken vorsichtig positiv in die Zukunft und drücken Erleichterung "
            "oder Hoffnung aus. Formuliere KEINE neuen Vorwürfe und keine zynischen Untertöne.\n"
            "Jeder Kommentar muss sich klar auf das obenstehende Statement beziehen.\n"
            "Gib deine Antwort NUR als JSON-Liste von Strings zurück."
        )
    )

    result = neutral_llm.invoke([system_msg, human_msg])
    comments = _safe_load_json(result.content) or []

    if not isinstance(comments, list) or not comments:
        comments = [
            f"Danke, {company_name} – so eine klare Stellungnahme hätte ich mir von Anfang an gewünscht.",
            f"Ich finde es stark, wie {company_name} hier Verantwortung übernimmt und konkrete Schritte ankündigt.",
        ]

    normalized_existing = {
        _normalize_comment(c) for c in previous_replies if c.strip()
    }
    unique_round: List[str] = []
    seen_round: set[str] = set()

    for raw in comments:
        c = str(raw).strip()
        if not c:
            continue
        norm = _normalize_comment(c)
        if norm in normalized_existing or norm in seen_round:
            continue
        seen_round.add(norm)
        unique_round.append(c)

    if not unique_round:
        unique_round = [
            f"Für mich ist das Thema mit dieser Antwort erledigt – danke, {company_name}.",
            f"Ich hoffe, dass {company_name} diese Linie beibehält. So fühlt sich das ernst gemeint an.",
        ]

    comment_tone = "supportive"
    state["last_comment_tone"] = comment_tone

    state["last_community_comments"] = unique_round
    for c in unique_round:
        entry: Dict[str, Any] = {
            "actor": "community",
            "round": round_num,
            "content": c,
            "tone": comment_tone,
        }
        if is_x:
            entry["section"] = "x_replies"
        state["history"].append(entry)

    # Nach dem Epilog ist der Status endgültig "user_won"
    state["status"] = "user_won"

    return state


# --------------------------------------------------------------------------- #
# Human-in-the-loop Node für Unternehmensantwort
# --------------------------------------------------------------------------- #

def company_response_node(state: ShitstormState) -> ShitstormState:
    """Unternehmensantwort via interrupt (Frontend liefert Text nach)."""
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
# Bewertungs-Bot (LLM-Evaluation)
# --------------------------------------------------------------------------- #

def llm_evaluate(state: ShitstormState, eval_llm: ChatOpenAI) -> ShitstormState:
    """Bewertet die Unternehmensantwort anhand der 4 Kern-Kriterien
    und prüft zusätzlich, ob das Thema in DMs/privat ausgelagert wird.
    """
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
            "ZUSÄTZLICH sollst du prüfen, ob die Antwort nahelegt, dass das eigentliche Problem\n"
            "in Direktnachrichten / privat gelöst werden soll (z.B. Formulierungen wie\n"
            "'schreib uns eine DM', 'wir klären das per Direktnachricht', 'melde dich privat',\n"
            "'wir schreiben dir eine DM', 'wir klären das im Privatchat').\n"
            "- Wenn das Thema überwiegend oder vollständig in DMs verlagert werden soll,\n"
            "  ist das kommunikativ problematisch. In diesem Fall setze ein eigenes Feld\n"
            '  \"dm_privatisierung\" auf true.\n'
            "- Wenn DMs nur zusätzlich zu einer transparenten öffentlichen Antwort erwähnt werden,\n"
            '  kannst du \"dm_privatisierung\" auf false lassen.\n\n'
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
            '  \"dm_privatisierung\": true/false,\n'
            '  \"feedback\": \"Kurzes, konkretes Feedback dazu, welche Kriterien gut erfüllt sind '
            "und welche noch verbessert werden sollten.\"\n"
            "}\n"
        )
    )

    result = eval_llm.invoke([system_msg, human_msg])
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
        criteria = {k: bool(v) for k, v in raw_criteria.items()}
        criteria_total = len(criteria)
        fulfilled_count = sum(1 for v in criteria.values() if v)
    else:
        # Fallback: aus overall grob ableiten
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

        keys = [
            "authentisch",
            "professionell",
            "positiv_loesungsorientiert",
            "ganzheitlich_einheitlich",
        ]
        criteria = {key: fulfilled_count > idx for idx, key in enumerate(keys)}

    if criteria_total > 0:
        fulfilled_ratio = fulfilled_count / criteria_total
    else:
        fulfilled_ratio = 0.0

    missing_criteria = [k for k, v in criteria.items() if not v]
    all_criteria_met = criteria_total > 0 and fulfilled_count == criteria_total

    if overall == 0.0 and criteria_total > 0:
        overall = fulfilled_ratio * 100.0

    # --- DM-Privatisierung prüfen -------------------------------------------
    dm_priv = data.get("dm_privatisierung", None)

    # Fallback, falls das Modell das Feld nicht liefert: einfache Heuristik
    if dm_priv is None:
        answer_lower = answer.lower()
        dm_keywords = [
            " dm ", "per dm", "direct message", "direktnachricht",
            "privatnachricht", "schreib uns privat", "schreibe uns privat",
            "melde dich privat", "im privatchat", "privat klären",
            "schreib uns eine dm", "wir schreiben dir eine dm",
        ]
        dm_priv = any(kw in answer_lower for kw in dm_keywords)

    dm_priv = bool(dm_priv)
    state["dm_privatisierung"] = dm_priv

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
                f"DM-Privatisierung: {dm_priv}. "
                f"Feedback: {feedback}"
            ),
        }
    )

    return state



# --------------------------------------------------------------------------- #
# Intensitäts-Update
# --------------------------------------------------------------------------- #

def update_intensity(state: ShitstormState) -> ShitstormState:
    """Passt die Intensität anhand der Kriterien an und speichert Δ.

    Logik:
    - >=80 % der Kriterien erfüllt -> Intensität stark runter, Sieg mit Epilog
    - 60–79 % -> Intensität deutlich runter, weiter „running“
    - <60 % -> Intensität +5 bis +20
    - Wenn die Antwort das Thema in DMs/privat verlagert (dm_privatisierung=True),
      wird zusätzlich ein Malus von +5 Punkten auf die Intensität aufgeschlagen.
    """
    prev = float(state.get("intensity", 50.0))

    criteria_total = int(state.get("criteria_total") or 0)
    fulfilled = int(state.get("criteria_fulfilled_count") or 0)

    if criteria_total > 0:
        fulfilled_ratio = fulfilled / criteria_total
    else:
        reaction_score = float(state.get("reaction_score", 50.0))
        fulfilled_ratio = max(0.0, min(1.0, reaction_score / 100.0))

    # --- Basis-Delta aus den Kriterien --------------------------------------
    if fulfilled_ratio >= 0.8:
        # Sehr gute Antwort -> Shitstorm bricht fast komplett ab
        base_new_intensity = min(prev, 5.0)
    elif fulfilled_ratio >= 0.6:
        # Gute Antwort -> deutliche Entspannung
        base_new_intensity = max(0.0, prev - 10.0)
    else:
        # Schlechte Antwort -> Verschlechterung um 5–20 Punkte
        missing_ratio = 1.0 - fulfilled_ratio
        worsen = 5.0 + missing_ratio * 15.0
        base_new_intensity = max(0.0, min(100.0, prev + worsen))

    # --- DM-Malus (+5 Punkte) ----------------------------------------------
    dm_priv = bool(state.get("dm_privatisierung"))
    new_intensity = base_new_intensity
    if dm_priv:
        new_intensity = max(0.0, min(100.0, new_intensity + 5.0))

    delta = new_intensity - prev

    state["intensity"] = new_intensity
    state["last_intensity_delta"] = float(delta)

    # Statuslogik (gute Reaktionen führen nie direkt zum „Verlust“)
    if new_intensity < 10.0:
        state["status"] = "user_won_pending_epilogue"
    elif new_intensity > 90.0 and fulfilled_ratio < 0.6:
        state["status"] = "user_lost"
    else:
        state["status"] = "running"

    state["history"].append(
        {
            "actor": "system",
            "round": state["round"],
            "content": (
                f"Intensität von {prev:.1f} auf {new_intensity:.1f} geändert "
                f"(Δ={delta:+.1f}, Verhältnis erfüllter Kriterien={fulfilled_ratio:.2f}, "
                f"DM-Privatisierung={dm_priv})."
            ),
        }
    )

    return state


def route_after_update(state: ShitstormState) -> str:
    """Routing: weiter simulieren, Epilog oder beenden."""
    status = state.get("status")
    if status == "running":
        return "continue"
    if status == "user_won_pending_epilogue":
        return "epilogue"
    return "end"


# --------------------------------------------------------------------------- #
# Zusammenfassung
# --------------------------------------------------------------------------- #

def summarize(state: ShitstormState, llm: ChatOpenAI) -> ShitstormState:
    """Erzeugt eine kurze Zusammenfassung des Verlaufs auf Basis der Kriterien
    und geht ausdrücklich auf DM-Privatisierung ein, falls sie vorkam.
    """
    outcome = {
        "user_won": "Der Shitstorm ist weitgehend abgeklungen.",
        "user_lost": "Der Shitstorm ist außer Kontrolle geraten.",
        "running": "Die Simulation wurde vorzeitig beendet.",
        "user_won_pending_epilogue": "Der Shitstorm ist faktisch gelöst, es folgt eine Abschlussbewertung.",
    }.get(state.get("status", "running"), "Die Simulation wurde vorzeitig beendet.")

    # Verlaufstext bauen
    history_lines: List[str] = []
    for h in state.get("history", []):
        actor = h.get("actor", "?")
        rnd = h.get("round", 0)
        content = str(h.get("content", ""))
        if len(content) > 400:
            content = content[:400] + " [...]"
        history_lines.append(f"Runde {rnd} - {actor}: {content}")

    history_text = "\n".join(history_lines[-60:])

    # DM-Privatisierung aus History ableiten
    dm_rounds: List[int] = []
    for h in state.get("history", []):
        if h.get("actor") == "coach":
            txt = str(h.get("content", ""))
            if "DM-Privatisierung: True" in txt:
                dm_rounds.append(int(h.get("round", 0)))

    dm_used = bool(dm_rounds) or bool(state.get("dm_privatisierung", False))

    if dm_used:
        if dm_rounds:
            rounds_str = ", ".join(str(r) for r in sorted(set(dm_rounds)))
            dm_info = (
                "Es gab Versuche, Teile der Kommunikation in Direktnachrichten/privat zu verlagern "
                f"(insbesondere in Runde(n): {rounds_str})."
            )
        else:
            dm_info = (
                "Es gab Versuche, Teile der Kommunikation in Direktnachrichten/privat zu verlagern."
            )
    else:
        dm_info = (
            "Es gab keine nennenswerten Versuche, das Thema überwiegend in Direktnachrichten/privat zu verschieben."
        )

    system_msg = SystemMessage(
        content=(
            "Du bist ein Coach für Krisenkommunikation und sollst eine Trainingssimulation auswerten.\n"
            "Bewerte insbesondere, wie gut die Antworten des Unternehmens folgende Kriterien erfüllt haben:\n"
            "- authentisch\n"
            "- professionell\n"
            "- positiv & lösungsorientiert\n"
            "- ganzheitlich & einheitlich\n\n"
            "Berücksichtige außerdem explizit den Umgang mit Transparenz und Kanalwahl:\n"
            "- Wurde versucht, die Diskussion oder Lösung in Direktnachrichten / private Kanäle zu verlagern?\n"
            "- Welche Wirkung hat das auf Vertrauen, Glaubwürdigkeit und Deeskalation?\n\n"
            "Fasse den Verlauf des Shitstorms und das Verhalten des Users zusammen.\n"
            "Gib konkrete Lernpunkte und Verbesserungsvorschläge, strukturiert an diesen Kriterien.\n"
            "Antwort auf Deutsch, in 2–4 kurzen Absätzen und erwähne ausdrücklich auch den Umgang mit DMs."
        )
    )

    human_msg = HumanMessage(
        content=(
            f"Ausgangssituation: Shitstorm auf {state['platform']} wegen \"{state['cause']}\".\n"
            f"Unternehmen: {state['company_name']}\n"
            f"Endgültige Shitstorm-Intensität: {state['intensity']:.1f} / 100\n"
            f"Ergebnis: {outcome}\n\n"
            "Ausschnitte aus dem Verlauf:\n"
            f"{history_text}\n\n"
            f"Hinweis zum Umgang mit Direktnachrichten (DMs): {dm_info}\n"
            "Gib bitte eine zusammenhängende Auswertung, in der du auch kurz kommentierst,\n"
            "ob der Einsatz von DMs kommunikativ sinnvoll war oder eher Vertrauen gekostet hat."
        )
    )

    result = llm.invoke([system_msg, human_msg])
    summary = result.content.strip()
    state["summary"] = summary
    return state



# --------------------------------------------------------------------------- #
# Graph-Bau (Multi-Bot mit Epilog)
# --------------------------------------------------------------------------- #

def build_graph():
    """Erzeugt den ausführbaren LangGraph-Workflow für die Simulation."""
    # Drei LLM-Instanzen für die Multi-Bot-Logik:
    negative_llm = _make_llm("NEGATIVE", default_temp=0.4)  # harsche Kommentare
    neutral_llm = _make_llm("NEUTRAL", default_temp=0.4)    # neutrale/positive Kommentare
    eval_llm = _make_llm("EVAL", default_temp=0.0)          # Bewertungs-Bot

    def community_node(state: ShitstormState) -> ShitstormState:
        return community_round(state, negative_llm=negative_llm, neutral_llm=neutral_llm)

    def epilogue_node(state: ShitstormState) -> ShitstormState:
        return epilogue_community_round(state, neutral_llm=neutral_llm)

    def evaluate_node(state: ShitstormState) -> ShitstormState:
        return llm_evaluate(state, eval_llm=eval_llm)

    def summarize_node(state: ShitstormState) -> ShitstormState:
        return summarize(state, eval_llm=eval_llm)

    workflow = StateGraph(ShitstormState)

    workflow.add_node("community_round", community_node)
    workflow.add_node("epilogue_community", epilogue_node)
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
            "epilogue": "epilogue_community",
            "end": "summarize",
        },
    )
    workflow.add_edge("epilogue_community", "summarize")
    workflow.set_finish_point("summarize")

    return workflow.compile(name="Shitstorm-Simulation (Multi-Bot mit Epilog)")


# Diese Variable wird von LangGraph Server / LangGraph Cloud geladen
graph = build_graph()

__all__ = ["graph", "ShitstormState"]
