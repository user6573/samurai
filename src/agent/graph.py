Hier ist eine **vollständige** `graph.py`, in der:

* dein bisheriger Ablauf (Community → Company → Bewertung → Intensität → ggf. wieder Community) erhalten bleibt
* die **neue Checkliste** als Bewertungsgrundlage genutzt wird
* die **Intensität** stark sinkt, wenn *alle* Kriterien erfüllt sind – und sonst um **10–50 Punkte steigt**
* das **End-Feedback** sich explizit auf diese Kriterien bezieht
* der Graph für den **LangGraph-Server** geeignet ist (mit `interrupt` für deine Website)

> ⚠️ Wichtig:
> In `langgraph.json` sollte dein Graph weiterhin so referenziert werden:
>
> ```json
> {
>   "graphs": {
>     "shitstorm": {
>       "module": "agent.graph",
>       "graph": "graph"
>     }
>   }
> }
> ```

---

```python
import os
import json
from typing import List, Literal, TypedDict, Any, Dict

# LangSmith / LangChain Tracing explizit deaktivieren,
# damit keine nervige 403-Fehlermeldung kommt
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

from langgraph.graph import StateGraph, END
from langgraph.types import interrupt
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


class ShitstormState(TypedDict, total=False):
    # Grunddaten
    platform: str
    cause: str
    company_name: str
    round: int
    history: List[dict]
    last_company_response: str
    last_community_comments: List[str]

    # Scores (für UI / Kompatibilität)
    politeness_score: float
    responsibility_score: float
    reaction_score: float  # Gesamtscore

    # Shitstorm-Intensität & Status
    intensity: float
    status: Literal["running", "user_won", "user_lost"]
    summary: str

    # Neue Felder für Checklisten-Logik
    criteria_fulfilled_count: int
    criteria_total: int
    all_criteria_fulfilled: bool


# Ein LLM für alle Knoten
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=MODEL_NAME, temperature=0.3)


def safe_load_json(text: str):
    """Versuche, robust JSON aus einem LLM-Output zu laden."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start: end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                return None
        return None


# ---------------------------------------------------------------------------
# Community-Runde
# ---------------------------------------------------------------------------

def community_round(state: ShitstormState, llm: ChatOpenAI) -> ShitstormState:
    """Generiert Community-Kommentare für die aktuelle Runde."""
    state["round"] = int(state.get("round", 0)) + 1
    round_num = state["round"]

    platform = state.get("platform", "X/Twitter")
    cause = state.get("cause", "nicht näher beschriebene Ursache")
    company_name = state.get("company_name", "Dein Unternehmen")
    intensity = float(state.get("intensity", 70.0))
    last_answer = state.get("last_company_response", "")
    reaction_score = float(state.get("reaction_score", 0.0))

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
    comments = safe_load_json(result.content) or []

    if not isinstance(comments, list) or not comments:
        # Fallback: Zeilenweise interpretieren
        lines = [ln.strip("- ").strip() for ln in result.content.splitlines() if ln.strip()]
        comments = lines[:5] or [
            "Ich bin echt sauer über diese Situation.",
            "So kann ein Unternehmen nicht mit seinen Kund:innen umgehen.",
        ]

    comments = [str(c) for c in comments]

    state["last_community_comments"] = comments
    history = state.get("history") or []
    for c in comments:
        history.append(
            {
                "actor": "community",
                "round": round_num,
                "content": c,
            }
        )
    state["history"] = history

    print("\n" + "=" * 70)
    print(f"Runde {round_num}")
    print(f"Aktueller Shitstorm-Intensitätswert: {state.get('intensity', 70.0):.1f}/100")
    print("-" * 70)
    print("Die Community kommentiert:")
    for idx, c in enumerate(comments, 1):
        print(f"{idx}. {c}")
    print("-" * 70)

    return state


# ---------------------------------------------------------------------------
# Unternehmens-Antwort (via interrupt – für Web / LangGraph Server)
# ---------------------------------------------------------------------------

def company_response(state: ShitstormState) -> ShitstormState:
    """
    Fragt die Antwort des Unternehmens ab.

    Im LangGraph-Server-Kontext passiert das über `interrupt(...)`.
    Die Website ruft dann /runs/wait mit `command: { "resume": "<Antwort-Text>" }` auf.
    """
    answer = interrupt(
        "company_response",
        description="Antwort des Unternehmens auf die aktuellen Community-Kommentare.",
    )

    # Nach dem Resume-Lauf bekommt `answer` den tatsächlichen String-Wert.
    answer_str = str(answer).strip()
    state["last_company_response"] = answer_str

    history = state.get("history") or []
    history.append(
        {
            "actor": "company",
            "round": state.get("round", 0),
            "content": answer_str,
        }
    )
    state["history"] = history
    return state


# ---------------------------------------------------------------------------
# Bewertung nach Checkliste
# ---------------------------------------------------------------------------

def llm_evaluate(state: ShitstormState, llm: ChatOpenAI) -> ShitstormState:
    """Bewertet die Unternehmensantwort anhand einer festen Checkliste von Kriterien."""
    platform = state.get("platform", "X/Twitter")
    cause = state.get("cause", "nicht näher beschriebene Ursache")
    company_name = state.get("company_name", "Dein Unternehmen")
    answer = state.get("last_company_response", "")

    criteria_keys = [
        "concise_precise",              # Text mit so viel Inhalt wie nötig, kaum Interpretationsspielraum
        "what_happened",                # Was ist passiert?
        "apology_statement",            # Statement bzw. Entschuldigung
        "speaker_identified",           # Wer sagt das?
        "solution_offered",             # Konkrete Lösung / nächster Schritt
        "fast",                         # Schnell
        "authentic",                    # Authentisch
        "professional",                 # Professionell
        "verified_transparent",         # Verifiziert & transparent
        "positive_solution_oriented",   # Positiv & lösungsorientiert
        "holistic_consistent",          # Ganzheitlich & einheitlich
    ]

    system_msg = SystemMessage(
        content=(
            "Du bist ein professioneller Coach für Krisenkommunikation in sozialen Medien.\n"
            "Deine Aufgabe ist es, eine Unternehmensantwort strikt anhand einer Checkliste zu bewerten.\n"
            "Für jedes Kriterium entscheidest du klar TRUE oder FALSE – kein 'teilweise', keine Abstufungen.\n"
            "Antworte AUSSCHLIESSLICH mit einem gültigen JSON-Objekt, ohne erklärenden Text außerhalb von JSON."
        )
    )

    human_msg = HumanMessage(
        content=(
            "Bewerte die folgende Antwort eines Unternehmens in einem Shitstorm.\n\n"
            f"Plattform: {platform}\n"
            f"Unternehmen: {company_name}\n"
            f"Ursache des Shitstorms: {cause}\n\n"
            f"Antwort des Unternehmens:\n\"\"\"{answer}\"\"\"\n\n"
            "Die Antwort soll folgende Kriterien erfüllen (alle möglichst eindeutig, knapp und ohne Interpretationsspielraum):\n"
            "1) concise_precise: Text mit so viel Inhalt wie notwendig, aber nicht mehr. Klar, konkret, kaum Spielraum zur Interpretation.\n"
            "2) what_happened: Es wird verständlich erklärt, was passiert ist.\n"
            "3) apology_statement: Es gibt ein klares Statement und/oder eine Entschuldigung.\n"
            "4) speaker_identified: Es ist nachvollziehbar, wer spricht (z.B. Unternehmen, Rolle oder Person genannt).\n"
            "5) solution_offered: Das Unternehmen bietet eine konkrete Lösung, Wiedergutmachung oder nächste Schritte an.\n"
            "6) fast: Die Antwort vermittelt, dass das Unternehmen schnell reagiert (z.B. signalisiert zügiges Handeln oder zeitnahe Maßnahmen).\n"
            "7) authentic: Wirkt ehrlich, menschlich und authentisch (kein leeres PR-Blabla).\n"
            "8) professional: Professioneller Ton, respektvoll, keine Schuldzuweisungen.\n"
            "9) verified_transparent: Verifiziert & transparent (z.B. nachvollziehbare Fakten, klare Informationen, nichts wird offensichtlich verschleiert).\n"
            "10) positive_solution_oriented: Positiv und lösungsorientiert formuliert.\n"
            "11) holistic_consistent: Ganzheitlich & einheitlich – Botschaft wirkt in sich stimmig und würde auch zum restlichen Auftritt passen.\n\n"
            "Gib NUR folgendes JSON zurück:\n"
            "{\n"
            "  \"criteria\": {\n"
            "    \"concise_precise\": true/false,\n"
            "    \"what_happened\": true/false,\n"
            "    \"apology_statement\": true/false,\n"
            "    \"speaker_identified\": true/false,\n"
            "    \"solution_offered\": true/false,\n"
            "    \"fast\": true/false,\n"
            "    \"authentic\": true/false,\n"
            "    \"professional\": true/false,\n"
            "    \"verified_transparent\": true/false,\n"
            "    \"positive_solution_oriented\": true/false,\n"
            "    \"holistic_consistent\": true/false\n"
            "  },\n"
            "  \"overall\": <Zahl 0-100, je höher desto besser>,\n"
            "  \"feedback\": \"Kurzes, konkretes Feedback für die Nutzerin / den Nutzer – welche Kriterien sind erfüllt, welche fehlen?\"\n"
            "}\n"
        )
    )

    result = llm.invoke([system_msg, human_msg])
    data = safe_load_json(result.content) or {}

    def to_bool(v) -> bool:
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        return s in ("1", "true", "yes", "ja")

    # Kriterien aus JSON extrahieren
    raw_criteria = data.get("criteria") or {}
    criteria_flags: Dict[str, bool] = {
        key: to_bool(raw_criteria.get(key, False)) for key in criteria_keys
    }

    fulfilled_count = sum(1 for v in criteria_flags.values() if v)
    total_criteria = len(criteria_keys)
    all_fulfilled = fulfilled_count == total_criteria and total_criteria > 0

    # Score aus Anteil erfüllter Kriterien ableiten (0–100)
    completeness_score = (fulfilled_count / total_criteria * 100.0) if total_criteria else 0.0

    def clamp_score(v, default):
        try:
            x = float(v)
        except (TypeError, ValueError):
            x = default
        return max(0.0, min(100.0, x))

    overall_from_model = clamp_score(data.get("overall"), completeness_score)
    overall = overall_from_model

    feedback = data.get("feedback") or "Keine detaillierte Rückmeldung verfügbar."

    # Für Abwärtskompatibilität: politeness / responsibility einfach mit dem Gesamtscore mappen
    state["politeness_score"] = overall
    state["responsibility_score"] = overall
    state["reaction_score"] = overall

    # Neue Felder im State für die Intensitätslogik
    state["criteria_fulfilled_count"] = fulfilled_count
    state["criteria_total"] = total_criteria
    state["all_criteria_fulfilled"] = all_fulfilled

    # Menschliches Feedback in der CLI-Ausgabe (Server → Logs)
    print("\nBewertung deiner Antwort (Checkliste):")
    print(f"  Erfüllte Kriterien: {fulfilled_count} / {total_criteria}")
    print(f"  Gesamtscore:        {overall:5.1f} / 100")

    print("\nKriterien:")
    for key in criteria_keys:
        mark = "✅" if criteria_flags[key] else "❌"
        print(f"  {mark} {key}")

    print("\nFeedback des Krisen-Coachs:")
    print(f"  {feedback}")

    # Für den Verlauf im State protokollieren
    history = state.get("history") or []
    history.append(
        {
            "actor": "coach",
            "round": state.get("round", 0),
            "content": (
                f"Kriterien erfüllt: {fulfilled_count}/{total_criteria}. "
                f"Gesamtscore: {overall:.1f}. Feedback: {feedback}"
            ),
        }
    )
    state["history"] = history

    return state


# ---------------------------------------------------------------------------
# Intensität-Update nach neuer Logik
# ---------------------------------------------------------------------------

def update_intensity(state: ShitstormState) -> ShitstormState:
    """Aktualisiert die Shitstorm-Intensität auf Basis der Checkliste.

    - Wenn ALLE Kriterien erfüllt sind, sinkt die Intensität sehr stark (nahezu sicherer Sieg).
    - Wenn NICHT alle Kriterien erfüllt sind, steigt die Intensität um 10–50 Punkte
      (abhängig davon, wie viele Kriterien fehlen).
    """
    prev = float(state.get("intensity", 70.0))

    fulfilled = int(state.get("criteria_fulfilled_count", 0) or 0)
    total = int(state.get("criteria_total", 0) or 0)
    all_full = bool(state.get("all_criteria_fulfilled")) and total > 0

    if all_full:
        # Starker Abfall – so, dass man praktisch sicher gewinnt.
        # Wir senken um mindestens 50 Punkte oder bis nahe 0.
        drop = max(50.0, prev + 5.0)   # +5 als Sicherheitsmarge, damit wir wirklich drunter kommen
        delta = -drop
        new_intensity = max(0.0, prev + delta)
    else:
        # Je mehr Kriterien fehlen, desto stärker steigt die Intensität (10–50 Punkte).
        if total > 0:
            missing = max(0, total - fulfilled)
            fraction = missing / total  # 0..1
        else:
            # Falls aus irgendeinem Grund keine Kriterien im State – maximaler Schaden.
            fraction = 1.0

        delta = 10.0 + 40.0 * fraction  # 10–50
        if delta < 10.0:
            delta = 10.0
        if delta > 50.0:
            delta = 50.0

        new_intensity = max(0.0, min(100.0, prev + delta))

    state["intensity"] = new_intensity

    if new_intensity < 10:
        state["status"] = "user_won"
    elif new_intensity > 90:
        state["status"] = "user_lost"
    else:
        state["status"] = "running"

    print("\nAktualisierte Shitstorm-Intensität:")
    print(f"  Vorher: {prev:5.1f} / 100")
    print(f"  Änderung: {delta:+5.1f}")
    print(f"  Jetzt:  {new_intensity:5.1f} / 100")

    if all_full:
        reason_text = (
            f"Alle Checklisten-Kriterien erfüllt ({fulfilled}/{total}) – starke Deeskalation."
        )
    else:
        reason_text = (
            f"Nicht alle Checklisten-Kriterien erfüllt ({fulfilled}/{total}) – "
            f"Intensität steigt um {delta:+.1f}."
        )

    history = state.get("history") or []
    history.append(
        {
            "actor": "system",
            "round": state.get("round", 0),
            "content": (
                f"Intensität von {prev:.1f} auf {new_intensity:.1f} geändert. {reason_text}"
            ),
        }
    )
    state["history"] = history

    return state


def route_after_update(state: ShitstormState) -> str:
    """Bestimmt, ob weiter simuliert oder beendet wird."""
    status = state.get("status", "running")
    if status == "running":
        return "continue"
    return "end"


# ---------------------------------------------------------------------------
# Zusammenfassung
# ---------------------------------------------------------------------------

def summarize(state: ShitstormState, llm: ChatOpenAI) -> ShitstormState:
    """Erzeugt eine kurze Zusammenfassung des Verlaufs."""
    outcome = {
        "user_won": "Der Shitstorm ist weitgehend abgeklungen.",
        "user_lost": "Der Shitstorm ist außer Kontrolle geraten.",
        "running": "Die Simulation wurde vorzeitig beendet.",
    }.get(state.get("status", "running"), "Die Simulation wurde beendet.")

    # Kurzes, kompaktes Protokoll für das LLM
    history_lines = []
    for h in state.get("history", []):
        actor = h.get("actor", "?")
        rnd = h.get("round", 0)
        content = h.get("content", "")
        if len(content) > 400:
            content = content[:400] + " [...]"
        history_lines.append(f"Runde {rnd} - {actor}: {content}")

    history_text = "\n".join(history_lines[-60:])  # letzte 60 Einträge reichen hier

    system_msg = SystemMessage(
        content=(
            "Du bist ein Coach für Krisenkommunikation und sollst eine Trainingssimulation auswerten.\n"
            "Fasse den Verlauf des Shitstorms und das Verhalten des Users zusammen.\n"
            "Nutze dabei explizit folgende Kriterien als Grundlage der Bewertung:\n"
            "- Text: so viel Inhalt wie nötig, klar und ohne Spielraum zur Interpretation\n"
            "- Was ist passiert?\n"
            "- Statement / Entschuldigung\n"
            "- Wer sagt das?\n"
            "- Welche Lösung bietet das Unternehmen an?\n"
            "- Schnell\n"
            "- Authentisch\n"
            "- Professionell\n"
            "- Verifiziert & transparent\n"
            "- Positiv & lösungsorientiert\n"
            "- Ganzheitlich & einheitlich\n"
            "Gib konkrete Lernpunkte und Verbesserungsvorschläge, orientiert an diesen Kriterien.\n"
            "Antwort auf Deutsch, in 2–4 kurzen Absätzen."
        )
    )

    human_msg = HumanMessage(
        content=(
            f"Ausgangssituation: Shitstorm auf {state.get('platform', 'X/Twitter')} "
            f"wegen \"{state.get('cause', 'nicht näher beschriebene Ursache')}\".\n"
            f"Unternehmen: {state.get('company_name', 'Dein Unternehmen')}\n"
            f"Endgültige Shitstorm-Intensität: {float(state.get('intensity', 0.0)):.1f} / 100\n"
            f"Ergebnis: {outcome}\n\n"
            "Ausschnitte aus dem Verlauf:\n"
            f"{history_text}"
        )
    )

    result = llm.invoke([system_msg, human_msg])
    summary = result.content.strip()
    state["summary"] = summary

    print("\n" + "=" * 70)
    print("SIMULATION BEENDET")
    print("=" * 70)
    print(f"Ergebnis: {outcome}")
    print(f"Endgültige Intensität: {float(state.get('intensity', 0.0)):.1f} / 100\n")
    print("Zusammenfassung & Lernpunkte:")
    print(summary)
    print("=" * 70)

    return state


# ---------------------------------------------------------------------------
# Graph aufbauen und für LangGraph Server bereitstellen
# ---------------------------------------------------------------------------

def build_graph() -> Any:
    """Erzeugt den LangGraph-Workflow für die Simulation."""
    workflow = StateGraph(ShitstormState)

    # Wrapper, damit das Modul-weite LLM verwendet wird
    def community_node(state: ShitstormState) -> ShitstormState:
        return community_round(state, llm)

    def evaluate_node(state: ShitstormState) -> ShitstormState:
        return llm_evaluate(state, llm)

    def summarize_node(state: ShitstormState) -> ShitstormState:
        return summarize(state, llm)

    workflow.add_node("community_round", community_node)
    workflow.add_node("company_response", company_response)
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

    return workflow.compile()


# Dieses Objekt wird vom LangGraph-Server aus langgraph.json geladen
graph = build_graph()
```
