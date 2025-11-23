import os

# LangSmith / LangChain Tracing explizit deaktivieren,
# damit keine nervige 403-Fehlermeldung kommt
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

import sys
import json
from typing import List, Literal, TypedDict

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


try:
    from langgraph.graph import StateGraph, END
except ImportError:
    print("Dieses Skript benötigt das Paket 'langgraph'. Installiere es mit:\n\n    pip install langgraph langchain langchain-openai\n")
    sys.exit(1)

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    print("Dieses Skript benötigt das Paket 'langchain-openai'. Installiere es mit:\n\n    pip install langchain-openai\n")
    sys.exit(1)

import os
import sys
import json
from typing import List, Literal, TypedDict

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage



class ShitstormState(TypedDict):
    platform: str
    cause: str
    company_name: str
    round: int
    history: List[dict]
    last_company_response: str
    last_community_comments: List[str]
    politeness_score: float
    responsibility_score: float
    reaction_score: float
    intensity: float
    status: Literal["running", "user_won", "user_lost"]
    summary: str


def safe_load_json(text: str):
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
        return None


def choose_platform() -> str:
    platforms = ["Facebook", "Instagram", "X/Twitter", "TikTok", "LinkedIn", "YouTube"]
    print("=== Shitstorm-Simulation mit LangGraph ===")
    print("Wähle eine Social-Media-Plattform:")
    for idx, p in enumerate(platforms, 1):
        print(f"  {idx}) {p}")
    while True:
        choice = input("Deine Wahl (Zahl oder Name): ").strip()
        if not choice:
            continue
        if choice.isdigit():
            i = int(choice)
            if 1 <= i <= len(platforms):
                return platforms[i - 1]
        else:
            for p in platforms:
                if choice.lower() in p.lower():
                    return p
        print("Bitte eine gültige Plattform eingeben.")


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
    for c in comments:
        state["history"].append(
            {
                "actor": "community",
                "round": round_num,
                "content": c,
            }
        )

    print("\n" + "=" * 70)
    print(f"Runde {round_num}")
    print(f"Aktueller Shitstorm-Intensitätswert: {state['intensity']:.1f}/100")
    print("-" * 70)
    print("Die Community kommentiert:")
    for idx, c in enumerate(comments, 1):
        print(f"{idx}. {c}")
    print("-" * 70)

    return state


def company_response(state: ShitstormState) -> ShitstormState:
    """Fragt die Antwort des Unternehmens über die CLI ab."""
    print("Formuliere jetzt deine Antwort / deinen Post als Unternehmen.")
    print("Tipp: Sei höflich, übernimm Verantwortung und biete konkrete Schritte an.")
    print("Mehrzeilige Eingabe – leere Zeile beendet die Eingabe.\n")

    lines: List[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":
            if lines:
                break
            else:
                print("Die Antwort darf nicht komplett leer sein. Bitte schreibe zumindest einen Satz.")
                continue
        lines.append(line)

    answer = "\n".join(lines).strip()
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
    """Bewertet die Unternehmensantwort nach Höflichkeit und Verantwortungsübernahme."""
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
    data = safe_load_json(result.content) or {}

    def clamp_score(v, default=50.0):
        try:
            x = float(v)
        except (TypeError, ValueError):
            x = default
        return max(0.0, min(100.0, x))

    politeness = clamp_score(data.get("politeness"))
    responsibility = clamp_score(data.get("responsibility"))
    overall = clamp_score(data.get("overall", (politeness + responsibility) / 2))

    feedback = data.get("feedback") or "Keine detaillierte Rückmeldung verfügbar."

    state["politeness_score"] = politeness
    state["responsibility_score"] = responsibility
    state["reaction_score"] = overall

    print("\nBewertung deiner Antwort:")
    print(f"  Höflichkeit:            {politeness:5.1f} / 100")
    print(f"  Verantwortungsübernahme:{responsibility:5.1f} / 100")
    print(f"  Gesamtscore:            {overall:5.1f} / 100")
    print("\nFeedback des Krisen-Coachs:")
    print(f"  {feedback}")

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

    # Einfache, gut nachvollziehbare Heuristik:
    if score >= 80:
        delta = -20
    elif score >= 65:
        delta = -12
    elif score >= 50:
        delta = -5
    elif score >= 35:
        delta = +5
    elif score >= 20:
        delta = +12
    else:
        delta = +20

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

    # Kurzes, kompaktes Protokoll für das LLM
    history_lines = []
    for h in state["history"]:
        actor = h.get("actor", "?")
        rnd = h.get("round", 0)
        content = h.get("content", "")
        # Etwas kürzen, damit es nicht explodiert
        if len(content) > 400:
            content = content[:400] + " [...]"
        history_lines.append(f"Runde {rnd} - {actor}: {content}")

    history_text = "\n".join(history_lines[-60:])  # letzte 60 Einträge reichen hier

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

    print("\n" + "=" * 70)
    print("SIMULATION BEENDET")
    print("=" * 70)
    print(f"Ergebnis: {outcome}")
    print(f"Endgültige Intensität: {state['intensity']:.1f} / 100\n")
    print("Zusammenfassung & Lernpunkte:")
    print(summary)
    print("=" * 70)

    return state


def build_graph(llm: ChatOpenAI):
    """Erzeugt den LangGraph-Workflow für die Simulation."""
    def community_node(state: ShitstormState) -> ShitstormState:
        return community_round(state, llm)

    def evaluate_node(state: ShitstormState) -> ShitstormState:
        return llm_evaluate(state, llm)

    def summarize_node(state: ShitstormState) -> ShitstormState:
        return summarize(state, llm)

    workflow = StateGraph(ShitstormState)

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


def main():
    # API-Key muss ausschließlich über Umgebungsvariable kommen
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Fehler: Die Umgebungsvariable OPENAI_API_KEY ist nicht gesetzt.\n"
            "Bitte setze sie z.B. mit:\n"
            "  export OPENAI_API_KEY='dein-key-hier'\n"
            "und starte das Skript erneut."
        )
        sys.exit(1)

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0.3)

    platform = choose_platform()
    cause = input("\nWas ist die Ursache des Shitstorms (z.B. Produktfehler, unangebrachte Kampagne, schlechter Kundenservice)?\n> ").strip()
    if not cause:
        cause = "nicht näher beschriebene Ursache"

    company_name = input(
        "\nWie heißt das betroffene Unternehmen? (ENTER für 'Dein Unternehmen')\n> "
    ).strip() or "Dein Unternehmen"

    print("\nAusgangslage:")
    print(f"- Plattform: {platform}")
    print(f"- Unternehmen: {company_name}")
    print(f"- Ursache: {cause}")
    print("- Ausgangs-Intensität des Shitstorms: 70/100")
    print("\nZiel: Bringe die Intensität durch gute Antworten unter 10.")
    print("Achtung: Steigt sie über 90, ist der Shitstorm außer Kontrolle.\n")

    initial_state: ShitstormState = {
        "platform": platform,
        "cause": cause,
        "company_name": company_name,
        "round": 0,
        "history": [],
        "last_company_response": "",
        "last_community_comments": [],
        "politeness_score": 0.0,
        "responsibility_score": 0.0,
        "reaction_score": 0.0,
        "intensity": 70.0,
        "status": "running",
        "summary": "",
    }

    app = build_graph(llm)

    try:
        app.invoke(initial_state)
    except KeyboardInterrupt:
        print("\nSimulation durch Benutzer abgebrochen.")

    print("\nDanke fürs Durchspielen der Shitstorm-Simulation!")
    print("Starte das Skript erneut, um ein neues Szenario zu üben.")


if __name__ == "__main__":
    main()
