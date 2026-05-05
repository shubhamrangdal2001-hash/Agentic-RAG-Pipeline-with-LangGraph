import os , json
from dotenv import load_dotenv
from typing import TypedDict, List
from langgraph.graph import StateGraph, START ,  END
# from langchain_openai import ChatOPenAI
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun 
from langchain_groq import ChatGroq 

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant" , api_key=os.getenv("GROQ_API_KEY"))

class AgentState(TypedDict):
    goal: str
    tasks: List[str]
    results: List[str]
    critique: str 
    approved: bool 
    score: float
    iterations: int

search = DuckDuckGoSearchRun()

def planner(state: AgentState) -> AgentState:
    system = """You are a planning agent. Break the user's goal into at most 5 concrete, actionable tasks. Respond only with a valid JSON array of strings. No preamble, no markdown"""
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"Goal: {state['goal']}")
    ]
    response = llm.invoke(messages).content.strip()
    try:
        clean = response.replace("'''json","").replace("'''","").strip()
        tasks = json.loads(clean)
    except json.JSONDecodeError:
        tasks = [response] 
    print(f"\n [Planner] Generated {len(tasks)} tasks:")
    for i , t in enumerate(tasks): print(f" {i+1}. {t}")
    return {**state, "tasks":tasks}

def executor(state: AgentState) -> AgentState:
    results = []
    critique_ctx = ""
    if state["critique"]:
        critique_ctx = f"\n\nYour previous attempt was rejected. Critique: {state['critique']}\nImprove your output accordingly"
    for task in state["tasks"]:
        system = f"""You are an execution agent. Complete the task below thoroughly. Use web search if you need current information. {critique_ctx}"""
        search_ctx = ""
        try: 
            search_result = search.run(task[:100])
            search_ctx = f"\n\nWeb search result for content:\n{search_result[:800]}"
        except:
            pass
        message = [
            SystemMessage(content=system),
            HumanMessage(content=f"Task: {task} {search_ctx}")
            ]

        result = llm.invoke(message).content
        results.append(result)
        print(f"\n[Execution] Task: {task[:60]} \nResult: {result[:120]}")
    return {**state, "results": results , "iterations": state["iterations"] + 1}

initial_state: AgentState = {
    "goal":  "Research and summarise the top 3 trends in generative AI for 2025",
    "tasks": [],
    "results": [],
    "critique":"",
    "approved": False ,
    "iterations": 0
}


def verifier(state: AgentState) -> AgentState:
    # Safety net: approve after 3 iterations regardless
    if state["iterations"] >= 3:
        print("[Verifier] Max iterations reached — force approving.")
        return {**state, "approved": True}

    # Combine task results
    combined_results = "\n\n".join(
        f"Task {i+1}: {t}\nResult: {r}"
        for i, (t, r) in enumerate(zip(state["tasks"], state["results"]))
    )

    # System prompt
    system = """You are a quality verifier. Evaluate the results against the original goal using this rubric:

Completeness: Does it fully address the goal? (0-0.4)
Accuracy: Is the information correct and specific? (0-0.3)
Clarity: Is it well-structured and clear? (0-0.3)

Sum the scores for a total between 0.0 and 1.0.

Respond ONLY as JSON:
{"score": 0.85, "approved": true, "critique": "..."}
"""

    messages = [
        SystemMessage(content=system),
        HumanMessage(
            content=f"Original goal: {state['goal']}\n\nResults:\n{combined_results}"
        )
    ]

    try:
        raw = llm.invoke(messages).content.strip()

        # Clean possible markdown formatting
        clean = raw.replace("```json", "").replace("```", "").strip()

        verdict = json.loads(clean)

        approved = verdict.get("approved", False)
        critique = verdict.get("critique", "")
        score = verdict.get("score", 0.0)

    except Exception as e:
        approved = False
        critique = f"Parsing failed: {str(e)}"
        score = 0.0

    print(f"\n[Verifier] Score: {score:.2f} | Approved: {approved}")

    return {
        **state,
        "approved": approved,
        "score": score,
        "critique": critique,
        "iterations": state["iterations"] + 1
    }

graph = StateGraph(AgentState)
graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_node("verifier", verifier)
graph.add_edge(START, "planner")
graph.add_edge("planner", "executor")
graph.add_edge("executor", "verifier")
graph.add_edge("verifier", END)
app = graph.compile()

final_state = app.invoke(initial_state)
print("\n ======== FINAL OUTPUT =========")
for i , (task , result) in enumerate(zip(final_state["tasks"], final_state["results"])):
    print(f"\n [Task {i+1}] {task}\n {result}")
print(f"\n Completed in {final_state['iterations']} iteration(s).")
print(f"Approved: {final_state.get('approved', False)}")
print(f"Score: {final_state.get('score', 0.0):.2f}")
print(f"Critique: {final_state.get('critique', '')}")