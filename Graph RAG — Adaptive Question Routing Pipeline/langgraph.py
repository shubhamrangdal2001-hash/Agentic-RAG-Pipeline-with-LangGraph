import os, json, time
from dotenv import load_dotenv
from typing import TypedDict, Optional
from dataclasses import dataclass, field
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))

# ─────────────────────────────────────────────
# Metrics tracker (per-run accumulator)
# ─────────────────────────────────────────────
@dataclass
class Metrics:
    question:         str   = ""
    route_taken:      str   = ""
    week_retrieved:   Optional[str] = None
    router_reason:    str   = ""

    # Latency (seconds per node)
    latency_router:   float = 0.0
    latency_retrieve: float = 0.0
    latency_generate: float = 0.0

    # Token counts (prompt + completion per node)
    tokens_router_prompt:     int = 0
    tokens_router_completion: int = 0
    tokens_gen_prompt:        int = 0
    tokens_gen_completion:    int = 0

    # Accuracy score (0.0 – 1.0) from self-eval
    accuracy_score:   float = 0.0
    accuracy_reason:  str   = ""

    @property
    def total_latency(self) -> float:
        return self.latency_router + self.latency_retrieve + self.latency_generate

    @property
    def total_tokens(self) -> int:
        return (self.tokens_router_prompt + self.tokens_router_completion +
                self.tokens_gen_prompt   + self.tokens_gen_completion)


# Global list — one entry per question
run_metrics: list[Metrics] = []
_current: Metrics = Metrics()   # live reference during a run


# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────
class GraphState(TypedDict):
    question: str
    route:    str
    week:     Optional[str]
    reason:   str
    context:  str
    answer:   str


# ─────────────────────────────────────────────
# Knowledge base
# ─────────────────────────────────────────────
COURSE_NOTES = {
    "Week 10": "Week 10: Intro to LangGraph — nodes, edges, StateGraph, START/END and Agents.",
    "Week 9":  "Week 9: RAG — embeddings, vector stores, retrieval chains.",
    "Week 8":  "Week 8: RNN, LSTM and Transformers — sequence modeling, attention, "
               "self-attention and attention mechanism in Transformers.",
}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _token_counts(response_obj) -> tuple[int, int]:
    """Extract prompt + completion tokens from a Groq LangChain response."""
    usage = getattr(response_obj, "usage_metadata", None) or {}
    if hasattr(usage, "get"):
        prompt     = usage.get("input_tokens",  0) or usage.get("prompt_tokens",     0)
        completion = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
    else:
        prompt = completion = 0
    return int(prompt), int(completion)


# ─────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────
def router(state: GraphState) -> GraphState:
    global _current
    system = """You are a routing agent. Given a student's question, decide:
- If it is about course content (RNN, LSTM, Transformers, RAG, embeddings, LangGraph, etc.) → route = "rag", provide the week (Week 8 / Week 9 / Week 10) or null
- If it can be answered directly (math, general knowledge) → route = "direct"
Respond ONLY as JSON: {"route": "rag"|"direct", "week": "Week 8"|"Week 9"|"Week 10"|null, "reason": "..."}

Course topics:
- Week 8: RNN, LSTM, Transformers, sequence modeling, attention, self-attention
- Week 9: RAG, embeddings, vector stores, retrieval chains
- Week 10: LangGraph, nodes, edges, StateGraph, START/END, Agents"""

    t0 = time.perf_counter()
    resp = llm.invoke([SystemMessage(content=system),
                       HumanMessage(content=state["question"])])
    _current.latency_router = time.perf_counter() - t0

    p, c = _token_counts(resp)
    _current.tokens_router_prompt     = p
    _current.tokens_router_completion = c

    raw = resp.content.strip()
    try:
        verdict = json.loads(raw.replace("```json", "").replace("```", "").strip())
    except json.JSONDecodeError:
        verdict = {"route": "direct", "week": None, "reason": "parse error"}

    _current.route_taken    = verdict["route"]
    _current.week_retrieved = verdict.get("week")
    _current.router_reason  = verdict.get("reason", "")

    print(f"  [Router] route={verdict['route']} | week={verdict.get('week')} | "
          f"latency={_current.latency_router:.2f}s | tokens={p+c}")
    return {**state,
            "route":  verdict["route"],
            "week":   verdict.get("week"),
            "reason": verdict.get("reason", "")}


def retrieve(state: GraphState) -> GraphState:
    global _current
    t0   = time.perf_counter()
    week = state.get("week")
    context = COURSE_NOTES.get(week, "\n".join(COURSE_NOTES.values()))
    _current.latency_retrieve = time.perf_counter() - t0

    print(f"  [Retrieve] week={week} | latency={_current.latency_retrieve*1000:.1f}ms")
    return {**state, "context": context}


def direct_answer(state: GraphState) -> GraphState:
    global _current
    _current.latency_retrieve = 0.0
    print("  [Direct] Skipping retrieval.")
    return {**state, "context": ""}


def generate(state: GraphState) -> GraphState:
    global _current
    if state["context"]:
        system = (f"You are a helpful course assistant. Use the following notes to answer.\n\n"
                  f"Course notes:\n{state['context']}")
    else:
        system = "You are a helpful assistant. Answer the question directly and concisely."

    t0   = time.perf_counter()
    resp = llm.invoke([SystemMessage(content=system),
                       HumanMessage(content=state["question"])])
    _current.latency_generate = time.perf_counter() - t0

    p, c = _token_counts(resp)
    _current.tokens_gen_prompt     = p
    _current.tokens_gen_completion = c

    answer = resp.content
    print(f"  [Generate] latency={_current.latency_generate:.2f}s | tokens={p+c}")
    return {**state, "answer": answer}


# ─────────────────────────────────────────────
# Accuracy evaluator  (LLM-as-judge, 0.0–1.0)
# ─────────────────────────────────────────────
def evaluate_accuracy(question: str, answer: str, context: str) -> tuple[float, str]:
    system = """You are a strict answer-quality evaluator.
Score the answer on a scale from 0.0 to 1.0 using:
  Relevance   0–0.4  Does it directly address the question?
  Correctness 0–0.3  Is the information accurate?
  Completeness 0–0.3 Is the answer sufficiently detailed?

If course-notes context is provided, the answer must draw from it.
Respond ONLY as JSON: {"score": 0.85, "reason": "..."}"""

    content = f"Question: {question}\n\nAnswer: {answer}"
    if context:
        content += f"\n\nCourse notes used:\n{context}"

    try:
        raw = llm.invoke([SystemMessage(content=system),
                          HumanMessage(content=content)]).content.strip()
        parsed = json.loads(raw.replace("```json","").replace("```","").strip())
        return float(parsed.get("score", 0.0)), parsed.get("reason", "")
    except Exception as e:
        return 0.0, f"eval error: {e}"


# ─────────────────────────────────────────────
# Routing condition
# ─────────────────────────────────────────────
def route_question(state: GraphState) -> str:
    return "retrieve" if state["route"] == "rag" else "direct_answer"


# ─────────────────────────────────────────────
# Build graph
# ─────────────────────────────────────────────
graph = StateGraph(GraphState)
graph.add_node("router",        router)
graph.add_node("retrieve",      retrieve)
graph.add_node("direct_answer", direct_answer)
graph.add_node("generate",      generate)

graph.add_edge(START, "router")
graph.add_conditional_edges("router", route_question,
                             {"retrieve": "retrieve", "direct_answer": "direct_answer"})
graph.add_edge("retrieve",      "generate")
graph.add_edge("direct_answer", "generate")
graph.add_edge("generate",       END)

app = graph.compile()


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
questions = [
    "What did we cover in week 9?",
    "What is 2 + 2?",
    "Explain embeddings from the course notes.",
    "Explain the attention mechanism in transformers.",
]

for q in questions:
    print(f"\n{'='*60}")
    print(f"Question: {q}")

    _current          = Metrics()
    _current.question = q

    result = app.invoke({
        "question": q,
        "route": "", "week": None, "reason": "",
        "context": "", "answer": ""
    })

    # Accuracy eval
    score, reason = evaluate_accuracy(q, result["answer"], result["context"])
    _current.accuracy_score  = score
    _current.accuracy_reason = reason

    run_metrics.append(_current)
    print(f"  [Accuracy] score={score:.2f} | {reason[:80]}")
    print(f"Answer: {result['answer'][:200]}")


# ─────────────────────────────────────────────
# Summary report
# ─────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"

def score_color(s: float) -> str:
    if s >= 0.8: return GREEN
    if s >= 0.5: return YELLOW
    return RED

print(f"\n\n{'='*60}")
print(f"{BOLD}{'='*60}")
print(f"  PIPELINE METRICS REPORT")
print(f"{'='*60}{RESET}")

for i, m in enumerate(run_metrics, 1):
    route_label = (f"RAG → {m.week_retrieved}" if m.route_taken == "rag"
                   else "DIRECT")
    sc = score_color(m.accuracy_score)

    print(f"\n{BOLD}{CYAN}[Q{i}] {m.question}{RESET}")
    print(f"  {'Route taken':<22} {BOLD}{route_label}{RESET}")
    print(f"  {'Router reason':<22} {DIM}{m.router_reason[:70]}{RESET}")
    print(f"  {'Latency (router)':<22} {m.latency_router:.3f}s")
    print(f"  {'Latency (retrieve)':<22} {m.latency_retrieve*1000:.1f}ms")
    print(f"  {'Latency (generate)':<22} {m.latency_generate:.3f}s")
    print(f"  {'Total latency':<22} {BOLD}{m.total_latency:.3f}s{RESET}")
    print(f"  {'Tokens (router)':<22} {m.tokens_router_prompt} prompt + "
          f"{m.tokens_router_completion} completion")
    print(f"  {'Tokens (generate)':<22} {m.tokens_gen_prompt} prompt + "
          f"{m.tokens_gen_completion} completion")
    print(f"  {'Total tokens':<22} {BOLD}{m.total_tokens}{RESET}")
    print(f"  {'Accuracy score':<22} {sc}{BOLD}{m.accuracy_score:.2f}{RESET}  "
          f"{DIM}{m.accuracy_reason[:70]}{RESET}")

# Aggregate
total_q   = len(run_metrics)
avg_lat   = sum(m.total_latency   for m in run_metrics) / total_q
avg_tok   = sum(m.total_tokens    for m in run_metrics) / total_q
avg_acc   = sum(m.accuracy_score  for m in run_metrics) / total_q
rag_count = sum(1 for m in run_metrics if m.route_taken == "rag")

print(f"\n{BOLD}{'─'*60}")
print(f"  AGGREGATE  ({total_q} questions)")
print(f"{'─'*60}{RESET}")
print(f"  {'Avg total latency':<25} {avg_lat:.3f}s")
print(f"  {'Avg total tokens':<25} {avg_tok:.0f}")
print(f"  {'Avg accuracy score':<25} {score_color(avg_acc)}{avg_acc:.2f}{RESET}")
print(f"  {'RAG routes':<25} {rag_count}/{total_q}")
print(f"  {'Direct routes':<25} {total_q - rag_count}/{total_q}")
print(f"{BOLD}{'='*60}{RESET}\n")