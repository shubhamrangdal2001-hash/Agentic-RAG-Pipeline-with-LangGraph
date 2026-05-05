# 🤖 Agentic RAG Pipeline with LangGraph

A self-improving multi-agent research pipeline built with **LangGraph**, **LangChain**, and **Groq**. The system plans, executes, and verifies research tasks in a feedback loop — automatically retrying if quality doesn't meet the bar.

---

## 🧠 How It Works

The pipeline runs three agents in a graph:

```
START → Planner → Executor → Verifier → END
```

| Agent | Role |
|-------|------|
| **Planner** | Breaks the user's goal into up to 5 concrete tasks |
| **Executor** | Completes each task, augmented with live DuckDuckGo web search |
| **Verifier** | Scores results on completeness, accuracy, and clarity (0.0–1.0); critiques if rejected |

If the verifier rejects the results, the critique is fed back into the executor for improvement. The loop runs for a **maximum of 3 iterations** before force-approving.

---

## 📦 Installation

```bash
pip install langgraph langchain langchain-groq langchain-community duckduckgo-search python-dotenv
```

---

## ⚙️ Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key at [console.groq.com](https://console.groq.com).

---

## 🚀 Usage

Set your goal in `Rag.py`:

```python
initial_state: AgentState = {
    "goal": "Research and summarise the top 3 trends in generative AI for 2025",
    ...
}
```

Then run:

```bash
python Rag.py
```

---

## 📊 Output

The script prints step-by-step progress and a final summary:

```
[Planner] Generated 5 tasks:
  1. ...

[Execution] Task: ...
  Result: ...

[Verifier] Score: 0.87 | Approved: True

======== FINAL OUTPUT =========
[Task 1] ...
 ...

Completed in 2 iteration(s).
Approved: True
Score: 0.87
```

---

## 🗂️ Project Structure

```
.
├── Rag.py          # Main pipeline script
├── .env            # API keys (not committed)
└── README.md
```

---

## 🔧 Customization

- **Change the LLM model** — swap `llama-3.1-8b-instant` for any Groq-supported model (e.g. `mixtral-8x7b-32768`)
- **Change the goal** — update `initial_state["goal"]` to any research objective
- **Adjust quality threshold** — modify the verifier's rubric in its system prompt
- **Increase max iterations** — change the `>= 3` guard in `verifier()`

---

## 📋 Agent State Schema

```python
class AgentState(TypedDict):
    goal: str          # The research objective
    tasks: List[str]   # Planned sub-tasks
    results: List[str] # Execution outputs
    critique: str      # Verifier feedback for retry
    approved: bool     # Whether results passed verification
    score: float       # Quality score (0.0 – 1.0)
    iterations: int    # Loop counter
```

---

## 🛠️ Tech Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — agent orchestration graph
- [LangChain](https://langchain.com) — LLM tooling & message types
- [Groq](https://groq.com) — fast LLM inference (LLaMA 3.1)
- [DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/) — real-time web search

---

## 📄 License

MIT
