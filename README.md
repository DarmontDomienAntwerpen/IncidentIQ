# IncidentIQ 🔴

> AI-powered Incident Intelligence — Transform YouTube incident training videos into actionable knowledge.

> Completely adaptable to specific needs and/or industries.

Built by Domien Darmont as the final project for the **Ironhack AI Engineer Bootcamp**.

---

## What it does

Paste a YouTube URL → IncidentIQ loads the transcript, stores it in Pinecone, and lets you:

- **Chat** — Ask any question about the incident video
- **Key Concepts PDF** — Generate a branded cheatsheet with key points and AI recommendations
- **Visual Timeline** — Visualize the incident: Notification → Arrival → Problems → Solutions → End
- **XVR Scenario** — Generate a ready-to-use simulation brief for XVR operators
- **Send** — Email any generated document to your team via Gmail

---

## Stack

| Layer | Technology |
|---|---|
| LLM | GPT-4o-mini |
| Embeddings | text-embedding-3-small |
| Vector DB | Pinecone |
| Agent | LangGraph |
| RAG | LangChain + multi-query retrieval |
| Observability | LangSmith |
| UI | Flask + HTML |
| PDF | ReportLab |

---

## Architecture

```
YouTube URL
    │
    ▼
Pinecone cache check ──► hit: skip YouTube
    │
    ▼ miss
YouTube Transcript API
    │
    ▼
RecursiveCharacterTextSplitter (500 / 50)
    │
    ▼
Pinecone (index: incidentiq, 1536 dims, cosine)
    │
    ▼
LangGraph Agent ◄──► Tools
    │                  ├── search_video_knowledge (multi-query RAG)
    │                  ├── generate_xvr_scenario
    │                  ├── generate_visual_summary
    │                  └── send_gmail_tool
    ▼
Flask/HTML UI
```

---

## Setup

### 1. Clone

```bash
git clone https://github.com/yourname/incidentiq.git
cd incidentiq
```

### 2. Create environment

```bash
python -m venv incidentiq_env
source incidentiq_env/bin/activate
pip install -r requirements.txt
```

### 3. Environment variables

Create a `.env` file:

```env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
LANGSMITH_API_KEY=...
GMAIL_DISTRIBUTION_LIST=email1@domain.com,email2@domain.com
```

### 4. Gmail (optional)

To enable email sending, place your `credentials.json` (Google OAuth) in the project root. On first run, a browser window will open to authenticate.

### 5. Run

```bash
python app.py
```

Open `http://localhost:7860`

---

## Notebooks

| Notebook | Description |
|---|---|
| `01_transcript_pipeline.ipynb` | YouTube fetch, Pinecone cache, chunking |
| `02_rag_chain.ipynb` | Query rewriting, multi-query retrieval |
| `03_agent_tools.ipynb` | All 5 tools with quality tests |
| `04_langgraph_agent.ipynb` | LangGraph agent with memory |
| `05_evaluation.ipynb` | RAGAs, ROUGE, BLEU evaluation |

---

## Evaluation results

| Metric | Score |
|---|---|
| RAGAs Faithfulness | 0.795 |
| RAGAs Answer Relevancy | 0.555 |
| RAGAs Context Recall | 0.750 |
| RAGAs Average | 0.700 |
| ROUGE-1 | 0.382 |
| BLEU | 0.069 |

---

## Domain-agnostic

Built for fire services — adaptable to police, EMS, military or any organization that learns from video. One config change swaps the domain.

---

## Team

Built by Domien Darmont — Ironhack AI Engineer Bootcamp 2025
