# Nikshay-Graph

TB dropout prediction using contact-network graph propagation.

Built for the Microsoft AI Unlocked "AI for India" hackathon by IIT Madras.

---

## The problem

India has 2.4 million TB patients on treatment. The regimen takes 6 months. Most dropouts happen around week 8-10, right when patients feel better and think they are cured. A dropout that becomes drug-resistant TB costs Rs 6 lakh to treat and can infect 15 people a year. The government's Nikshay system tells you who already dropped out. It does not tell you who is about to.

## What this does

Nikshay-Graph models every patient as a node in a network connected to their household contacts, their ASHA worker, their nearest PHC, and the welfare schemes they depend on. A Temporal Graph Network learns from silence signals, missed doses, and contact vulnerability to predict dropout risk before it happens.

Every morning each ASHA worker gets a ranked visit list in their own language. The District TB Officer sees which workers are overloaded, which blocks are hotspots, and which contacts need screening that week.

## Pipeline

```
Stage 1  ->  Graph construction (Azure Cosmos DB Gremlin + Azure AI Language NER)
Stage 2  ->  Temporal Graph Network with GRU memory and graph attention
Stage 3  ->  Three-component risk score (TGN + BBN prior + ASHA load)
Stage 4  ->  Template-based explanations with Azure AI Foundry safety validation
Stage 5  ->  Multilingual briefings (Azure AI Translator + Azure AI Speech)
```

## Risk score

Each patient gets a dropout probability score combining three signals:

- **TGN output** (60%) — graph-aware, temporally-informed model
- **BBN prior** (40%, fades to 0) — logistic model from published Tamil Nadu TB literature odds ratios; retires as real dropout cases accumulate
- **ASHA load** (20%, permanent) — system-level risk from worker overload

Thresholds tighten as treatment progresses. A score of 0.60 at week 4 is MEDIUM risk. The same score at week 22 is HIGH risk.

## Stack

| Service | Purpose |
|---|---|
| Azure Cosmos DB (Gremlin API) | Patient-contact graph storage |
| Azure AI Language | NER on ASHA intake notes to extract contact names |
| Azure AI Translator | English briefings to Tamil, Telugu, Hindi, Kannada, Marathi, Gujarati, Bengali |
| Azure AI Speech | Neural TTS voice notes for ASHA workers |
| Azure AI Foundry | Content safety validation on all explanations |
| Azure Event Hubs | Real-time event stream for ASHA reply processing |
| Azure ML | TGN model endpoint (production) |
| Streamlit | Dashboard for ASHA portal and District Officer command view |

## Setup

```bash
git clone https://github.com/your-org/nikshay-graph
cd nikshay-graph
pip install -r requirements.txt
cp .env.example .env
# Fill in Azure keys in .env
python utils/dataset_gen.py          # generates data/nikshay_grounded_dataset.json
python main.py --limit 100           # runs full pipeline
streamlit run app.py                 # opens dashboard
```

## .env keys required

```
COSMOS_ENDPOINT=wss://your-account.gremlin.cosmos.azure.com:443/
COSMOS_KEY=
COSMOS_DATABASE=NikshayDB
COSMOS_GRAPH=PatientGraph
LANGUAGE_ENDPOINT=
LANGUAGE_KEY=
TRANSLATOR_KEY=
TRANSLATOR_REGION=centralindia
SPEECH_KEY=
SPEECH_REGION=centralindia
EVENTHUB_CONNECTION_STRING=
EVENTHUB_NAME=graph-events
FOUNDRY_ENDPOINT=
FOUNDRY_KEY=
```

## Project structure

```
nikshay-graph/
├── agents/
│   ├── stage1_nlp.py       # graph construction and ASHA writeback
│   ├── stage2_tgn.py       # temporal graph network
│   ├── stage3_score.py     # risk scoring
│   ├── stage4_explain.py   # template explanations and safety validation
│   └── stage5_voice.py     # translation, TTS, and briefing generation
├── utils/
│   ├── cosmos_client.py    # Cosmos DB connection and query layer
│   └── dataset_gen.py      # synthetic dataset generator (1000 patients, Tondiarpet)
├── data/
│   └── nikshay_grounded_dataset.json
├── main.py                 # pipeline orchestrator
├── app.py                  # Streamlit dashboard
└── .env
```

## Live demo

[nikshaydashboard.azurewebsites.net](https://nikshaydashboard.azurewebsites.net)

---
