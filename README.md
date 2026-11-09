# Agentic AI Travel Planning Assistant (LangChain + Streamlit)

An AI travel agent that plans multi-day trips using LangChain tools:
flight search, hotel recommendation, places discovery, live weather (Open-Meteo) and budget estimation.

## Project structure
```
travel_agent/
├── app.py                 # tools + agent + Streamlit UI
├── requirements.txt
├── data/                  # put flights.json, hotels.json, places.json here
└── .streamlit/secrets.toml.example
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Choose a provider (OpenAI / Groq free tier / Gemini) and paste an API key in the sidebar,
or select **Offline (no LLM)** mode which needs no key.

## Datasets
Download `flights.json`, `hotels.json`, `places.json` from the Drive link in the project brief
and place them in `data/`. Field names are auto-detected (e.g. `price`/`fare`, `from`/`source`,
`type`/`category`). If a file is missing, small built-in sample data is used.
You can also upload the files from the app sidebar.

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repository (include the `data/` JSON files).
2. Go to https://share.streamlit.io -> **New app** -> pick the repo, branch, and `app.py`.
3. In **Advanced settings -> Secrets** add e.g. `OPENAI_API_KEY = "sk-..."`
   (or `GROQ_API_KEY`, `GOOGLE_API_KEY`).
4. Deploy.

## Agent workflow
User query -> understand -> flight tool -> hotel tool -> places tool -> weather tool -> budget tool
-> final structured itinerary (Markdown + JSON) with reasoning ("Why we selected this").
