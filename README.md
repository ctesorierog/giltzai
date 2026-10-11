# GiltzAI · Donostia Meeting Minds 2026 (City category)

GiltzAI helps Donostia / San Sebastián decide **where to send its tourist-flat inspectors first**.
It matches 1,626 Airbnb listings against the official Basque register (REATE). Claude reads the descriptions of the 413 listings that have no valid code, and the app estimates by barrio how many entire homes are probably **not in the register** (about 345, range 330–360).
It then turns that estimate into an inspection plan, a day-by-day route and a field log that re-plans after each real check.
It works **at barrio level only**. "Unregistered" means "not found in the register", not "illegal", and every case needs an official inspection.
A chat assistant ("Ask GiltzAI", in Spanish, Basque or English) answers only with figures from the app's data tools.

**Team GiltzAI:** Carlos Tesoriero · Marcos Tesoriero

## Repository layout

```
app/                  Streamlit app (entry point: app/app.py)
  agent.py            Claude agent + guardrails (safety filter, scope rules)
  giltzai_tools.py    data tools used by the app and the agent
  data/               de-identified data bundle (no listing ids, URLs, hosts, addresses or coordinates)
  tests/              red-team cases + runner (python app/tests/run_redteam.py --offline)
  requirements.txt
  .streamlit/config.toml
scripts/              data pipeline that rebuilds app/data from the raw open data (see scripts/README.md)
.streamlit/config.toml           theme (Streamlit Cloud reads it from the repo root)
.streamlit/secrets.toml.example  names of the secrets (reference only)
```

## Run it locally (Windows, Anaconda)

1. Open **Anaconda Prompt** (Start menu → type "Anaconda Prompt").
2. Go to the folder: `cd %USERPROFILE%\Downloads\GiltzAI`
3. First time only: `pip install -r app\requirements.txt`
4. Start: `streamlit run app\app.py` → the browser opens at http://localhost:8501 (stop with Ctrl+C).
5. For the chat, create a file `GiltzAI\.env` with one line `ANTHROPIC_API_KEY=sk-ant-...`, then restart. **Never share or commit this file.**
   The map, comparison, plan, route and field-log tabs work without a key.
6. Free guardrail checks (no API calls): `python app\tests\run_redteam.py --offline` → must end with `OFFLINE ALL PASS`.

Python 3.11–3.13 work (tested with 3.12 and 3.13).

## Public demo (Streamlit Community Cloud)

The app is deployed from a **private** GitHub repository. Three settings are pasted as secrets in the Cloud dashboard, never in the code:

| Secret | Effect |
|---|---|
| `ANTHROPIC_API_KEY` | enables the chat. Delete it to switch the chat off. |
| `GILTZAI_DEMO_MODE = "1"` | each visitor gets their own temporary field log, so visitors never see each other's test entries and nothing is stored. |
| `GILTZAI_CHAT_PASSCODE` | the chat asks for this passcode before it calls the paid API. All other tabs are open. |

The API key is a prepaid key with $5 of credit and auto-reload off, so the maximum cost is capped.

## Data, licences and attribution

| Source | Licence | In this repo? |
|---|---|---|
| Inside Airbnb, Euskadi snapshot 2026-09-25 | CC BY 4.0. Its data policy says "Do not republish the data". | **No raw files.** Only the URLs, dates and sha256 (`app/data/sources.csv`) and derived, de-identified per-listing flags with no ids, coordinates or hosts (`app/data/listing_flags.csv`). |
| Gobierno Vasco · Open Data Euskadi, REATE tourist-flat register | CC BY 4.0 | No raw files. Only derived barrio counts. |
| Ayuntamiento de Donostia · datos abiertos (city census of tourist flats, barrio boundaries, postal addresses, Eustat income by barrio) | CC BY-SA 3.0 | Derived barrio tables and simplified barrio polygons. **GiltzAI's derived barrio tables (`app/data/zone_estimate.csv`, `app/data/zones.geojson`) are shared under CC BY-SA 3.0.** |
| Basemap | © OpenStreetMap contributors © CARTO | Loaded at run time |

Attribution line: *Inside Airbnb (CC BY 4.0) · Gobierno Vasco – Open Data Euskadi (CC BY 4.0) · Ayuntamiento de Donostia – datos abiertos (CC BY-SA 3.0) · basemap © OpenStreetMap contributors © CARTO.*

## AI declaration

Claude (Anthropic) is used in three ways:
1. as the **runtime agent** behind "Ask GiltzAI" (it calls the app's data tools and adds no figures of its own);
2. to **read the descriptions of 413 listings** without a valid register code (`scripts/llm_read_descriptions.py`). It extracts what the host declares, such as a mid-term rental or a hotel-like business, and **never decides legality**;
3. as a **coding and writing assistant** for the code, the documentation and the report.

All figures shown come from the data bundle and are computed by code at run time.

## Privacy rules

- Results are at barrio level only. Barrios with fewer than 5 listings are hidden.
- The app, the data bundle and the chat contain no addresses, listing ids or URLs, host names, register codes or coordinates. Host ids were salted and hashed in the pipeline.
- The agent refuses requests about individual flats or hosts. A post-answer filter removes codes, URLs, street addresses and coordinates.
- Secrets (`.env`, `.streamlit/secrets.toml`), raw data, labelling files and the full GeoPackage are excluded by `.gitignore` and are not in this repository.
