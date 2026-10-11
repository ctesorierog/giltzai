# GiltzAI data pipeline

These scripts rebuild the app's data bundle (`app/data/`) from the original open-data files.
The raw files are **not** in this repository: Inside Airbnb's data policy says "Do not republish the data". Download them yourself with the steps below.

Extra packages (on top of `app/requirements.txt`):

```
pip install geopandas shapely rapidfuzz scikit-learn openpyxl numpy
```

Run every command from the repository root (`GiltzAI/`). On Windows use `python` and `\` where this page shows `python3` and `/`.

## 1. Download the raw data into `raw/`

`build_db.py` expects exactly these file names. The URLs, snapshot dates and sha256 checksums of the files we used are in `app/data/sources.csv`. After downloading, compare each file's sha256 with that table. Live portals change, so a newer download gives slightly different figures.

| id | URL | save as |
|---|---|---|
| S1a | https://data.insideairbnb.com/spain/pv/euskadi/2026-09-25/data/listings.csv.gz | `raw/insideairbnb/iab_euskadi_listings.csv.gz` |
| S1b | https://data.insideairbnb.com/spain/pv/euskadi/2026-09-25/data/calendar.csv.gz | `raw/insideairbnb/iab_euskadi_calendar.csv.gz` |
| S1c | https://data.insideairbnb.com/spain/pv/euskadi/2026-09-25/data/reviews.csv.gz | `raw/insideairbnb/iab_euskadi_reviews.csv.gz` |
| S1d | https://data.insideairbnb.com/spain/pv/euskadi/2026-09-25/visualisations/neighbourhoods.geojson | `raw/insideairbnb/iab_euskadi_neighbourhoods.geojson` |
| S2a | https://opendata.euskadi.eus/contenidos/ds_recursos_turisticos/habitaciones_viviendas_turisti/opendata/viviendas.xlsx | `raw/register/reate_viviendas.xlsx` |
| S2b | https://opendata.euskadi.eus/contenidos/ds_recursos_turisticos/habitaciones_viviendas_turisti/opendata/habitaciones.xlsx | `raw/register/reate_habitaciones.xlsx` |
| S3 | https://www.donostia.eus/datosabiertos/recursos/censo-viviendas-turisticas/urb_ckan_vtur_censo.csv | `raw/donostia/dss_urb_ckan_vtur_censo.csv` |
| S4a | https://www.donostia.eus/datosabiertos/recursos/mapa_auzoak/auzoak.json | `raw/donostia/dss_auzoak.json` |
| S4b | https://www.donostia.eus/datosabiertos/recursos/mapa_unidades_menores/unitatetxikiak.json | `raw/donostia/dss_unitatetxikiak.json` |
| S5 | https://www.donostia.eus/datosabiertos/recursos/direcciones_activas/indarrekohelbideak.csv | `raw/donostia/dss_indarrekohelbideak.csv` |
| S6a | https://www.donostia.eus/datosabiertos/recursos/eustat_renta/eustatrentabarrio.csv | `raw/donostia/dss_eustatrentabarrio.csv` |
| S7 | https://www.donostia.eus/datosabiertos/recursos/demografia-datosgenerales/demografiadatosgeneralesbarrio.csv | `raw/donostia/dss_demografiadatosgeneralesbarrio.csv` |

Example on Linux or macOS:

```
mkdir -p raw/insideairbnb raw/register raw/donostia
curl -L -o raw/insideairbnb/iab_euskadi_listings.csv.gz https://data.insideairbnb.com/spain/pv/euskadi/2026-09-25/data/listings.csv.gz
# ... one line per row of the table ...
sha256sum raw/*/*
```

Inside Airbnb keeps only recent snapshots online. If the 2026-09-25 URL no longer works, use the current Euskadi snapshot from https://insideairbnb.com/get-the-data/ and expect different figures.

## 2. Run order

| step | command | reads | writes |
|---|---|---|---|
| 1 | `python3 scripts/build_db.py` | `raw/**`, `scripts/qa_views.sql` | `out/giltzai.gpkg` (all tables, QA views), `out/labels_to_fill.csv` (labelling sheet, has listing URLs: **private, never share**), `out/build_stats.json`, `secret_salt.txt` (host-hash salt, **private**) |
| 2 | `python3 scripts/check_and_document.py` | `out/giltzai.gpkg`, raw listings file | `out/hand_checks.md` (private: traces listings to raw rows), `out/schema.sql`, `out/data_dictionary.md` |
| 3 | `python3 scripts/llm_read_descriptions.py out/_llm_input.csv out/llm_text_signals.csv` (needs `ANTHROPIC_API_KEY`) | `out/_llm_input.csv` | `out/llm_text_signals.csv` |
| 4 | `python3 scripts/model.py --runs 1000` | `out/giltzai.gpkg`, `out/llm_text_signals.csv`, hand labels `out/GiltzAI_labels_80_v2.xlsx` + `out/label_corrections.csv` (optional) | tables `listing_features`, `listing_scores`, `zone_estimate`, `zone_mc_ranks`, `benchmark` in the GeoPackage + `out/model_results.json` |
| 5 | `python3 scripts/export_app_data.py` | `out/giltzai.gpkg`, `out/model_results.json` | `out/app/data/` → copy those files into `app/data/` |

`qa_views.sql` defines QA views that must return 0 rows. `build_db.py` loads it.

Notes:
- **Step 3 input.** `out/_llm_input.csv` holds the listing descriptions, so it is not shipped. Rebuild it with `python3 scripts/prepare_llm_input.py` (reads the GeoPackage) with the Donostia listings whose status is `C_claims_seasonal_rental`, `D_code_not_in_register` or `E_no_code` (413 rows in our snapshot). It has the columns `listing_id, exact_match_status, room_type, minimum_nights, name, description`, taken from tables `listings` and `listings_raw`. If you skip step 3, `model.py` sets the AI text signals to 0.
- **Hand labels.** The labels (52 listings) are not shipped because the labelling sheet contains listing URLs. Without them `model.py` uses the documented prior rules and skips the listing-level benchmark.
- The scripts copy the GeoPackage to the home folder while they work, because SQLite locking fails on some mounted drives.
- Never publish `raw/`, `out/`, `secret_salt.txt` or any `.gpkg`. `.gitignore` excludes them.
