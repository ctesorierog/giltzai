"""Builds out/_llm_input.csv: the listings Claude reads (status C/D/E = no valid register code), with their title and description.
Run after build_db.py and before llm_read_descriptions.py:   python3 scripts/prepare_llm_input.py
The file contains listing texts from Inside Airbnb: keep it local (do not republish)."""
import os, shutil, sqlite3
import pandas as pd
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = f'{BASE}/out'
DB = os.path.expanduser('~/giltzai_llm_tmp.gpkg'); shutil.copyfile(f'{OUT}/giltzai.gpkg', DB)   # local copy (SQLite on network/Windows mounts)
con = sqlite3.connect(DB)
q = """SELECT l.listing_id, l.exact_match_status, l.room_type, l.minimum_nights, r.name, r.description
       FROM listings l JOIN listings_raw r ON r.id = l.listing_id
       WHERE l.exact_match_status IN ('C_claims_seasonal_rental', 'D_code_not_in_register', 'E_no_code')"""
d = pd.read_sql(q, con)
d['description'] = (d.description.fillna('').str.replace('<br />', ' ', regex=False)
                    .str.replace(r'<[^>]+>', ' ', regex=True).str.slice(0, 1500))
d.to_csv(f'{OUT}/_llm_input.csv', index=False)
print(len(d), 'listings ->', f'{OUT}/_llm_input.csv', d.exact_match_status.value_counts().to_dict())
