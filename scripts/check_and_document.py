"""GiltzAI Phase 1 - Agent 3 QA + docs.
1) Hand checks: 5 listings traced back to the RAW gz file with an independent reader (csv module), compared with the DB.
2) comprobar_cifra: the headline figures are recomputed from the raw files without using build_db.py code.
3) Writes out/schema.sql (from sqlite_master) and out/data_dictionary.md.
Run after build_db.py:  python3 scripts/check_and_document.py"""
import os, csv, gzip, re, json, sqlite3, shutil, hashlib
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
R, OUT = f'{BASE}/raw', f'{BASE}/out'
TMP = os.path.expanduser('~/giltzai_check_tmp.gpkg')
shutil.copyfile(f'{OUT}/giltzai.gpkg', TMP)  # read a local copy (Windows mount + SQLite)
con = sqlite3.connect(TMP)
csv.field_size_limit(10**8)

# ---------------------------------------------------------------- 1) hand checks
L = pd.read_sql('SELECT listing_id, zone_name, room_type, latitude, longitude, code_best, exact_match_status, host_hash FROM listings', con)
pick = []
for st in ['A_matched_register', 'E_no_code', 'D_code_not_in_register', 'C_claims_seasonal_rental', 'B_other_licence_type']:
    pick.append(L[L.exact_match_status == st].sort_values('listing_id').iloc[len(L[L.exact_match_status == st]) // 2])
pick = pd.DataFrame(pick)
want = set(pick.listing_id.astype(str))
raw = {}
with gzip.open(f'{R}/insideairbnb/iab_euskadi_listings.csv.gz', 'rt', encoding='utf-8', newline='') as f:
    for row in csv.DictReader(f):
        if row['id'] in want:
            raw[row['id']] = row
reg = pd.read_sql('SELECT register_id, address_raw FROM register', con).set_index('register_id')
lines = ['# GiltzAI · hand checks (listing traced back to the raw Inside Airbnb file)', '',
         'Independent reader: Python `csv` module on `raw/insideairbnb/iab_euskadi_listings.csv.gz` (not the pandas pipeline).', '',
         '| # | listing_id | status in DB | raw room_type = DB | raw lat/lon = DB (±1e-6) | raw licence (first 60 chars) | DB code | code in REATE? | verdict |',
         '|---|---|---|---|---|---|---|---|---|']
ok_all = True
for i, (_, p) in enumerate(pick.iterrows(), 1):
    r = raw[str(p.listing_id)]
    rt = r['room_type'] == p.room_type
    ll = abs(float(r['latitude']) - p.latitude) < 1e-6 and abs(float(r['longitude']) - p.longitude) < 1e-6
    lic = (r['license'] or '').replace('<br />', ' / ')
    # manual code read: the 8-char regional code as written in the raw licence text
    m = re.search(r'Regional registration number / ([A-Za-z]{3}[\s-]?\d+)', lic)
    in_reate = p.code_best in reg.index if isinstance(p.code_best, str) else False
    code_ok = (p.exact_match_status == 'A_matched_register') == in_reate
    verdict = rt and ll and code_ok
    ok_all &= verdict
    lines.append(f'| {i} | {p.listing_id} | {p.exact_match_status} | {"yes" if rt else "NO"} | {"yes" if ll else "NO"} | {lic[:60]} | {p.code_best} | '
                 f'{"yes: " + reg.loc[p.code_best].address_raw.split("(")[0].strip() if in_reate else "no"} | {"MATCH" if verdict else "MISMATCH"} |')
lines += ['', f'**Result: {"5/5 match" if ok_all else "MISMATCH - investigate"}.**', '']

# ---------------------------------------------------------------- 2) comprobar_cifra (independent recompute from raw)
rows = []
with gzip.open(f'{R}/insideairbnb/iab_euskadi_listings.csv.gz', 'rt', encoding='utf-8', newline='') as f:
    for row in csv.DictReader(f):
        nb = row['neighbourhood_cleansed']
        if nb.startswith('Donostia'):
            rows.append(row)
reate = pd.concat([pd.read_excel(f'{R}/register/reate_viviendas.xlsx'), pd.read_excel(f'{R}/register/reate_habitaciones.xlsx')])
codes = set(reate[reate.Municipio.astype(str).str.contains('Donostia')]['Nº registro'].str.upper())
def any_code(row):
    s = (row['license'] + ' ' + row['name'] + ' ' + row['description']).upper().replace(' ', '').replace('-', '')
    found = set()
    for m in re.finditer(r'([EL]SS)(\d{3,6})', s):
        d = m.group(2)[:5] if len(m.group(2)) == 6 else m.group(2)
        found.add(f'{m.group(1)}{int(d):05d}')
    return found
ent = [r for r in rows if r['room_type'] == 'Entire home/apt']
ent_match = sum(1 for r in ent if any_code(r) & codes)
db = dict(con.execute("SELECT exact_match_status, COUNT(*) FROM listings WHERE room_type='Entire home/apt' GROUP BY 1").fetchall())
fig = {
    'donostia_listings (raw csv)': (len(rows), con.execute('SELECT COUNT(*) FROM listings').fetchone()[0]),
    'entire_homes (raw csv)': (len(ent), con.execute("SELECT COUNT(*) FROM listings WHERE room_type='Entire home/apt'").fetchone()[0]),
    'entire homes whose text contains a Donostia REATE ESS/LSS code': (ent_match, db.get('A_matched_register', 0)),
    'REATE Donostia units': (len(codes), con.execute('SELECT COUNT(*) FROM register').fetchone()[0]),
    'city census units (raw csv rows)': (len(pd.read_csv(f'{R}/donostia/dss_urb_ckan_vtur_censo.csv')), con.execute('SELECT COUNT(*) FROM register_city').fetchone()[0]),
}
lines += ['# comprobar_cifra · headline figures recomputed from raw files', '',
          'Simple independent rule (regex on licence + title + description); small differences are expected where the pipeline is stricter.', '',
          '| figure | independent recompute | database | equal? |', '|---|---|---|---|']
for k, (a, b) in fig.items():
    lines.append(f'| {k} | {a} | {b} | {"yes" if a == b else f"diff {a - b:+d}"} |')
open(f'{OUT}/hand_checks.md', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')

# ---------------------------------------------------------------- 3) schema.sql + data dictionary
ddl = [r[0] + ';' for r in con.execute("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'gpkg_%' AND name NOT LIKE 'rtree_%' "
                                        "AND name NOT LIKE 'sqlite_%' ORDER BY type DESC, name").fetchall()]
open(f'{OUT}/schema.sql', 'w', encoding='utf-8').write('-- GiltzAI giltzai.gpkg schema (generated from sqlite_master by check_and_document.py)\n'
                                                     '-- Spatial layers are GeoPackage feature tables (EPSG:4326); qa_* views must all return 0 rows.\n\n' + '\n\n'.join(ddl) + '\n')
DESC = {
    'sources': 'One row per raw file: URL, owner, licence, snapshot date, download time, bytes, sha256.',
    'zones': '19 official Donostia barrios (S4a) + principal postal addresses (S5, proxy denominator), Eustat income (S6a). population/dwellings NULL until the padron feed works.',
    'zones_unidad_menor': '105 official sub-barrio units (S4b), for finer maps later.',
    'listings_raw': 'Donostia rows of the Inside Airbnb detailed file, all columns EXCEPT host name/about/location/urls/photos/listing url; host_id replaced by host_hash.',
    'listings': 'Clean Donostia listings (1 row per listing): zone, border distance, licence fields, best normalised code, exact-match status A-E.',
    'register_raw': 'REATE rows for Donostia (S2a+S2b) minus phone, e-mail, web, commercial name.',
    'register': 'REATE Donostia, normalised code, parsed + geocoded address (via S5), zone, link to city census.',
    'register_city': 'Municipal census of tourist flats/rooms (S3), parsed + geocoded, zone (reported vs geocoded).',
    'listing_zone': 'Listing -> barrio (point in polygon; 3 nearest), distance to barrio border, border_risk_150m flag (coordinate offset).',
    'code_candidates': 'Every code-like string found per listing and where it came from (regional field, national number, plain field, free text).',
    'match_candidates': 'Listing x register candidate pairs (code similarity >= 75 or distance <= 250 m), top 10 per listing. Candidate generation only, NOT the model.',
    'labels': 'Empty ground-truth table for the 150 hand-labelled listings (filled from out/labels_to_fill.csv).',
    'zone_baseline': 'Naive per-barrio counts: listings, entire homes, exact-match status, REATE and city-census units, listings per 100 addresses, naive ranks.',
}
COL = {
    'exact_match_status': 'A matched REATE Donostia code | B other licence type (hotel/hostel/agrotourism) | C claims seasonal-rental exemption | D code-like string not in REATE | E no code anywhere',
    'code_best': 'Best normalised code (LLL99999). Priority: regional field > national number suffix > plain field > free text',
    'host_hash': 'salted SHA-256 of host_id (16 hex chars); salt kept outside the deliverables',
    'border_risk_150m': 'TRUE if the (offset) point lies < 150 m from its barrio border -> zone assignment uncertain',
    'geocode_quality': 'exact | number_no_bis | nearest_number (same street, nearest portal) | unmatched',
    'in_city_census': 'REATE unit found in the municipal census at the same street + portal + floor/door',
    'portal_in_city_census': 'same street + portal found in the municipal census (floor/door may differ)',
    'heuristic_score': '0.6*code_similarity/100 + 0.4*(1 - distance/250 m), only to order candidates for labelling',
    'n_postal_addresses': 'principal active postal addresses in the barrio (S5); temporary proxy for housing stock',
}
dd = ['# GiltzAI · data dictionary (giltzai.gpkg)', '', 'Generated by `scripts/check_and_document.py`. Rebuild everything with `python3 scripts/build_db.py`.', '']
for t in [r[0] for r in con.execute("SELECT table_name FROM gpkg_contents ORDER BY data_type DESC, table_name").fetchall()]:
    n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
    dd += [f'## `{t}` · {n:,} rows', DESC.get(t, ''), '', '| column | type | note |', '|---|---|---|']
    for _, name, typ, *_ in con.execute(f'PRAGMA table_info("{t}")').fetchall():
        dd.append(f'| {name} | {typ} | {COL.get(name, "")} |')
    dd.append('')
dd += ['## QA views', 'All `qa_*` views must return 0 rows: ' + ', '.join(f'`{r[0]}`' for r in con.execute("SELECT name FROM sqlite_master WHERE type='view' AND name LIKE 'qa_%' ORDER BY 1").fetchall()), '']
open(f'{OUT}/data_dictionary.md', 'w', encoding='utf-8').write('\n'.join(dd))
con.close()
print('\n'.join(lines))
