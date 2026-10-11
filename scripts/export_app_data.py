"""Exports the light, de-identified data bundle the app/agent uses (no geopandas needed on the user's PC).
out/app/data/: zones.geojson (simplified polygons), zone_estimate.csv, listing_flags.csv (NO ids, NO coordinates, NO hosts),
sources.csv, model_results.json, summary.json (figures the agent can verify with check_figure)."""
import os, json, shutil, sqlite3
import pandas as pd, geopandas as gpd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = f'{BASE}/out'; D = f'{OUT}/app/data'
os.makedirs(D, exist_ok=True)
DB = os.path.expanduser('~/giltzai_export_tmp.gpkg'); shutil.copyfile(f'{OUT}/giltzai.gpkg', DB)
con = sqlite3.connect(DB)
z = gpd.read_file(DB, layer='zones')[['zone_code', 'zone_name', 'n_postal_addresses', 'income_personal_eur', 'income_year', 'area_km2', 'geometry']]
ze = pd.read_sql('SELECT * FROM zone_estimate', con)
z = z.merge(ze[['zone_code', 'expected_unregistered', 'unreg_p10', 'unreg_p90', 'rank_ai', 'rank_naive', 'hit_rate', 'publishable', 'listings_in_scope']], on='zone_code', how='left')
z['geometry'] = z.to_crs(25830).simplify(15).to_crs(4326)
_tmp = os.path.expanduser('~/zones_tmp.geojson')
if os.path.exists(_tmp): os.remove(_tmp)
z.to_file(_tmp, driver='GeoJSON'); shutil.copyfile(_tmp, f'{D}/zones.geojson')   # overwrite (no delete needed on the Windows mount)
ze.to_csv(f'{D}/zone_estimate.csv', index=False)
f = pd.read_sql('SELECT * FROM listing_features', con).merge(pd.read_sql('SELECT listing_id, p_unreg FROM listing_scores', con), on='listing_id', how='left', suffixes=('', '_s'))
L = pd.read_sql('SELECT listing_id, room_type, exact_match_status, baseline_exact_field FROM listings', con)
f = f.merge(L, on='listing_id')
f['zone_name'] = f.zone_code.map(z.set_index('zone_code').zone_name)
keep = ['zone_code', 'zone_name', 'room_type', 'exact_match_status', 'baseline_exact_field', 'in_scope', 'entire_home', 'has_code',
        'code_in_register', 'seasonal_claim', 'code_not_in_register', 'no_code', 'code_shared', 'exact_far_or_size_mismatch', 'active_ltm',
        'calendar_blocked', 'host_5plus', 'border_risk', 'superhost', 'new_listing', 'min_nights_31plus', 'seasonal_but_short_stays', 'host_mid_4_10', 'llm_midterm_claim', 'llm_hotel_like', 'any_midterm_claim', 'p_prior', 'p_unreg']
flags = f[keep].sample(frac=1, random_state=1).reset_index(drop=True)   # shuffled, no identifiers
flags.to_csv(f'{D}/listing_flags.csv', index=False)
pd.read_sql('SELECT source_id, name, owner, url, licence, snapshot_date, reliability, sha256 FROM sources', con).to_csv(f'{D}/sources.csv', index=False)
shutil.copyfile(f'{OUT}/model_results.json', f'{D}/model_results.json')
reg = pd.read_sql('SELECT COUNT(*) n FROM register', con).n[0]; city = pd.read_sql('SELECT COUNT(*) n FROM register_city', con).n[0]
ent = flags[flags.room_type == 'Entire home/apt']
summary = {
    'listings_total': int(len(flags)), 'entire_homes': int(len(ent)), 'in_scope_listings': int(flags.in_scope.sum()),
    'entire_homes_matched_register': int((ent.exact_match_status == 'A_matched_register').sum()),
    'entire_homes_no_valid_code': int(ent.exact_match_status.isin(['D_code_not_in_register', 'E_no_code']).sum()),
    'entire_homes_seasonal_claim': int((ent.exact_match_status == 'C_claims_seasonal_rental').sum()),
    'entire_homes_naive_exact_match': int(ent.baseline_exact_field.astype(bool).sum()),
    'register_units_donostia': int(reg), 'city_census_units': int(city),
    'expected_unregistered_total': json.load(open(f'{OUT}/model_results.json'))['expected_unregistered_total'],
    'ai_text_read_listings': int(((flags.exact_match_status.isin(['C_claims_seasonal_rental', 'D_code_not_in_register', 'E_no_code']))).sum()),
    'ai_text_hotel_like_excluded': int(flags.llm_hotel_like.sum()),
    'ai_text_midterm_claims': int(flags.llm_midterm_claim.sum()),
    'midterm_claim_but_short_stays': int((flags.seasonal_but_short_stays == 1).sum()),
    'snapshot_listings': '2026-09-25 (scraped 28-29 Sep 2026)', 'snapshot_register': '2026-10-06', 'snapshot_city_census': '2026-10-04',
}
json.dump(summary, open(f'{D}/summary.json', 'w'), indent=1)
APP = f'{BASE}/app/data'; os.makedirs(APP, exist_ok=True)
for fn in os.listdir(D): shutil.copyfile(f'{D}/{fn}', f'{APP}/{fn}')   # keep the app bundle in sync
print(json.dumps(summary, indent=1)); print(os.listdir(D))
