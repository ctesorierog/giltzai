"""GiltzAI Phase 2 - model.
Listing level : features -> P(unregistered)   (prior rules now; logistic regression once out/GiltzAI_labels_80.xlsx is labelled)
Zone level    : Monte Carlo (location jitter <= 150 m  x  Bernoulli(P)) -> expected unregistered units per barrio + P10-P90 range
Benchmarks    : listing matcher vs naive exact-field rule (precision / recall on the hand labels)
                zone ranking vs naive "count listings per barrio" (labelled not_found listings covered by the top-k barrios)
Writes tables into out/giltzai.gpkg (listing_features, listing_scores, zone_estimate, zone_mc_ranks, benchmark) + out/model_results.json
Run:  python3 scripts/model.py [--runs 1000]"""
import os, sys, json, shutil, sqlite3, argparse
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Point

ap = argparse.ArgumentParser(); ap.add_argument('--runs', type=int, default=1000); args = ap.parse_args()
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = f'{BASE}/out'
DB_FINAL = f'{OUT}/giltzai.gpkg'
DB = os.path.expanduser('~/giltzai_model_tmp.gpkg')
shutil.copyfile(DB_FINAL, DB)
con = sqlite3.connect(DB)
rng = np.random.default_rng(20261011)
SCOPE_ROOMS = ('Entire home/apt', 'Private room')
JITTER_M = 150

# ------------------------------------------------------------------ load
L = gpd.read_file(DB, layer='listings')
Z = gpd.read_file(DB, layer='zones')[['zone_code', 'zone_name', 'n_postal_addresses', 'geometry']]
M = pd.read_sql('SELECT * FROM match_candidates', con)
REG = gpd.read_file(DB, layer='register')

# ------------------------------------------------------------------ features (one row per listing)
F = pd.DataFrame({'listing_id': L.listing_id})
st = L.exact_match_status
F['in_scope'] = L.room_type.isin(SCOPE_ROOMS) & (st != 'B_other_licence_type')
F['entire_home'] = (L.room_type == 'Entire home/apt').astype(int)
F['has_code'] = L.code_best.notna().astype(int)
F['code_in_register'] = (st == 'A_matched_register').astype(int)
F['seasonal_claim'] = (st == 'C_claims_seasonal_rental').astype(int)
F['code_not_in_register'] = (st == 'D_code_not_in_register').astype(int)
F['no_code'] = (st == 'E_no_code').astype(int)
F['code_from_text_only'] = (L.code_best_origin == 'free_text').astype(int)
F['code_nonstandard_format'] = ((~L.code_format_standard.astype(bool)) & L.code_best.notna()).astype(int)
F['code_shared'] = (L.code_shared_n_listings >= 2).astype(int)
F['several_codes'] = (L.n_distinct_codes >= 2).astype(int)
ex = M[M.code_exact.astype(bool)].sort_values('distance_m').groupby('listing_id').first()
F['exact_cand_distance_m'] = F.listing_id.map(ex.distance_m)
F['exact_cand_cap_diff'] = (F.listing_id.map(ex.register_capacity).astype(float) - F.listing_id.map(ex.listing_accommodates).astype(float)).abs()
ex_reg_cap = F.listing_id.map(ex.register_capacity).astype(float); ex_acc = F.listing_id.map(ex.listing_accommodates).astype(float)
room_lic = F.listing_id.map(ex.register_id).astype(str).str.startswith('L')
size_bad = np.where(room_lic, ex_acc > ex_reg_cap, (ex_reg_cap - ex_acc).abs() > 2)   # room licence: capacity is the whole home's total
F['exact_far_or_size_mismatch'] = ((F.exact_cand_distance_m > 500) | size_bad).astype(int)
F['best_code_similarity'] = F.listing_id.map(M.groupby('listing_id').code_similarity.max()).fillna(0)
F['active_ltm'] = (L.number_of_reviews_ltm > 0).astype(int)
F['calendar_blocked'] = (L.availability_365 == 0).astype(int)
F['host_5plus'] = (L.calculated_host_listings_count >= 5).astype(int)
# Boto-Garcia (2025, Tourism Economics 31(4)): illegality peaks among mid-size hosts (4-5 listings in Menorca); Donostia peaks at 4-10
F['host_mid_4_10'] = L.calculated_host_listings_count.between(4, 10).astype(int)
F['border_risk'] = L.border_risk_150m.astype(int)
F['zone_code'] = L.zone_code
# behaviour signals (exploratory analysis 10-Oct: no-code / unmatched entire homes are newer, have fewer reviews and fewer superhosts;
# ratings do NOT differ; 89% of seasonal-rental claims require >= 31 nights, i.e. the claim is usually consistent)
raw = pd.read_sql('SELECT id AS listing_id, host_is_superhost FROM listings_raw', con)
F['superhost'] = F.listing_id.map(raw.set_index('listing_id').host_is_superhost.eq('t')).fillna(False).astype(int)
fr = pd.to_datetime(L.first_review, errors='coerce')
F['new_listing'] = ((L.number_of_reviews == 0) | ((pd.Timestamp('2026-09-25') - fr).dt.days < 365)).astype(int).values
F['min_nights_31plus'] = (L.minimum_nights >= 31).astype(int).values
# AI text reading (scripts/llm_read_descriptions.py, Claude): what hosts declare in the description of listings without a valid code
llm_p = f'{OUT}/llm_text_signals.csv'
if os.path.exists(llm_p):
    T = pd.read_csv(llm_p, dtype={'listing_id': 'int64'}).set_index('listing_id')
    tb = lambda c: F.listing_id.map(T[c].astype(str).str.lower().eq('true')).fillna(False).astype(int)
    F['llm_midterm_claim'] = tb('midterm_or_non_tourist_claim'); F['llm_hotel_like'] = tb('hotel_like')
else:
    F['llm_midterm_claim'] = 0; F['llm_hotel_like'] = 0
F['any_midterm_claim'] = ((F.seasonal_claim == 1) | (F.llm_midterm_claim == 1)).astype(int)
F['seasonal_but_short_stays'] = ((F.any_midterm_claim == 1) & (F.min_nights_31plus == 0)).astype(int)   # claims mid-term but accepts short stays
F.loc[F.llm_hotel_like == 1, 'in_scope'] = False      # hotels / guesthouses / residences belong to other registers

# ------------------------------------------------------------------ P(unregistered): prior rules (documented assumptions, replaced by the fitted model when labels exist)
def prior(r):
    if not r.in_scope:
        return 0.0
    if r.code_in_register:
        return 0.30 if r.exact_far_or_size_mismatch == 1 else 0.03
    if r.seasonal_claim or r.llm_midterm_claim:
        return 0.15 if r.min_nights_31plus else 0.65   # claim consistent with >= 31-night minimum stay vs claim but short stays allowed
    if r.code_not_in_register:
        return 0.75
    p = 0.85                       # no code anywhere
    if r.calendar_blocked and not r.active_ltm:
        p = 0.45                   # probably inactive / not bookable
    return p
F['p_prior'] = F.apply(prior, axis=1)

# ------------------------------------------------------------------ labels -> fitted model + listing benchmark
FEATS = ['entire_home', 'has_code', 'code_in_register', 'seasonal_claim', 'code_not_in_register', 'no_code', 'code_from_text_only',
         'code_nonstandard_format', 'code_shared', 'several_codes', 'exact_far_or_size_mismatch', 'active_ltm', 'calendar_blocked', 'host_5plus',
         'superhost', 'new_listing', 'seasonal_but_short_stays', 'host_mid_4_10', 'llm_midterm_claim']
bench, model_used = {}, 'prior_rules'
lab_path = next((p for p in (f'{OUT}/GiltzAI_labels_80_v2.xlsx', f'{OUT}/GiltzAI_labels_80.xlsx') if os.path.exists(p)), None)
labels = None
if lab_path:
    x = pd.read_excel(lab_path, sheet_name='Label here', dtype=str)
    link_col = next(c for c in x.columns if str(c).startswith('Listing link'))
    x['listing_id'] = x[link_col].str.extract(r'/rooms/(\d+)', expand=False).astype('int64')
    x['label'] = x['YOUR LABEL'].fillna('').str.strip().str.lower()
    x['who'] = x['Who'].fillna('').str.lower().replace({'you': 'human'}) if 'Who' in x else 'human'
    corr_p = f'{OUT}/label_corrections.csv'
    if os.path.exists(corr_p):
        corr = pd.read_csv(corr_p, dtype=str)
        for _, cr in corr.iterrows():
            x.loc[x['orig #'].astype(str) == cr.orig_no, 'label'] = cr.new_label
        bench['label_corrections'] = len(corr)
    lab_all = x[x.label.isin(['registered', 'not_found', 'unclear'])]
    if len(lab_all) >= 30:
        hum = lab_all[lab_all.who == 'human']
        dup = hum[hum.listing_id.duplicated(keep=False)].groupby('listing_id').label.agg(lambda s: s.nunique() == 1)
        bench['labels_file'] = os.path.basename(lab_path)
        bench['labelled_rows'] = int(len(lab_all)); bench['rows_by_who'] = lab_all.who.value_counts().to_dict()
        bench['unlabelled_rows'] = int((x.label == '').sum())
        bench['self_agreement'] = f'{int(dup.sum())}/{len(dup)} (repeated rows labelled by hand)'
        labels = lab_all.drop_duplicates('listing_id', keep='first')
        bench['labels'] = labels.label.value_counts().to_dict()
        bench['labels_human_only'] = labels[labels.who == 'human'].label.value_counts().to_dict()
        y_df = labels[labels.label != 'unclear'].merge(F, on='listing_id')
        y = (y_df.label == 'not_found').astype(int).values
        # baseline: naive exact match of the regional field (registered if exact) ; prior-rules matcher ; fitted model (leave-one-out)
        base = L.set_index('listing_id').baseline_exact_field.reindex(y_df.listing_id).astype(bool).values
        def prf(pred_unreg, y):
            tp = int(((pred_unreg == 1) & (y == 1)).sum()); fp = int(((pred_unreg == 1) & (y == 0)).sum()); fn = int(((pred_unreg == 0) & (y == 1)).sum())
            tn = int(((pred_unreg == 0) & (y == 0)).sum())
            p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
            return dict(precision=round(p, 3), recall=round(r, 3), f1=round(2 * p * r / (p + r), 3) if p + r else 0, accuracy=round((tp + tn) / len(y), 3), tp=tp, fp=fp, fn=fn, tn=tn)
        bench['n_eval'] = int(len(y)); bench['n_not_found'] = int(y.sum())
        bench['baseline_exact_field'] = prf((~base).astype(int), y)
        bench['ai_prior_rules'] = prf((y_df.p_prior.values >= 0.5).astype(int), y)
        hm = (y_df.who == 'human').values
        if hm.sum() >= 10 and 0 < y[hm].sum():
            bench['human_labelled_only'] = dict(n=int(hm.sum()), n_not_found=int(y[hm].sum()), baseline_exact_field=prf((~base[hm]).astype(int), y[hm]),
                                                ai_prior_rules=prf((y_df.p_prior.values[hm] >= 0.5).astype(int), y[hm]))
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import LeaveOneOut, cross_val_predict
            Xl = y_df[FEATS].values
            if 0 < y.sum() < len(y):
                clf = LogisticRegression(C=0.5, max_iter=1000, class_weight='balanced')
                p_loo = cross_val_predict(clf, Xl, y, cv=LeaveOneOut(), method='predict_proba')[:, 1]
                bench['ai_logreg_loo'] = prf((p_loo >= 0.5).astype(int), y)
                clf.fit(Xl, y)
                bench['logreg_coefficients'] = {f: round(c, 3) for f, c in zip(FEATS, clf.coef_[0])}
                F['p_model'] = np.where(F.in_scope, clf.predict_proba(F[FEATS].values)[:, 1], 0.0)
                # shrink toward the prior when labels are few (n_eval < 150): documented blend
                w = min(1.0, len(y) / 150)
                F['p_unreg'] = (w * F.p_model + (1 - w) * F.p_prior).round(4)
                model_used = f'logreg (LOO-validated) blended with prior, weight {w:.2f}'
        except Exception as e:
            bench['logreg_error'] = str(e)
if 'p_unreg' not in F:
    F['p_unreg'] = F.p_prior
bench['model_used'] = model_used

# ------------------------------------------------------------------ zone Monte Carlo: location jitter x Bernoulli
S = L[F.in_scope.values].to_crs(25830)
Fs = F[F.in_scope].set_index('listing_id').loc[S.listing_id]
Z25 = Z.to_crs(25830)
x0, y0 = S.geometry.x.values, S.geometry.y.values
p = Fs.p_unreg.values
n, R = len(S), args.runs
ang = rng.uniform(0, 2 * np.pi, (R, n)); rad = JITTER_M * np.sqrt(rng.uniform(0, 1, (R, n)))
px, py = (x0 + rad * np.cos(ang)).ravel(), (y0 + rad * np.sin(ang)).ravel()
pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(px, py), crs=25830)
j = gpd.sjoin(pts, Z25[['zone_code', 'geometry']], how='left', predicate='within')
j = j[~j.index.duplicated()]
zc = j.zone_code.reindex(range(len(pts))).values.reshape(R, n)
orig = np.tile(Fs.zone_code.values, (R, 1))
zc = np.where(np.isnan(zc.astype(float)), orig, zc).astype(int)       # jitter into the sea / outside -> keep original barrio
draw = rng.uniform(0, 1, (R, n)) < p
codes = Z.zone_code.values
est = np.zeros((R, len(codes))); cnt = np.zeros((R, len(codes)))
for k, c in enumerate(codes):
    m = zc == c
    est[:, k] = (m & draw).sum(1); cnt[:, k] = m.sum(1)
ranks = (-est).argsort(1).argsort(1) + 1
ze = pd.DataFrame({'zone_code': codes, 'zone_name': Z.zone_name.values,
                   'listings_in_scope_mean': cnt.mean(0).round(1),
                   'expected_unregistered': est.mean(0).round(1),
                   'unreg_p10': np.percentile(est, 10, axis=0), 'unreg_p90': np.percentile(est, 90, axis=0),
                   'rank_median': np.median(ranks, 0), 'rank_p10': np.percentile(ranks, 10, axis=0), 'rank_p90': np.percentile(ranks, 90, axis=0),
                   'p_top3': (ranks <= 3).mean(0).round(3), 'p_top5': (ranks <= 5).mean(0).round(3)})
ze['share_unregistered'] = (ze.expected_unregistered / ze.listings_in_scope_mean.replace(0, np.nan)).round(3)
ze = ze.merge(Z[['zone_code', 'n_postal_addresses']], on='zone_code')
ze['unreg_per_100_addresses'] = (100 * ze.expected_unregistered / ze.n_postal_addresses.replace(0, np.nan)).round(2)
# reasons per zone (counts on original assignment, in-scope listings)
Fz = F[F.in_scope]
rs = Fz.groupby('zone_code').agg(no_code=('no_code', 'sum'), code_not_in_register=('code_not_in_register', 'sum'), seasonal_claim=('seasonal_claim', 'sum'),
                                 code_shared=('code_shared', 'sum'), host_5plus_unmatched=('host_5plus', lambda s: int((s & (Fz.loc[s.index, 'code_in_register'] == 0)).sum())),
                                 matched=('code_in_register', 'sum'), border_risk=('border_risk', 'sum'))
ze = ze.merge(rs.reset_index(), on='zone_code', how='left').fillna(0)
base_cnt = L.groupby('zone_code').size()
ze['naive_listings_all'] = ze.zone_code.map(base_cnt).fillna(0).astype(int)
ze['rank_naive'] = ze.naive_listings_all.rank(ascending=False, method='min').astype(int)
ze['rank_ai'] = ze.expected_unregistered.rank(ascending=False, method='min').astype(int)
ze['listings_in_scope'] = ze.zone_code.map(F[F.in_scope].groupby('zone_code').size()).fillna(0).astype(int)
ze['publishable'] = ze.listings_in_scope >= 5   # small-cell rule: hide barrios with < 5 in-scope listings
ze['hit_rate'] = (ze.expected_unregistered / ze.listings_in_scope_mean.replace(0, np.nan)).round(3)
ze['rank_hit_rate'] = ze.hit_rate.where(ze.publishable).rank(ascending=False, method='min')
ze = ze.sort_values('rank_ai')

# ------------------------------------------------------------------ zone benchmark (needs labels): labelled not_found listings found in top-k barrios
if labels is not None:
    nf = labels[labels.label == 'not_found'].merge(F[['listing_id', 'zone_code']], on='listing_id')
    zb = {}
    for k in (1, 2, 3, 5):
        ai_top = set(ze.nsmallest(k, 'rank_ai').zone_code); nv_top = set(ze.sort_values('rank_naive').head(k).zone_code)
        zb[f'top{k}'] = dict(ai_found=int(nf.zone_code.isin(ai_top).sum()), naive_found=int(nf.zone_code.isin(nv_top).sum()),
                            ai_listings_to_check=int(ze[ze.zone_code.isin(ai_top)].listings_in_scope_mean.sum()),
                            naive_listings_to_check=int(ze[ze.zone_code.isin(nv_top)].listings_in_scope_mean.sum()))
    bench['zone_topk_vs_naive'] = zb
# stability: 3 weightings of the seasonal-rental uncertainty
stab = {}
for name, pc in [('seasonal_low_0.1', 0.1), ('seasonal_mid_0.4', 0.4), ('seasonal_high_0.7', 0.7)]:
    pp = np.where(Fs.seasonal_claim.values == 1, pc, p)
    e = pd.Series([(Fs.zone_code.values == c).astype(float) @ pp for c in codes], index=Z.zone_name.values)
    stab[name] = list(e.sort_values(ascending=False).head(5).index)
bench['stability_top5'] = stab
bench['stable_top3_all_weightings'] = len({tuple(v[:3]) for v in stab.values()}) == 1

# ------------------------------------------------------------------ write
scores = F[['listing_id', 'in_scope', 'zone_code', 'p_prior', 'p_unreg']]
for name, df in dict(listing_features=F, listing_scores=scores, zone_estimate=pd.DataFrame(ze)).items():
    con.execute(f'DROP TABLE IF EXISTS "{name}"'); con.execute('DELETE FROM gpkg_contents WHERE table_name = ?', (name,))
    df.to_sql(name, con, index=False)
    con.execute("INSERT INTO gpkg_contents(table_name,data_type,identifier,last_change) VALUES (?, 'attributes', ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))", (name, name))
con.commit(); con.close()
shutil.copyfile(DB, DB_FINAL)
res = dict(model_used=model_used, runs=R, jitter_m=JITTER_M, in_scope_listings=int(F.in_scope.sum()),
           expected_unregistered_total=round(float(est.sum(1).mean()), 1),
           total_p10_p90=[float(np.percentile(est.sum(1), 10)), float(np.percentile(est.sum(1), 90))],
           benchmark=bench,
           zones=ze[['zone_name', 'expected_unregistered', 'unreg_p10', 'unreg_p90', 'rank_ai', 'rank_p10', 'rank_p90', 'rank_naive', 'p_top3', 'publishable']].to_dict('records'))
json.dump(res, open(f'{OUT}/model_results.json', 'w'), indent=1, default=str)
print(json.dumps({k: v for k, v in res.items() if k != 'zones'}, indent=1, default=str))
print(ze[['zone_name', 'listings_in_scope_mean', 'expected_unregistered', 'unreg_p10', 'unreg_p90', 'rank_ai', 'rank_p10', 'rank_p90', 'rank_naive', 'p_top3', 'publishable']].to_string(index=False))
