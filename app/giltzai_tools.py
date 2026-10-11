"""GiltzAI agent tools. Pure functions over the de-identified data bundle in ./data.
Rules built in (flAIre habits): every figure comes from data and carries its source; zone level only;
barrios with < 5 in-scope listings are suppressed; nothing here can return an individual flat, address or host."""
import json, math
from pathlib import Path
import pandas as pd

DATA = Path(__file__).parent / 'data'
MIN_CELL = 5
_cache = {}


def _load():
    if not _cache:
        _cache['ze'] = pd.read_csv(DATA / 'zone_estimate.csv')
        _cache['flags'] = pd.read_csv(DATA / 'listing_flags.csv')
        _cache['summary'] = json.load(open(DATA / 'summary.json', encoding='utf-8'))
        _cache['model'] = json.load(open(DATA / 'model_results.json', encoding='utf-8'))
        _cache['sources'] = pd.read_csv(DATA / 'sources.csv')
    return _cache


# Problem evidence from documents (context, NOT model output). Source cards E1/E2 in out/G1_validation_report.md.
PROBLEM_CONTEXT = {
    'ministry_flagged_illegal_donostia': 138, 'ministry_flagged_illegal_basque_country': 472,
    'inspection_unit_people_before': 4, 'inspection_unit_people_now': 1,
    'src_motion': 'Donostia city council motion on illegal tourist flats, 24-Oct-2025 (public record, citing the Spanish Ministry)',
    'closed_2018_2022': 530, 'fine_min_eur': 600, 'fine_max_eur': 50000,
    'src_press': 'Ondavasca, 5-Oct-2022, citing the city council',
}

SRC_MODEL = 'GiltzAI model on Inside Airbnb 25-Sep-2026 + REATE 06-Oct-2026 + Donostia city census 04-Oct-2026 (Monte Carlo, 1000 runs, 150 m location jitter)'
SRC_DATA = 'Inside Airbnb Euskadi snapshot 25-Sep-2026; REATE (Gobierno Vasco) 06-Oct-2026; official Donostia barrios'


def _r(x, d=1):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), d)


def _norm(name):
    k = str(name or '').strip().upper()
    for a, b in (('Á', 'A'), ('É', 'E'), ('Í', 'I'), ('Ó', 'O'), ('Ú', 'U'), ('Ü', 'U'), ('Ñ', 'N')):
        k = k.replace(a, b)
    k = ' '.join(k.replace('-', ' ').replace('_', ' ').replace('/', ' ').replace('(', ' ').replace(')', ' ').replace('"', ' ').replace("'", ' ').split())
    for pre in ('BARRIO DE ', 'BARRIO ', 'AUZOA ', 'EL ', 'LA ', 'THE '):   # "el Antiguo", "barrio de Gros"
        if k.startswith(pre):
            k = k[len(pre):]
    for suf in (' AUZOA', ' AUZOAN', ' BARRIO', ' DISTRICT', ' NEIGHBOURHOOD', ' NEIGHBORHOOD'):
        if k.endswith(suf):
            k = k[: -len(suf)]
    return k.strip()


ALIASES = {'CENTRO': 'ERDIALDEA', 'CENTRE': 'ERDIALDEA', 'CENTER': 'ERDIALDEA', 'PARTE VIEJA': 'ERDIALDEA', 'ALDE ZAHARRA': 'ERDIALDEA',
           'CASCO VIEJO': 'ERDIALDEA', 'OLD TOWN': 'ERDIALDEA', 'AREA ROMANTICA': 'ERDIALDEA', 'ERDIGUNEA': 'ERDIALDEA',
           'AMARA': 'AMARABERRI', 'AMARA BERRI': 'AMARABERRI', 'ANTIGUO': 'ANTIGUA', 'ANTIGUOA': 'ANTIGUA',
           'ULIA': 'ATEGORRIETA ULIA', 'ATEGORRIETA': 'ATEGORRIETA ULIA', 'MIRACRUZ': 'MIRAKRUZ BIDEBIETA', 'MIRAKRUZ': 'MIRAKRUZ BIDEBIETA',
           'BIDEBIETA': 'MIRAKRUZ BIDEBIETA', 'MIRAMON': 'MIRAMON ZORROAGA', 'ZORROAGA': 'MIRAMON ZORROAGA', 'LOYOLA': 'LOIOLA',
           'ALZA': 'ALTZA', 'EGUIA': 'EGIA', 'AYETE': 'AIETE', 'IGUELDO': 'IGELDO', 'GROSS': 'GROS', 'OARAIN': 'OARAIN'}


def _zone(name):
    """Map a user-typed barrio name (aliases, accents, small typos) to one row of zone_estimate, or None."""
    import difflib
    ze = _load()['ze']
    names = {_norm(z): i for i, z in enumerate(ze.zone_name)}
    key = _norm(name)
    if not key:
        return None
    key = ALIASES.get(key, key)
    if key in names:
        return ze.iloc[names[key]]
    parts = [n for n in names if len(key) >= 4 and (key in n.split() or n in key.split())]   # whole-word match, e.g. "ULIA"
    if len(parts) == 1:
        return ze.iloc[names[parts[0]]]
    close = difflib.get_close_matches(key, list(names) + list(ALIASES), n=1, cutoff=0.8)     # small typos only
    if close:
        k = ALIASES.get(close[0], close[0])
        return ze.iloc[names[k]] if k in names else None
    return None


def _suggest(name):
    import difflib
    ze = _load()['ze']
    pool = {_norm(z): z for z in ze.zone_name}
    return [pool[m] for m in difflib.get_close_matches(_norm(name), list(pool), n=3, cutoff=0.7)]


def _int(x, default, lo, hi):
    """Parse and clamp an integer tool input. Returns (value, adjusted_flag)."""
    try:
        v = int(round(float(x)))
    except (TypeError, ValueError):
        return default, True
    c = max(lo, min(hi, v))
    return c, c != v


def rank_zones(metric: str = 'expected', top_k: int = 5) -> dict:
    """Rank barrios. metric: 'expected' (expected unregistered units), 'hit_rate' (share of listings likely unregistered = efficiency per visit),
    'naive' (baseline: count of all listings)."""
    ze = _load()['ze']
    pub = ze[ze.publishable.astype(bool)].copy()
    col = {'expected': 'expected_unregistered', 'hit_rate': 'hit_rate', 'naive': 'naive_listings_all'}.get(metric)
    if col is None:
        return {'error': "metric must be 'expected' (volume), 'hit_rate' (efficiency per check) or 'naive' (listing count)"}
    top_k, _ = _int(top_k, 5, 1, len(pub))
    pub = pub.sort_values(col, ascending=False).head(top_k)
    rows = [dict(rank=i + 1, barrio=r.zone_name, expected_unregistered=_r(r.expected_unregistered), range_p10_p90=[_r(r.unreg_p10, 0), _r(r.unreg_p90, 0)],
                 hit_rate=_r(r.hit_rate, 3), listings_in_scope=int(r.listings_in_scope), naive_listing_count=int(r.naive_listings_all),
                 naive_rank=int(r.rank_naive), ai_rank_range_p10_p90=[_r(r.rank_p10, 0), _r(r.rank_p90, 0)], prob_in_top3=_r(r.p_top3, 3))
            for i, (_, r) in enumerate(pub.iterrows())]
    hidden = ze[~ze.publishable.astype(bool)].zone_name.tolist()
    return {'metric': metric, 'ranking': rows, 'suppressed_small_barrios': hidden,
            'note': 'Zone-level estimates. Ranges = 10th-90th percentile over 1000 simulations (location blur + model uncertainty). Not evidence against any flat.',
            'source': SRC_MODEL}


def explain_zone(barrio: str) -> dict:
    """Reasons behind a barrio's estimate: counts of listings by licence situation (suppressed below 5)."""
    z = _zone(barrio)
    if z is None:
        ze = _load()['ze']
        return {'error': f'Unknown barrio "{str(barrio)[:60]}". It is not one of the 19 official Donostia barrios.',
                'did_you_mean': _suggest(barrio),
                'known_barrios': ze[ze.publishable.astype(bool)].zone_name.tolist(),
                'suppressed_small_barrios': ze[~ze.publishable.astype(bool)].zone_name.tolist(),
                'aliases': 'Centro / Parte Vieja / Alde Zaharra = ERDIALDEA; Amara = AMARABERRI; Antiguo = ANTIGUA; Ulia = ATEGORRIETA - ULIA'}
    if not bool(z.publishable):
        return {'barrio': z.zone_name, 'suppressed': True, 'reason': f'fewer than {MIN_CELL} listings in scope: details hidden to protect privacy'}
    f = _load()['flags']; s = f[(f.zone_name == z.zone_name) & f.in_scope.astype(bool)]
    sup = lambda n: int(n) if n >= MIN_CELL or n == 0 else f'<{MIN_CELL}'
    unmatched = s[s.code_in_register == 0]
    return {'barrio': z.zone_name, 'expected_unregistered': _r(z.expected_unregistered), 'range_p10_p90': [_r(z.unreg_p10, 0), _r(z.unreg_p90, 0)],
            'hit_rate': _r(z.hit_rate, 3), 'ai_rank': int(z.rank_ai), 'naive_rank': int(z.rank_naive),
            'listings_in_scope': int(len(s)),
            'reasons': {'no_licence_code_anywhere': sup(s.no_code.sum()), 'code_not_found_in_register': sup(s.code_not_in_register.sum()),
                        'claims_seasonal_rental_exemption': sup(s.seasonal_claim.sum()), 'code_matches_register': sup(s.code_in_register.sum()),
                        'midterm_claim_but_allows_short_stays': sup(s.seasonal_but_short_stays.sum()) if 'seasonal_but_short_stays' in s else 'n/a',
                        'midterm_rental_claims_read_in_description': sup(s.llm_midterm_claim.sum()) if 'llm_midterm_claim' in s else 'n/a',
                        'unmatched_new_listings_under_1_year': sup(((unmatched.new_listing == 1) if 'new_listing' in unmatched else unmatched.no_code * 0).sum()),
                        'register_code_reused_by_several_listings': sup(s.code_shared.sum()),
                        'unmatched_listings_of_hosts_with_5plus_listings': sup((unmatched.host_5plus == 1).sum()),
                        'unmatched_listings_of_mid_size_hosts_4_to_10': sup((unmatched.host_mid_4_10 == 1).sum()) if 'host_mid_4_10' in unmatched else 'n/a',
                        'unmatched_with_fully_blocked_calendar': sup((unmatched.calendar_blocked == 1).sum())},
            'location_uncertainty': f"{sup(s.border_risk.sum())} of {len(s)} listings lie < 150 m from the barrio border (Inside Airbnb blurs locations by up to 150 m); included in the range.",
            'source': SRC_DATA}


def compare_baseline() -> dict:
    """AI vs simple baselines: listing matching (exact licence field) and zone ranking (count listings per barrio)."""
    d = _load(); s, b = d['summary'], d['model']['benchmark']
    out = {'listing_matching': {
        'naive_exact_field_matches_entire_homes': s['entire_homes_naive_exact_match'],
        'giltzai_normalised_matches_entire_homes': s['entire_homes_matched_register'],
        'registered_flats_naive_rule_would_wrongly_flag': s['entire_homes_matched_register'] - s['entire_homes_naive_exact_match'],
        'meaning': ('843 and 914 are entire homes RECOGNISED AS REGISTERED (their licence code is in REATE), not flagged ones. The 71 difference = '
                    'entire homes showing a register code in a non-exact format (spaces, dashes, missing zero, code in the text) that an exact rule would '
                    'wrongly treat as unregistered.'),
        'source': SRC_DATA}}
    if 'baseline_exact_field' in b:
        out['hand_label_benchmark'] = {k: b[k] for k in ('labelled_rows', 'rows_by_who', 'label_corrections', 'self_agreement', 'labels', 'n_eval', 'n_not_found',
                                                         'baseline_exact_field', 'ai_prior_rules', 'ai_logreg_loo', 'human_labelled_only') if k in b}
        bl, ai = b['baseline_exact_field'], b.get('ai_logreg_loo', b['ai_prior_rules'])
        out['plain_summary'] = (f"On {b['n_eval']} labelled listings ({b['n_not_found']} not in register): the naive exact-code rule raises {bl['fp']} false alarms "
                                f"(registered flats it would send inspectors to) and misses {bl['fn']}; GiltzAI raises {ai['fp']} false alarms and misses {ai['fn']} "
                                f"(F1 {bl['f1']} vs {ai['f1']}). Small sample: indicative, not definitive.")
        coefs = b.get('logreg_coefficients', {})
        out['strongest_signals'] = dict(sorted(coefs.items(), key=lambda kv: -abs(kv[1]))[:6])
        out['hand_label_benchmark']['note'] = 'Positive class = "not found in register". Precision/recall on hand-labelled listings; unclear labels excluded.'
        ho = b.get('human_labelled_only', {})
        if ho:
            out['labelling_caveat'] = (f"{b.get('rows_by_who', {}).get('auto', '?')} of {b.get('labelled_rows')} labels were filled by a strict rule (same code within 300 m and "
                                       f"capacity +/-2), and most 'registered' labels are auto-filled. On the {ho.get('n')} human-labelled rows only "
                                       f"({ho.get('n') - ho.get('n_not_found', 0)} registered), the exact rule scores F1 {ho['baseline_exact_field']['f1']} and GiltzAI rules "
                                       f"F1 {ho['ai_prior_rules']['f1']}: the 0-vs-3 false-alarm result rests on rule-labelled rows. Needs more human labels.")
    else:
        out['hand_label_benchmark'] = 'pending: hand labels not loaded yet'
    if 'zone_topk_vs_naive' in b:
        out['zone_ranking_topk'] = b['zone_topk_vs_naive']
    out['stability'] = {'top5_under_3_assumptions_for_seasonal_rentals': b.get('stability_top5'), 'top3_identical': b.get('stable_top3_all_weightings')}
    ze = d['ze']; pub = ze[ze.publishable.astype(bool)]
    moved = pub[(pub.rank_ai - pub.rank_naive).abs() >= 2][['zone_name', 'rank_naive', 'rank_ai']]
    hr = pub.sort_values('hit_rate', ascending=False).head(3)[['zone_name', 'hit_rate', 'rank_naive']]
    out['zone_ranking_differences'] = {'barrios_moving_2plus_places_vs_naive': moved.to_dict('records'),
                                       'top3_by_hit_rate': hr.to_dict('records')}
    sh = _simple_share().reindex(pub.zone_name.values)
    rho = pd.Series(pub.hit_rate.values).rank().corr(pd.Series(sh.values).rank())
    out['zone_efficiency_vs_simple_share'] = {
        'spearman_rank_correlation': _r(rho, 2),
        'top3_giltzai_hit_rate': pub.sort_values('hit_rate', ascending=False).zone_name.head(3).tolist(),
        'top3_simple_share_no_ai': sh.sort_values(ascending=False).index[:3].tolist(),
        'meaning': ('Ordering barrios by the plain share of listings failing an exact licence match (no AI) gives almost the same efficiency order as '
                    'GiltzAI. At barrio level the AI adds calibrated counts with ranges and the exclusion of hotels and declared mid-term rentals, '
                    'not a different order. The measurable AI gain is at listing level.')}
    out['model_used'] = b.get('model_used')
    return out


def _simple_share():
    """No-AI baseline per barrio: share of in-scope listings whose licence field does NOT exactly match a register code."""
    f = _load()['flags']; s = f[f.in_scope.astype(bool)]
    return 1 - s.groupby('zone_name').baseline_exact_field.mean()


def inspection_plan(inspector_days: int = 5, checks_per_day: int = 8, metric: str = 'hit_rate') -> dict:
    """Allocate inspector checks to barrios to maximise expected unregistered units found; compared with a naive plan
    that follows listing counts. A 'check' = reviewing one listing in that barrio."""
    try:
        d_raw, c_raw = float(inspector_days), float(checks_per_day)
    except (TypeError, ValueError):
        return {'error': 'inspector_days and checks_per_day must be numbers (e.g. 5 days, 8 checks per day).'}
    if d_raw < 1 or c_raw < 1 or d_raw != d_raw or c_raw != c_raw:
        return {'error': 'inspector_days and checks_per_day must be at least 1. Ask the user for positive values (default 8 checks per day).'}
    if metric not in ('hit_rate', 'expected'):
        return {'error': "metric must be 'hit_rate' (efficiency per check) or 'expected' (volume)"}
    inspector_days, adj_d = _int(d_raw, 5, 1, 60)
    checks_per_day, adj_c = _int(c_raw, 8, 1, 30)
    ze = _load()['ze']; pub = ze[ze.publishable.astype(bool)].copy()
    total = max(1, int(inspector_days) * int(checks_per_day))
    pub['N'] = pub.listings_in_scope.astype(int); pub['h'] = pub.hit_rate.fillna(0)
    _fr = _rates().set_index('barrio')
    n_logged = int(_fr.checks_logged.sum())
    if n_logged:   # field results update the plan: Bayesian hit rates + listings already checked removed
        pub['h'] = pub.zone_name.map(_fr.updated_hit_rate).fillna(pub.h)
        pub['N'] = pub.zone_name.map(_fr.listings_left).fillna(pub.N).astype(int)

    def run(order):
        left, plan, found = total, [], 0.0
        for _, r in order.iterrows():
            if left <= 0:
                break
            n = int(min(left, r.N))
            if n <= 0:
                continue
            plan.append(dict(barrio=r.zone_name, checks=n, expected_found=_r(n * r.h)))
            found += n * r.h; left -= n
        return plan, found
    ai_plan, ai_found = run(pub.sort_values('h', ascending=False) if metric == 'hit_rate' else pub.sort_values('expected_unregistered', ascending=False))
    nv_plan, nv_found = run(pub.sort_values('naive_listings_all', ascending=False))
    pub['share'] = pub.zone_name.map(_simple_share()).fillna(0)
    _, sh_found = run(pub.sort_values('share', ascending=False))
    gain = (ai_found / nv_found - 1) * 100 if nv_found else None
    out = {'checks_available': total, 'giltzai_plan': ai_plan, 'giltzai_expected_found': _r(ai_found), 'naive_plan_by_listing_count': nv_plan,
            'naive_expected_found': _r(nv_found), 'gain_percent': _r(gain),
            'simple_share_rule_expected_found': _r(sh_found),
            'note': ('Expected values from the GiltzAI model (the same model scores every plan). The naive plan goes where most listings are. '
                     'A simple no-AI rule that orders barrios by the share of listings failing an exact licence match is scored too '
                     '(simple_share_rule_expected_found): at barrio level it reaches about the same result as the GiltzAI efficiency plan, so the plan gain '
                     'comes from planning by efficiency, not from the AI. The AI gain is at listing level (codes read in any format, hotel and mid-term '
                     'declarations read, fewer false alarms). Every case still requires an official inspection.'), 'source': SRC_MODEL}
    if adj_d or adj_c:
        out['input_adjusted'] = (f'Request ({d_raw:g} days x {c_raw:g} checks/day) was capped to the planning limits of 1-60 days and 1-30 checks/day: '
                                 f'planned {inspector_days} days x {checks_per_day} checks. Plan longer periods in blocks.')
    out['inputs_used'] = {'inspector_days': inspector_days, 'checks_per_day': checks_per_day, 'metric': metric}
    out['field_log_checks_used'] = n_logged
    return out


def what_if(seasonal_claim_probability: float = None, exclude_blocked_calendars: bool = False, entire_homes_only: bool = False, top_k: int = 5) -> dict:
    """Re-rank barrios under different assumptions (expected values; ranges only in the full model)."""
    d = _load(); f = d['flags'][d['flags'].in_scope.astype(bool)].copy()
    p = f.p_unreg.copy()
    top_k, _ = _int(top_k, 5, 1, 17)
    if seasonal_claim_probability is not None:
        try:
            v = float(seasonal_claim_probability)
        except (TypeError, ValueError):
            return {'error': 'seasonal_claim_probability must be a number between 0 and 1 (e.g. 0.5 = 50%).'}
        if v != v:
            return {'error': 'seasonal_claim_probability must be a number between 0 and 1 (e.g. 0.5 = 50%).'}
        if 1 < v <= 100:      # "50" meant as a percentage
            v = v / 100
        seasonal_claim_probability = max(0.0, min(1.0, v))
        p[f.seasonal_claim == 1] = seasonal_claim_probability
    if exclude_blocked_calendars:
        p[(f.calendar_blocked == 1) & (f.active_ltm == 0)] = 0.0
    if entire_homes_only:
        p[f.entire_home == 0] = 0.0
    f['p'] = p
    g = f.groupby('zone_name').agg(expected=('p', 'sum'), n=('p', 'size'))
    g = g[g.n >= MIN_CELL].sort_values('expected', ascending=False)
    base = f.groupby('zone_name').p_unreg.sum().reindex(g.index).rank(ascending=False, method='min')
    rows = [dict(rank=i + 1, barrio=z, expected_unregistered=_r(r.expected), base_rank=int(base[z])) for i, (z, r) in enumerate(g.head(int(top_k)).iterrows())]
    return {'assumptions': dict(seasonal_claim_probability=seasonal_claim_probability, exclude_blocked_calendars=exclude_blocked_calendars,
                                entire_homes_only=entire_homes_only),
            'total_expected_unregistered': _r(g.expected.sum()), 'ranking': rows, 'source': SRC_MODEL}


FIGURES = {
    'listings_total': lambda f: len(f),
    'entire_homes': lambda f: int((f.room_type == 'Entire home/apt').sum()),
    'in_scope_listings': lambda f: int(f.in_scope.astype(bool).sum()),
    'entire_homes_matched_register': lambda f: int(((f.room_type == 'Entire home/apt') & (f.exact_match_status == 'A_matched_register')).sum()),
    'entire_homes_no_valid_code': lambda f: int(((f.room_type == 'Entire home/apt') & f.exact_match_status.isin(['D_code_not_in_register', 'E_no_code'])).sum()),
    'entire_homes_seasonal_claim': lambda f: int(((f.room_type == 'Entire home/apt') & (f.exact_match_status == 'C_claims_seasonal_rental')).sum()),
    'entire_homes_naive_exact_match': lambda f: int(((f.room_type == 'Entire home/apt') & f.baseline_exact_field.astype(bool)).sum()),
    # 413 = listings of ALL room types whose code is not in the register (no code + code not in REATE + seasonal claim); Claude read these descriptions
    'ai_text_read_listings': lambda f: int(f.exact_match_status.isin(['C_claims_seasonal_rental', 'D_code_not_in_register', 'E_no_code']).sum()),
    'ai_text_hotel_like_excluded': lambda f: int(f.llm_hotel_like.sum()),
    'ai_text_midterm_claims': lambda f: int(f.llm_midterm_claim.sum()),
}


def check_figure(name: str) -> dict:
    """Recompute a headline figure directly from the listing data and compare it with the published summary."""
    d = _load()
    if name not in FIGURES:
        return {'error': f'unknown figure. Only these figures can be verified: {list(FIGURES)}. Anything else is not in the data.'}
    v = FIGURES[name](d['flags'])
    pub = d['summary'].get(name)
    out = {'figure': name, 'recomputed': v, 'published': pub, 'match': v == pub, 'source': SRC_DATA}
    if name == 'ai_text_read_listings':
        out['meaning'] = ('All room types whose licence code is not in the register: no code + code not in REATE + seasonal-rental claim. '
                          'Different from entire_homes_no_valid_code (196 = entire homes with no code or a code not in REATE).')
    return out


def data_sources() -> dict:
    """Where the data comes from, licences, dates and known limits."""
    s = _load()['sources']
    return {'sources': s[['source_id', 'name', 'owner', 'licence', 'snapshot_date', 'reliability']].to_dict('records'),
            'evidence_from_literature': ['Boto-Garcia (2025), Tourism Economics 31(4): matching Airbnb listings to official licences in Menorca (3,800 listings), illegality peaks among mid-size hosts (4-5 listings). In Donostia the unmatched share also peaks among hosts with 6-10 listings (47%) vs 18% for single-listing hosts and 16% for hosts with >20.'],
            'ai_text_reading': ('Claude (claude-haiku-5-5) read the descriptions of the 413 listings (all room types) with no licence code found in the register: '
                            '286 with no code, 34 with a code not in REATE and 93 claiming the seasonal-rental exemption. (196 is a different figure: ENTIRE HOMES '
                            'with no code or a code not in REATE.) It extracted only what hosts declare: mid-term / non-tourist rental claims (66), '
                            'hotel / guesthouse / residence (41, excluded as out of scope), licence mentions. It never decides legality.'),
            'problem_context': [f"{PROBLEM_CONTEXT['src_motion']}: the Ministry flagged {PROBLEM_CONTEXT['ministry_flagged_illegal_donostia']} illegal tourist homes in Donostia "
                                f"({PROBLEM_CONTEXT['ministry_flagged_illegal_basque_country']} in the Basque Country), and the municipal inspection unit went from "
                                f"{PROBLEM_CONTEXT['inspection_unit_people_before']} people to {PROBLEM_CONTEXT['inspection_unit_people_now']}.",
                                f"{PROBLEM_CONTEXT['src_press']}: {PROBLEM_CONTEXT['closed_2018_2022']} illegal tourist flats closed in 2018-2022; fines from "
                                f"{PROBLEM_CONTEXT['fine_min_eur']} to {PROBLEM_CONTEXT['fine_max_eur']:,} euros.",
                                'These are context figures from documents, not outputs of the GiltzAI model.'],
            'operations_and_scale': ['Refresh: re-run the pipeline on each new Inside Airbnb snapshot and the daily REATE file (open data).',
                                     'REATE and the Inside Airbnb Euskadi snapshot already cover all Basque municipalities; another city needs its barrio boundaries and hand labels.',
                                     'Dependency: Inside Airbnb is a third-party scrape. Official path: EU Regulation 2024/1028 (applies from 20 May 2026) makes platforms share listing data with authorities; the Spanish Supreme Court (STS 620/2026, May 2026) annulled the national Registro Unico articles but kept the digital single window and platform data duties, so the regional REATE register stays the reference.'],
            'signals_not_used': ['Guest ratings: identical for matched and unmatched listings (median 4.8), so not informative.',
                                 'Cross-listing on Booking/Vrbo: no public data.'],
            'limits': ['Only Airbnb listings (Inside Airbnb scrape); Booking/Vrbo not included -> undercount.',
                       'Listing locations are blurred by up to 150 m; handled with simulation ranges.',
                       '"No licence code" is not proof of illegality (mid-term rentals, inactive listings, errors). Every case needs an official inspection.',
                       'Population/dwellings per barrio unavailable on download day; postal addresses used as a proxy.',
                       'REATE shows no new registrations in 2026 (latest 16-Dec-2025).'],
            'privacy': 'Zone-level only; hosts hashed; barrios with fewer than 5 listings suppressed; no addresses or listing ids exposed.'}



# ============================================================================ FIELD LOOP + ROUTE (11 Oct 2026)
# Inspector results update each barrio's hit rate (Bayesian Beta update) -> plans and routes re-optimise.
# Privacy: the log stores ONLY barrio + result + count + time (no addresses, codes, listing ids or free text).
import os as _os, math as _math, datetime as _dt

import contextvars as _cv
FIELD_LOG = Path(_os.environ.get('GILTZAI_FIELD_LOG', DATA / 'field_log.csv'))
_FIELD_LOG_CTX = _cv.ContextVar('giltzai_field_log', default=None)   # the app sets a per-visitor file in public demo mode


def set_field_log_path(path):
    _FIELD_LOG_CTX.set(Path(path))


def _log_path():
    return _FIELD_LOG_CTX.get() or FIELD_LOG
PRIOR_STRENGTH = 10      # the model's hit rate counts as 10 pseudo-checks; real checks gradually take over
RESULTS = {
    'not_in_register': 'listing checked and NOT found in the official register',
    'registered': 'listing checked and found in the register (legal)',
    'closed_or_inactive': 'listing no longer active / flat closed (not counted in the hit rate)',
    'out_of_scope': 'hotel, hostel or mid-term rental (not counted in the hit rate)',
}
_RESULT_WORDS = {
    'not_in_register': ('not in register', 'not_in_register', 'unregistered', 'not registered', 'illegal', 'sin registro', 'no registrado', 'no registrada',
                        'ilegal', 'erregistratu gabe', 'found', 'hit'),
    'registered': ('registered', 'legal', 'registrado', 'registrada', 'en regla', 'erregistratuta', 'ok'),
    'closed_or_inactive': ('closed', 'inactive', 'cerrado', 'cerrada', 'inactivo', 'vacío', 'vacio', 'itxita'),
    'out_of_scope': ('out of scope', 'out_of_scope', 'hotel', 'hostel', 'temporada', 'mid-term', 'midterm', 'residencia'),
}


def _result(x):
    k = str(x or '').strip().lower().replace('-', ' ').replace('_', ' ')
    if k.replace(' ', '_') in RESULTS:
        return k.replace(' ', '_')
    for res, words in _RESULT_WORDS.items():
        if any(k == w or k.startswith(w) for w in words):
            return res
    return None


def _read_log():
    cols = ['ts', 'barrio', 'result', 'checks', 'source']
    try:
        if _log_path().exists() and _log_path().stat().st_size > 0:
            d = pd.read_csv(_log_path())
            return d[cols] if set(cols) <= set(d.columns) else pd.DataFrame(columns=cols)
    except Exception:
        pass
    return pd.DataFrame(columns=cols)


def _rates():
    """Per publishable barrio: model hit rate, field results, Bayesian-updated hit rate, listings left to check."""
    ze = _load()['ze']; pub = ze[ze.publishable.astype(bool)].copy()
    lg = _read_log()
    lg['checks'] = pd.to_numeric(lg.checks, errors='coerce').fillna(0).astype(int)
    agg = lg.pivot_table(index='barrio', columns='result', values='checks', aggfunc='sum', fill_value=0) if len(lg) else pd.DataFrame()
    g = lambda z, c: int(agg.loc[z, c]) if (len(agg) and z in agg.index and c in agg.columns) else 0
    rows = []
    for _, r in pub.iterrows():
        z = r.zone_name; h0 = float(r.hit_rate if r.hit_rate == r.hit_rate else 0)
        k, f = g(z, 'not_in_register'), g(z, 'registered'); other = g(z, 'closed_or_inactive') + g(z, 'out_of_scope')
        a, b = h0 * PRIOR_STRENGTH + k, (1 - h0) * PRIOR_STRENGTH + f
        h = a / (a + b)
        sd = _math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
        n_left = max(0, int(r.listings_in_scope) - (k + f + other))
        rows.append(dict(barrio=z, listings_in_scope=int(r.listings_in_scope), model_hit_rate=round(h0, 3), checks_logged=k + f + other,
                         not_in_register=k, registered=f, closed_or_out_of_scope=other, updated_hit_rate=round(h, 3),
                         updated_range=[round(max(0, h - 1.28 * sd), 3), round(min(1, h + 1.28 * sd), 3)], listings_left=n_left))
    return pd.DataFrame(rows)


def log_check(barrio: str, result: str, checks: int = 1) -> dict:
    """Record the result of field checks in one barrio (only barrio + result + count are stored; never addresses, codes or flat details).
    result: not_in_register | registered | closed_or_inactive | out_of_scope. Updates the barrio's hit rate; plans and routes use it."""
    z = _zone(barrio)
    if z is None:
        return {'error': f'Unknown barrio "{str(barrio)[:60]}". Nothing was logged.', 'did_you_mean': _suggest(barrio)}
    if not bool(z.publishable):
        return {'error': f'{z.zone_name} has fewer than {MIN_CELL} listings in scope and is hidden for privacy; field results are not tracked there. Nothing was logged.'}
    res = _result(result)
    if res is None:
        return {'error': f'Unknown result "{str(result)[:40]}". Use one of: {list(RESULTS)}. Nothing was logged.'}
    n, adj = _int(checks, 1, 1, 30)
    before = _rates().set_index('barrio').loc[z.zone_name]
    if n > int(before.listings_left):
        return {'error': f'Only {int(before.listings_left)} listings in scope are left to check in {z.zone_name}; {n} checks cannot be logged. Nothing was logged.'}
    _log_path().parent.mkdir(parents=True, exist_ok=True)
    new = not _log_path().exists() or _log_path().stat().st_size == 0
    with open(_log_path(), 'a', encoding='utf-8', newline='') as fh:
        if new:
            fh.write('ts,barrio,result,checks,source\n')
        fh.write(f"{_dt.datetime.now().isoformat(timespec='seconds')},{z.zone_name},{res},{n},tool\n")
    after = _rates().set_index('barrio').loc[z.zone_name]
    out = {'logged': {'barrio': z.zone_name, 'result': res, 'meaning': RESULTS[res], 'checks': n},
           'hit_rate_before': float(before.updated_hit_rate), 'hit_rate_after': float(after.updated_hit_rate),
           'model_hit_rate': float(after.model_hit_rate), 'field_so_far': {'not_in_register': int(after.not_in_register), 'registered': int(after.registered),
                                                                          'closed_or_out_of_scope': int(after.closed_or_out_of_scope)},
           'listings_left_to_check': int(after.listings_left),
           'note': (f'Bayesian update: the model hit rate counts as {PRIOR_STRENGTH} pseudo-checks, each real check moves it. '
                    'Plans and routes now use the updated rates. Only barrio + result + count are stored.')}
    if adj:
        out['input_adjusted'] = 'checks capped to 1-30 per entry.'
    return out


def undo_last_check() -> dict:
    """Remove the most recent field-log entry (e.g. a typing mistake)."""
    lg = _read_log()
    if not len(lg):
        return {'error': 'The field log is empty; nothing to undo.'}
    last = lg.iloc[-1].to_dict()
    lg.iloc[:-1].to_csv(_log_path(), index=False)
    return {'removed': {k: last[k] for k in ('barrio', 'result', 'checks')}, 'entries_left': len(lg) - 1}


def reset_field_log() -> dict:
    """Demo helper (app only, not an agent tool): empty the field log."""
    _log_path().parent.mkdir(parents=True, exist_ok=True)
    _log_path().write_text('ts,barrio,result,checks,source\n', encoding='utf-8')
    return {'reset': True}


def field_summary() -> dict:
    """What inspectors have found so far vs what the model predicted, per barrio, and the updated hit rates used by plans and routes."""
    r = _rates()
    done = r[r.checks_logged > 0]
    k, f = int(done.not_in_register.sum()), int(done.registered.sum())
    pred = float((done.model_hit_rate * (done.not_in_register + done.registered)).sum())
    out = {'checks_logged': int(done.checks_logged.sum()), 'not_in_register': k, 'registered': f,
           'closed_or_out_of_scope': int(done.closed_or_out_of_scope.sum()),
           'field_hit_rate': round(k / (k + f), 3) if k + f else None,
           'model_expected_hits_for_these_checks': round(pred, 1),
           'barrios': done.drop(columns=['listings_in_scope']).to_dict('records'),
           'pilot_check': None, 'source': 'GiltzAI field log (inspector entries) + model hit rates'}
    if k + f >= 10:
        diff = k - pred
        out['pilot_check'] = ('Field results agree with the model so far.' if abs(diff) <= 2 * _math.sqrt(max(pred, 1)) else
                              f'Field results differ from the model ({k} found vs {pred:.1f} expected): treat the ranking with caution and add these checks as new labels before the next plan.')
    elif k + f:
        out['pilot_check'] = f'Too few checks to judge the model yet ({k + f} of the 10 needed).'
    return out


# ---- route
_CENT = {}


def _centroids():
    if not _CENT:
        gj = json.load(open(DATA / 'zones.geojson', encoding='utf-8'))
        for ft in gj['features']:
            gm = ft['geometry']; polys = gm['coordinates'] if gm['type'] == 'MultiPolygon' else [gm['coordinates']]
            best = None
            for poly in polys:
                ring = poly[0]; a = cx = cy = 0.0
                for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
                    c = x0 * y1 - x1 * y0; a += c; cx += (x0 + x1) * c; cy += (y0 + y1) * c
                if abs(a) > 1e-12 and (best is None or abs(a) > best[0]):
                    best = (abs(a), cx / (3 * a), cy / (3 * a))
            _CENT[ft['properties']['zone_name']] = (best[2], best[1])   # (lat, lon)
    return _CENT


def _km(p, q):
    la1, lo1, la2, lo2 = map(_math.radians, (p[0], p[1], q[0], q[1]))
    h = _math.sin((la2 - la1) / 2) ** 2 + _math.cos(la1) * _math.cos(la2) * _math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * _math.asin(_math.sqrt(h))


SPEED_KMH = {'car': 15, 'bike': 12, 'walk': 4.5}   # urban door-to-door incl. parking; detour factor 1.3 on straight lines


def route_plan(start_barrio: str, days: int = 1, hours_per_day: float = 6, minutes_per_check: int = 20, mode: str = 'car',
               max_checks_per_stop: int = 8, check_start_first: bool = False) -> dict:
    """Day-by-day order of barrios for field checks, starting each day from start_barrio: maximises expected finds per minute
    (checks + travel between barrio centres). Uses the field-updated hit rates. Barrio level only (no addresses).
    check_start_first=True forces a first block of checks in the start barrio (e.g. 'I want to start checking in Antiguo').
    Compared with (a) visiting barrios by hit rate ignoring travel and (b) going where most listings are."""
    z = _zone(start_barrio)
    if z is None:
        return {'error': f'Unknown start barrio "{str(start_barrio)[:60]}".', 'did_you_mean': _suggest(start_barrio)}
    if mode not in SPEED_KMH:
        return {'error': f"mode must be one of {list(SPEED_KMH)}"}
    days, adj1 = _int(days, 1, 1, 10)
    mpc, adj2 = _int(minutes_per_check, 20, 5, 120)
    stop_cap, _ = _int(max_checks_per_stop, 8, 1, 30)
    try:
        hpd = max(1.0, min(10.0, float(hours_per_day)))
    except (TypeError, ValueError):
        hpd = 6.0
    C = _centroids(); R = _rates()
    R = R[(R.listings_left > 0) & (R.updated_hit_rate > 0)]
    start = z.zone_name; home = C[start]

    def travel(a, b):
        return 0.0 if a == b else _km(C[a], C[b]) * 1.3 / SPEED_KMH[mode] * 60

    naive_n = dict(zip(_load()['ze'].zone_name, _load()['ze'].naive_listings_all))

    def simulate(how):
        left = dict(zip(R.barrio, R.listings_left)); rate = dict(zip(R.barrio, R.updated_hit_rate))
        plan, tot_f, tot_t, tot_c = [], 0.0, 0.0, 0
        for d in range(1, days + 1):
            cur, t_left, stops = start, hpd * 60, []
            first = check_start_first and how == 'ratio' and left.get(start, 0) > 0
            while True:
                best = None
                for b, n in left.items():
                    if n <= 0 or (first and b != start):
                        continue
                    tr = travel(cur, b); room = int((t_left - tr) // mpc)
                    k = min(n, stop_cap, room)
                    if k < 1:
                        continue
                    val = k * rate[b]
                    score = (val / (tr + k * mpc) if how == 'ratio' else rate[b] if how == 'hit_rate' else naive_n.get(b, 0)) + (1e-9 if b == cur else 0)
                    if best is None or score > best[0]:
                        best = (score, b, k, tr, val)
                first = False
                if best is None:
                    break
                _, b, k, tr, val = best
                if stops and stops[-1]['barrio'] == b:          # same barrio again: extend the stop
                    stops[-1]['checks'] += int(k); stops[-1]['expected_finds'] = round(stops[-1]['expected_finds'] + val, 1)
                else:
                    stops.append(dict(barrio=b, travel_min=round(tr), checks=int(k), expected_finds=round(val, 1), hit_rate=round(rate[b], 3)))
                left[b] -= k; t_left -= tr + k * mpc; cur = b
                tot_f += val; tot_t += tr; tot_c += k
            back = travel(cur, start)
            plan.append(dict(day=d, stops=stops, travel_back_to_start_min=round(back)))
            tot_t += back
        return plan, round(tot_f, 1), round(tot_t), tot_c

    plan, f1, t1, c1 = simulate('ratio')
    _, f0, t0, c0 = simulate('hit_rate')
    _, fn, tn, cn = simulate('naive')
    path = [{'barrio': start, 'lat': home[0], 'lon': home[1]}]
    for dd in plan[:1]:
        for s_ in dd['stops']:
            path.append({'barrio': s_['barrio'], 'lat': C[s_['barrio']][0], 'lon': C[s_['barrio']][1]})
    out = {'start': start, 'mode': mode, 'days': days, 'hours_per_day': hpd, 'minutes_per_check': mpc,
           'route': plan, 'expected_finds': f1, 'checks': c1, 'travel_minutes': t1,
           'comparison_hit_rate_order_ignoring_travel': {'expected_finds': f0, 'checks': c0, 'travel_minutes': t0},
           'comparison_most_listings_first': {'expected_finds': fn, 'checks': cn, 'travel_minutes': tn},
           'day1_path_barrio_centres': path,
           'field_log_used': int(_rates().checks_logged.sum()),
           'note': ('Greedy route: at each step go to the barrio with most expected finds per minute (travel between barrio centres x1.3 detour + '
                    f'{mpc} min per check), in blocks of up to {stop_cap} checks (consecutive blocks in the same barrio are shown as one stop); each day starts from {start}. Barrio level only: inspectors pick listings '
                    'inside a barrio with their usual procedure. Expected finds use field-updated hit rates.'), 'source': SRC_MODEL}
    if adj1 or adj2:
        out['input_adjusted'] = 'days capped to 1-10 and minutes per check to 5-120.'
    return out


TOOLS = {f.__name__: f for f in (rank_zones, explain_zone, compare_baseline, inspection_plan, what_if, check_figure, data_sources, log_check, undo_last_check, field_summary, route_plan)}

TOOL_SCHEMAS = [
    {'name': 'rank_zones', 'description': rank_zones.__doc__, 'input_schema': {'type': 'object', 'properties': {
        'metric': {'type': 'string', 'enum': ['expected', 'hit_rate', 'naive']}, 'top_k': {'type': 'integer', 'minimum': 1, 'maximum': 17}}}},
    {'name': 'explain_zone', 'description': explain_zone.__doc__, 'input_schema': {'type': 'object', 'properties': {
        'barrio': {'type': 'string', 'description': 'Barrio name, e.g. Gros, Erdialdea (Centro), Amaraberri, Ategorrieta - Ulia'}}, 'required': ['barrio']}},
    {'name': 'compare_baseline', 'description': compare_baseline.__doc__, 'input_schema': {'type': 'object', 'properties': {}}},
    {'name': 'inspection_plan', 'description': inspection_plan.__doc__, 'input_schema': {'type': 'object', 'properties': {
        'inspector_days': {'type': 'integer', 'minimum': 1, 'maximum': 60}, 'checks_per_day': {'type': 'integer', 'minimum': 1, 'maximum': 30},
        'metric': {'type': 'string', 'enum': ['hit_rate', 'expected']}}}},
    {'name': 'what_if', 'description': what_if.__doc__, 'input_schema': {'type': 'object', 'properties': {
        'seasonal_claim_probability': {'type': 'number', 'minimum': 0, 'maximum': 1}, 'exclude_blocked_calendars': {'type': 'boolean'},
        'entire_homes_only': {'type': 'boolean'}, 'top_k': {'type': 'integer', 'minimum': 1, 'maximum': 17}}}},
    {'name': 'check_figure', 'description': check_figure.__doc__ + ' Names and their published values (use these to map a number the user mentions to a name): '
        + '; '.join(f"{k} = {_load()['summary'].get(k)}" for k in FIGURES), 'input_schema': {'type': 'object', 'properties': {
        'name': {'type': 'string', 'enum': list(FIGURES)}}, 'required': ['name']}},
    {'name': 'data_sources', 'description': data_sources.__doc__, 'input_schema': {'type': 'object', 'properties': {}}},
    {'name': 'log_check', 'description': log_check.__doc__, 'input_schema': {'type': 'object', 'properties': {
        'barrio': {'type': 'string'}, 'result': {'type': 'string', 'enum': list(RESULTS)}, 'checks': {'type': 'integer', 'minimum': 1, 'maximum': 30}},
        'required': ['barrio', 'result']}},
    {'name': 'undo_last_check', 'description': undo_last_check.__doc__, 'input_schema': {'type': 'object', 'properties': {}}},
    {'name': 'field_summary', 'description': field_summary.__doc__, 'input_schema': {'type': 'object', 'properties': {}}},
    {'name': 'route_plan', 'description': route_plan.__doc__, 'input_schema': {'type': 'object', 'properties': {
        'start_barrio': {'type': 'string'}, 'days': {'type': 'integer', 'minimum': 1, 'maximum': 10}, 'hours_per_day': {'type': 'number', 'minimum': 1, 'maximum': 10},
        'minutes_per_check': {'type': 'integer', 'minimum': 5, 'maximum': 120}, 'mode': {'type': 'string', 'enum': ['car', 'bike', 'walk']},
        'max_checks_per_stop': {'type': 'integer', 'minimum': 1, 'maximum': 30}, 'check_start_first': {'type': 'boolean'}}, 'required': ['start_barrio']}},
]


def call_tool(name, args):
    """Run a tool safely: unknown args are dropped, errors are reported in plain words (never raw exceptions)."""
    import inspect, sys
    fn = TOOLS.get(name)
    if fn is None:
        return {'error': f'unknown tool. Available: {list(TOOLS)}'}
    params = inspect.signature(fn).parameters
    args = {k: v for k, v in (args or {}).items() if k in params} if isinstance(args, dict) else {}
    try:
        return fn(**args)
    except Exception as e:  # report, never invent; details go to the server log only
        print(f'[giltzai_tools] {name} failed: {type(e).__name__}', file=sys.stderr)
        return {'error': 'The tool could not compute this with the given inputs. Do not guess a figure; ask the user to rephrase or use another tool.'}
