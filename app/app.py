"""GiltzAI prototype - run:  streamlit run app.py   (from the GiltzAI/app folder)
Every figure on screen comes from the data bundle via giltzai_tools (import giltzai_tools as T) or data/summary.json."""
import json
import os
import unicodedata
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from html import escape
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

import giltzai_tools as T

# Public demo mode (Streamlit Cloud): set GILTZAI_DEMO_MODE=1 and GILTZAI_CHAT_PASSCODE in the app secrets.
DEMO = os.environ.get('GILTZAI_DEMO_MODE', '') == '1'
if DEMO:   # each visitor gets their own temporary field log, so judges don't see each other's test entries
    import tempfile, uuid
    if 'fl_path' not in st.session_state:
        st.session_state.fl_path = str(Path(tempfile.gettempdir()) / f'giltzai_field_{uuid.uuid4().hex}.csv')
    T.set_field_log_path(st.session_state.fl_path)

st.set_page_config(page_title='GiltzAI · Donostia', page_icon='🔑', layout='wide', initial_sidebar_state='collapsed')
APP = Path(__file__).parent
DATA = APP / 'data'
S = json.load(open(DATA / 'summary.json', encoding='utf-8'))
MR = json.load(open(DATA / 'model_results.json', encoding='utf-8'))

# ---------------------------------------------------------------- helpers
_NEW_ST = tuple(int(x) for x in st.__version__.split('.')[:2]) >= (1, 46)   # 'width' API; older versions use use_container_width


def show_df(df, **kw):
    if not _NEW_ST:
        kw.setdefault('use_container_width', True)
    st.dataframe(df, hide_index=True, **kw)


def html(s):
    """Render HTML; strip indentation so markdown never turns it into a code block."""
    st.markdown(' '.join(line.strip() for line in str(s).splitlines()), unsafe_allow_html=True)


ALIAS = {'ERDIALDEA': 'Centro / Parte Vieja', 'AMARABERRI': 'Amara', 'ANTIGUA': 'Antiguo', 'ALTZA': 'Alza', 'LOIOLA': 'Loyola'}


def nice(z, alias=True):
    t = str(z).title().replace(' - ', '–')
    return f'{t} ({ALIAS[z]})' if alias and z in ALIAS else t


def half_up(x):
    return int(Decimal(str(x)).quantize(Decimal('1'), rounding=ROUND_HALF_UP))


def n0(x):
    return '—' if x is None else f'{half_up(x):,}'   # half-up so 130.5 -> 131 everywhere (tables round the same way)


def n1(x):
    return '—' if x is None else f'{x:,.1f}'


def pct(x):
    return '—' if x is None else f'{x * 100:.0f}%'


def rng(r):
    return f'{r[0]:.0f}–{r[1]:.0f}' if r and None not in r else '—'


def day(s):
    try:
        d = datetime.strptime(str(s)[:10], '%Y-%m-%d')
        return f"{d.day} {d.strftime('%b %Y')}"
    except Exception:
        return str(s)


KEY_SVG = ('<svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">'
           '<circle cx="7.5" cy="15.5" r="4.5"/><path d="M10.7 12.3 21 2"/><path d="m16 7 3 3"/><path d="m19 4 2 2"/></svg>')
LOCK_SVG = ('<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">'
            '<rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/></svg>')

# ---------------------------------------------------------------- style
html("""<style>
:root{--ink:#0F172A;--muted:#526077;--line:#DCE3EC;--card:#FFFFFF;--brand:#0B5D7A;--brand2:#0E7490;--hot:#C2410C;--good:#047857;--soft:#EAF0F5;}
header[data-testid="stHeader"]{display:none;}
[data-testid="stSidebarCollapsedControl"]{display:none;}
.block-container,[data-testid="stMainBlockContainer"]{padding-top:1.1rem !important;padding-bottom:2rem !important;max-width:1760px;}
h1,h2,h3{letter-spacing:-0.01em;color:var(--ink);}
.gz-hero{display:flex;justify-content:space-between;align-items:flex-end;gap:24px;flex-wrap:wrap;margin-bottom:10px;}
.gz-brand{display:flex;align-items:center;gap:10px;color:var(--brand);font-weight:700;font-size:15px;letter-spacing:.04em;text-transform:uppercase;}
.gz-brand .city{color:var(--muted);font-weight:500;text-transform:none;letter-spacing:0;}
.gz-hero h1{font-size:36px;line-height:1.05;margin:2px 0 4px 0;padding:0;font-weight:800;}
.gz-hero h1 span{color:var(--brand);}
.gz-hero p{margin:0;color:var(--muted);font-size:15.5px;max-width:980px;}
.gz-badges{display:flex;flex-direction:column;align-items:flex-end;gap:6px;}
.gz-badge{display:inline-flex;align-items:center;gap:6px;border-radius:999px;padding:5px 12px;font-size:13px;font-weight:600;background:var(--soft);color:var(--ink);border:1px solid var(--line);}
.gz-badge.privacy{background:#ECFDF5;color:var(--good);border-color:#A7F3D0;}
.gz-kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:4px 0 12px 0;}
.gz-kpi{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:10px 16px;box-shadow:0 1px 2px rgba(15,23,42,.04);}
.gz-kpi .lab{font-size:13px;color:var(--muted);font-weight:600;}
.gz-kpi .val{font-size:28px;font-weight:800;color:var(--ink);line-height:1.15;margin-top:2px;font-variant-numeric:tabular-nums;}
.gz-kpi .val small{font-size:15px;font-weight:600;color:var(--muted);margin-left:6px;}
.gz-kpi .sub{font-size:12.5px;color:var(--muted);}
.gz-kpi.accent{background:linear-gradient(135deg,#0B5D7A,#0E7490);border:none;}
.gz-kpi.accent .lab,.gz-kpi.accent .sub,.gz-kpi.accent .val small{color:#CFE8F0;}
.gz-kpi.accent .val{color:#fff;}
.gz-steps{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:0;background:var(--card);border:1px solid var(--line);border-radius:14px;margin-bottom:6px;overflow:hidden;}
.gz-step{padding:8px 16px;display:flex;gap:12px;align-items:center;position:relative;}
.gz-step+.gz-step{border-left:1px solid var(--line);}
.gz-step .num{flex:0 0 28px;height:28px;border-radius:50%;background:var(--brand);color:#fff;font-weight:700;display:flex;align-items:center;justify-content:center;font-size:14px;}
.gz-step .t{font-weight:700;font-size:14.5px;color:var(--ink);}
.gz-step .d{font-size:12.5px;color:var(--muted);line-height:1.3;}
button[role="tab"]{font-size:16px !important;padding:10px 6px !important;}
button[role="tab"] p{font-size:16px !important;font-weight:600 !important;}
.gz-legend{display:flex;flex-wrap:wrap;align-items:center;gap:6px 14px;font-size:12.5px;color:var(--muted);margin:2px 0 6px 0;}
.gz-legend .ttl{font-weight:700;color:var(--ink);}
.gz-sw{display:inline-block;width:22px;height:12px;border-radius:3px;vertical-align:-1px;margin-right:4px;border:1px solid rgba(15,23,42,.15);}
.gz-sw.top{background:transparent;border:3px solid #0F172A;width:18px;height:10px;}
.gz-card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:12px 16px;margin-bottom:10px;}
.gz-rank{display:flex;gap:14px;align-items:center;}
.gz-rank .pos{flex:0 0 40px;height:40px;border-radius:10px;background:var(--hot);color:#fff;font-weight:800;font-size:20px;display:flex;align-items:center;justify-content:center;}
.gz-rank .name{font-weight:700;font-size:17px;color:var(--ink);line-height:1.2;}
.gz-rank .big{font-size:22px;font-weight:800;color:var(--ink);font-variant-numeric:tabular-nums;}
.gz-rank .big small{font-size:13px;font-weight:600;color:var(--muted);}
.gz-rank .meta{font-size:12.5px;color:var(--muted);}
.gz-chip{display:inline-block;border-radius:999px;padding:1px 8px;font-size:12px;font-weight:700;margin-left:4px;}
.gz-chip.up{background:#FFEDD5;color:#9A3412;}
.gz-chip.down{background:#E2E8F0;color:#334155;}
.gz-chip.same{background:#F1F5F9;color:#64748B;}
.gz-headline{background:var(--card);border:1px solid var(--line);border-left:6px solid var(--brand);border-radius:14px;padding:16px 20px;margin:6px 0 14px 0;}
.gz-headline .k{font-size:13px;font-weight:700;color:var(--brand);text-transform:uppercase;letter-spacing:.05em;}
.gz-headline .h{font-size:24px;font-weight:700;color:var(--ink);line-height:1.3;margin-top:2px;}
.gz-headline .h b{color:var(--brand);}
.gz-headline .h .gain{color:var(--hot);}
.gz-headline .s{font-size:14px;color:var(--muted);margin-top:4px;}
.gz-bars{margin:4px 0 6px 0;}
.gz-bar{display:grid;grid-template-columns:230px 1fr 90px;align-items:center;gap:12px;margin:8px 0;}
.gz-bar .l{font-weight:600;font-size:14px;color:var(--ink);}
.gz-bar .track{background:#EEF2F6;border-radius:8px;height:26px;overflow:hidden;}
.gz-bar .fill{height:100%;border-radius:8px;}
.gz-bar .v{font-weight:800;font-size:18px;font-variant-numeric:tabular-nums;}
.gz-tbl{width:100%;border-collapse:separate;border-spacing:0;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden;font-size:14.5px;}
.gz-tbl th{background:#F1F5F9;color:var(--muted);font-weight:700;text-align:right;padding:9px 14px;font-size:13px;}
.gz-tbl th:first-child,.gz-tbl td:first-child{text-align:left;}
.gz-tbl td{padding:9px 14px;border-top:1px solid var(--line);text-align:right;font-variant-numeric:tabular-nums;color:var(--ink);}
.gz-tbl td.fa,.gz-tbl th.fa{background:#FFF7ED;}
.gz-tbl th.fa{color:#9A3412;}
.gz-tbl td.fa{font-weight:800;font-size:17px;color:#9A3412;}
.gz-tbl td.fa.zero{color:var(--good);}
.gz-tbl tr.ai td{font-weight:600;}
.gz-tbl tr.ai td:first-child{color:var(--brand);}
.gz-sig{display:inline-block;border-radius:8px;padding:4px 10px;margin:3px 6px 3px 0;font-size:13px;font-weight:600;border:1px solid var(--line);background:var(--card);}
.gz-sig.pos{border-color:#FDBA74;color:#9A3412;background:#FFF7ED;}
.gz-sig.neg{border-color:#99F6E4;color:#115E59;background:#F0FDFA;}
.gz-caveat{font-size:12.5px;color:var(--muted);background:#F8FAFC;border:1px dashed var(--line);border-radius:10px;padding:8px 12px;margin-top:10px;line-height:1.5;}
.gz-caveat b{color:#334155;}
.gz-sec{font-size:19px;font-weight:700;color:var(--ink);margin:16px 0 6px 0;}
.gz-sec small{font-size:13px;font-weight:500;color:var(--muted);margin-left:8px;}
.gz-empty{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin-bottom:10px;}
.gz-empty .t{font-weight:700;font-size:18px;color:var(--ink);}
.gz-empty .d{color:var(--muted);font-size:14px;margin-top:2px;}
.gz-foot{font-size:12px;color:var(--muted);margin-top:18px;border-top:1px solid var(--line);padding-top:8px;}
div[data-testid="stExpander"] details{background:var(--card);border-radius:12px;}
.stButton button{border-radius:12px;text-align:left;min-height:64px;width:100%;justify-content:flex-start;}
.stButton button *{white-space:normal !important;overflow:visible !important;text-overflow:clip !important;text-align:left;}
.gz-sec.stack small{display:block;margin:2px 0 0 0;}
.gz-prob{display:grid;grid-template-columns:1.05fr 1fr 1fr 1.35fr;gap:0;background:#FFF7ED;border:1px solid #FED7AA;border-radius:14px;margin:0 0 12px 0;overflow:hidden;}
.gz-prob>div{padding:8px 14px;font-size:13px;color:#7C2D12;line-height:1.3;}
.gz-prob>div+div{border-left:1px solid #FED7AA;}
.gz-prob b{font-size:20px;color:#9A3412;font-variant-numeric:tabular-nums;margin-right:4px;}
.gz-prob .lead{font-size:11.5px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;color:#C2410C;display:block;}
.gz-prob .act{background:#F0F9FF;color:#0C4A6E;}
.gz-prob .act .lead{color:var(--brand);}
.gz-prob .src{font-size:11px;color:#9A3412;opacity:.8;}
@media (max-width:1100px){.gz-prob{grid-template-columns:1fr 1fr;}}
</style>""")

# ---------------------------------------------------------------- hero + KPIs + how it works
lo, hi = MR['total_p10_p90']
html(f"""<div class="gz-hero">
<div><div class="gz-brand">{KEY_SVG}<span>GiltzAI</span><span class="city">· Donostia / San Sebastián tourist-flat inspections</span></div>
<h1>Where to look <span>first</span></h1>
<p>Ranks barrios by how many holiday flats are probably <b>missing from the official register</b>, so inspectors spend visits where they find most.</p></div>
<div class="gz-badges"><span class="gz-badge privacy">{LOCK_SVG} Zone level only · no individual flats</span>
<span class="gz-badge">Data: Inside Airbnb {day(S.get('snapshot_listings'))} · REATE register {day(S.get('snapshot_register'))}</span></div>
</div>""")

PC = T.PROBLEM_CONTEXT
html(f"""<div class="gz-prob">
<div><span class="lead">The problem · Donostia</span><b>{PC['ministry_flagged_illegal_donostia']}</b>illegal tourist homes flagged by the Ministry<br/><span class="src">council motion, 24 Oct 2025</span></div>
<div><span class="lead">Who checks them</span><b>{PC['inspection_unit_people_now']}</b>inspector left in the city unit (was {PC['inspection_unit_people_before']})<br/><span class="src">same motion</span></div>
<div><span class="lead">How much</span><b>{PC['closed_2018_2022']}</b>flats closed 2018–22 · fines €{PC['fine_min_eur']:,}–{PC['fine_max_eur'] // 1000}k<br/><span class="src">Ondavasca 2022, citing the city</span></div>
<div class="act"><span class="lead">Who uses GiltzAI · what changes</span>The inspection unit picks this week's barrios → fewer visits to registered flats, more finds per check.</div>
</div>""")

html(f"""<div class="gz-kpis">
<div class="gz-kpi accent" title="Model estimate, 10th–90th percentile over {MR.get('runs', 1000):,} simulations. 'Unregistered' = not found in the official register, not 'illegal'.">
<div class="lab">Probably unregistered flats (city)</div><div class="val">{n0(S['expected_unregistered_total'])}<small>range {lo:.0f}–{hi:.0f}</small></div>
<div class="sub">model estimate · not found in register ≠ illegal</div></div>
<div class="gz-kpi" title="Code missing, or not found in REATE"><div class="lab">Entire homes without a valid code</div><div class="val">{n0(S['entire_homes_no_valid_code'])}</div><div class="sub">code missing or not in the register</div></div>
<div class="gz-kpi" title="Inside Airbnb snapshot"><div class="lab">Airbnb listings in Donostia</div><div class="val">{n0(S['listings_total'])}</div><div class="sub">Inside Airbnb · {day(S.get('snapshot_listings'))}</div></div>
<div class="gz-kpi" title="REATE, Gobierno Vasco"><div class="lab">Units in the official register</div><div class="val">{n0(S['register_units_donostia'])}</div><div class="sub">REATE (Gobierno Vasco) · {day(S.get('snapshot_register'))}</div></div>
</div>""")

html(f"""<div class="gz-steps">
<div class="gz-step"><div class="num">1</div><div><div class="t">Read</div><div class="d">Licence codes in any format in {n0(S['listings_total'])} listings; Claude reads the {n0(S.get('ai_text_read_listings'))} descriptions whose code is not in the register (all room types)</div></div></div>
<div class="gz-step"><div class="num">2</div><div><div class="t">Match</div><div class="d">Against {n0(S['register_units_donostia'])} register units → probability per listing, learned from hand labels</div></div></div>
<div class="gz-step"><div class="num">3</div><div><div class="t">Estimate</div><div class="d">{MR.get('runs', 1000):,} simulations per barrio ({MR.get('jitter_m', 150)} m location blur) → ranked barrios with ranges</div></div></div>
</div>""")

tab_map, tab_base, tab_plan, tab_route, tab_log, tab_chat = st.tabs(['Map & ranking', 'AI vs simple rule', 'Inspection plan', 'Route', 'Field log', 'Ask GiltzAI'])

# ---------------------------------------------------------------- map
METRICS = {
    'expected': ('Volume · expected unregistered flats', 'Expected unregistered flats',
                 'Where most unregistered flats probably are. Big central barrios lead.'),
    'hit_rate': ('Efficiency · hit rate per check', 'Hit rate (share of listings likely unregistered)',
                 'Chance that one check finds an unregistered flat. Best for a small inspection budget.'),
    'naive': ('Baseline · count listings (no AI)', 'Airbnb listings',
              'What a simple approach does: go where there are most listings.'),
}
RAMP = [[254, 240, 217], [253, 204, 138], [252, 141, 89], [227, 74, 51], [179, 0, 0]]   # ColorBrewer OrRd, 5 classes
HIDDEN = [203, 213, 225]


def metric_value(row, m):
    return {'expected': row['expected_unregistered'], 'hit_rate': row['hit_rate'], 'naive': row['naive_listing_count']}[m]


def fmt_metric(v, m):
    return pct(v) if m == 'hit_rate' else (n0(v) if m == 'naive' else n0(v))


def classes(vals):
    """Quantile breaks -> 5 colour classes (robust to the very skewed city centre)."""
    s = pd.Series(vals, dtype=float)
    br = sorted(set(s.quantile([.2, .4, .6, .8]).round(3).tolist()))
    return br


def cls_of(v, br):
    return sum(v > b for b in br)


def ring_centroid(ring):
    a = cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
        c = x0 * y1 - x1 * y0
        a += c; cx += (x0 + x1) * c; cy += (y0 + y1) * c
    if abs(a) < 1e-12:
        return [sum(p[0] for p in ring) / len(ring), sum(p[1] for p in ring) / len(ring)], 0
    return [cx / (3 * a), cy / (3 * a)], abs(a / 2)


def centroid(geom):
    polys = geom['coordinates'] if geom['type'] == 'MultiPolygon' else [geom['coordinates']]
    return max((ring_centroid(p[0]) for p in polys), key=lambda t: t[1])[0]


with tab_map:
    metric = st.radio('Rank barrios by', list(METRICS), horizontal=True, format_func=lambda m: METRICS[m][0], label_visibility='collapsed')
    r = T.rank_zones(metric, 50)
    rows = r['ranking']
    by_zone = {x['barrio']: x for x in rows}
    br = classes([metric_value(x, metric) for x in rows])
    top3 = [x['barrio'] for x in rows[:3]]

    gj = json.load(open(DATA / 'zones.geojson', encoding='utf-8'))
    labels = []
    for f in gj['features']:
        p = f['properties']; z = p['zone_name']; row = by_zone.get(z)
        if row is None:
            p['fill'] = HIDDEN + [150]; p['lw'] = 0
            p['tip'] = f"<b>{escape(nice(z))}</b><br/>Fewer than 5 listings: hidden to protect privacy"
            continue
        v = metric_value(row, metric)
        p['fill'] = RAMP[cls_of(v, br)] + [215]
        p['lw'] = 50 if z in top3 else 0
        p['tip'] = (f"<b>#{row['rank']} {escape(nice(z))}</b><br/>Expected unregistered: <b>{n0(row['expected_unregistered'])}</b> (range {rng(row['range_p10_p90'])})"
                    f"<br/>Hit rate: <b>{pct(row['hit_rate'])}</b> of {row['listings_in_scope']} listings in scope"
                    f"<br/>Rank if counting listings: #{row['naive_rank']}")
        txt_ = unicodedata.normalize('NFKD', f"{nice(z.split(' - ')[0], False)}\n{fmt_metric(v, metric)}".replace('–', '-')).encode('ascii', 'ignore').decode()
        labels.append({'pos': centroid(f['geometry']), 'text': txt_,   # deck.gl default font atlas = ASCII
                       'size': 15 if z in top3 else 12})


    for f in gj['features']:   # privacy: send only display props to the browser (no estimates for hidden barrios in the page payload)
        f['properties'] = {k: f['properties'][k] for k in ('zone_name', 'fill', 'lw', 'tip') if k in f['properties']}
    geo = pdk.Layer('GeoJsonLayer', gj, pickable=True, stroked=True, filled=True, get_fill_color='properties.fill',
                    get_line_color=[15, 23, 42, 200], get_line_width='properties.lw', line_width_min_pixels=1,   # metres (pydeck would turn 'pixels' into an expression)
                    auto_highlight=True, highlight_color=[11, 93, 122, 90])
    txt = pdk.Layer('TextLayer', labels, get_position='pos', get_text='text', get_size='size', get_color=[15, 23, 42, 255],
                    font_weight=700, font_family='"Source Sans Pro, Helvetica, Arial, sans-serif"', font_settings={'sdf': True}, outline_width=2, outline_color=[255, 255, 255, 235],
                    line_height=1.1)
    deck = pdk.Deck(layers=[geo, txt],
                    initial_view_state=pdk.ViewState(latitude=43.301, longitude=-1.997, zoom=11.8),
                    tooltip={'html': '{tip}', 'style': {'backgroundColor': '#0F172A', 'color': 'white', 'fontSize': '13px', 'borderRadius': '8px', 'padding': '8px 10px'}})

    left, right = st.columns([1.75, 1], gap='large')
    with left:
        fmt_b = (lambda b: pct(b)) if metric == 'hit_rate' else (lambda b: n0(b))
        edges = [None] + br + [None]
        sw = ''
        for i in range(len(br) + 1):
            a, b = edges[i], edges[i + 1]
            lab = f'≤ {fmt_b(b)}' if a is None else (f'> {fmt_b(a)}' if b is None else f'{fmt_b(a)}–{fmt_b(b)}')
            c = RAMP[i]
            sw += f'<span><span class="gz-sw" style="background:rgb({c[0]},{c[1]},{c[2]})"></span>{lab}</span>'
        html(f"""<div class="gz-legend"><span class="ttl">{METRICS[metric][1]}</span>{sw}
<span><span class="gz-sw" style="background:rgb({HIDDEN[0]},{HIDDEN[1]},{HIDDEN[2]})"></span>hidden (&lt; 5 listings)</span>
<span><span class="gz-sw top"></span>top 3</span></div>""")
        if _NEW_ST:
            st.pydeck_chart(deck, height=540)
        else:
            try:
                st.pydeck_chart(deck, use_container_width=True, height=540)
            except TypeError:
                st.pydeck_chart(deck, use_container_width=True)
    with right:
        html(f'<div class="gz-sec stack" style="margin-top:0">Start here<small>{escape(METRICS[metric][2])}</small></div>')
        for x in rows[:3]:
            v = metric_value(x, metric)
            if metric == 'expected':
                big = f"{n0(v)} <small>expected · range {rng(x['range_p10_p90'])}</small>"
                meta = f"Hit rate {pct(x['hit_rate'])} · {x['listings_in_scope']} listings · {pct(x['prob_in_top3'])} chance of being top 3"
            elif metric == 'hit_rate':
                big = f"{pct(v)} <small>hit rate · {x['listings_in_scope']} listings</small>"
                meta = f"Expected {n0(x['expected_unregistered'])} unregistered (range {rng(x['range_p10_p90'])})"
            else:
                big = f"{n0(v)} <small>listings</small>"
                meta = f"Expected {n0(x['expected_unregistered'])} unregistered · hit rate {pct(x['hit_rate'])}"
            chip = ''
            if metric != 'naive':
                d = x['naive_rank'] - x['rank']
                chip = (f'<span class="gz-chip up">▲ {d} vs counting listings</span>' if d > 0 else
                        f'<span class="gz-chip down">▼ {-d} vs counting listings</span>' if d < 0 else '')
            html(f"""<div class="gz-card"><div class="gz-rank"><div class="pos">{x['rank']}</div>
<div><div class="name">{escape(nice(x['barrio']))}{chip}</div><div class="big">{big}</div><div class="meta">{meta}</div></div></div></div>""")
        hidden = ', '.join(nice(z, False) for z in r['suppressed_small_barrios'])
        html(f'<div class="gz-caveat"><b>Hidden for privacy</b> (fewer than 5 listings): {escape(hidden)}.<br/>'
             f'Ranges = 10th–90th percentile over {MR.get("runs", 1000):,} simulations. Hit rate = expected ÷ listings the simulations place in the barrio (blurred locations near borders move between barrios), so it can differ slightly from expected ÷ listings shown. A ranking is a reason to look, never evidence against a flat.</div>')

    html(f'<div class="gz-sec">All barrios<small>sorted by {escape(METRICS[metric][1].lower())} · hover the map for details</small></div>')
    df = pd.DataFrame([{
        'Rank': x['rank'], 'Barrio': nice(x['barrio']),
        'Expected unregistered': half_up(x['expected_unregistered']), 'Range (P10–P90)': rng(x['range_p10_p90']),
        'Hit rate': (x['hit_rate'] or 0) * 100, 'Listings in scope': x['listings_in_scope'],
        'Rank by listing count': x['naive_rank'], 'Chance of top 3': (x['prob_in_top3'] or 0) * 100} for x in rows])
    show_df(df, height=35 * (len(df) + 1) + 3, column_config={
        'Rank': st.column_config.NumberColumn(width='small'),
        'Expected unregistered': st.column_config.NumberColumn(format='%.0f', help='Listings probably not in the official register (model mean)'),
        'Range (P10–P90)': st.column_config.TextColumn(help='10th–90th percentile over the simulations'),
        'Hit rate': st.column_config.ProgressColumn(format='%.0f%%', min_value=0, max_value=100, help='Share of the barrio\'s listings likely unregistered = expected finds per check (denominator = listings placed in the barrio across the 1,000 simulations)'),
        'Listings in scope': st.column_config.NumberColumn(help='Listings whose blurred point falls in the barrio, excluding hotels and other licence types'),
        'Rank by listing count': st.column_config.NumberColumn(format='#%d', help='Rank a simple count of Airbnb listings would give'),
        'Chance of top 3': st.column_config.NumberColumn(format='%.0f%%', help='Share of simulations in which the barrio is in the top 3 by volume'),
    })

    with st.expander('Why is a barrio ranked here?  ·  evidence behind each estimate'):
        names = [x['barrio'] for x in rows]
        z = st.selectbox('Barrio', names, format_func=nice)
        e = T.explain_zone(z)
        if 'error' in e:
            st.error(e['error'])
        elif e.get('suppressed'):
            st.info(e['reason'])
        else:
            html(f"""<div class="gz-kpis">
<div class="gz-kpi"><div class="lab">Expected unregistered</div><div class="val">{n0(e['expected_unregistered'])}<small>range {rng(e['range_p10_p90'])}</small></div></div>
<div class="gz-kpi"><div class="lab">Hit rate per check</div><div class="val">{pct(e['hit_rate'])}</div></div>
<div class="gz-kpi"><div class="lab">GiltzAI rank (volume)</div><div class="val">#{e['ai_rank']}<small>counting listings: #{e['naive_rank']}</small></div></div>
<div class="gz-kpi"><div class="lab">Listings in scope</div><div class="val">{n0(e['listings_in_scope'])}</div></div></div>""")
            REASONS = {'no_licence_code_anywhere': 'No licence code anywhere', 'code_not_found_in_register': 'Code shown but not found in the register',
                       'claims_seasonal_rental_exemption': 'Claims the seasonal-rental exemption', 'code_matches_register': 'Code matches the register (registered)',
                       'midterm_claim_but_allows_short_stays': 'Says mid-term rental but allows short stays',
                       'midterm_rental_claims_read_in_description': 'Mid-term rental claim read by Claude in the description',
                       'unmatched_new_listings_under_1_year': 'Unmatched and listed for less than 1 year',
                       'register_code_reused_by_several_listings': 'Register code reused by several listings',
                       'unmatched_listings_of_hosts_with_5plus_listings': 'Unmatched, host with 5+ listings',
                       'unmatched_listings_of_mid_size_hosts_4_to_10': 'Unmatched, mid-size host (4–10 listings)',
                       'unmatched_with_fully_blocked_calendar': 'Unmatched, calendar fully blocked'}
            rs = pd.DataFrame([{'Signal': REASONS.get(k, k.replace('_', ' ').capitalize()), 'Listings': str(v)} for k, v in e['reasons'].items()])
            show_df(rs, height=35 * (len(rs) + 1) + 3)
            st.caption(f"{e['location_uncertainty']}  ·  Counts below 5 shown as “<5”.  ·  Source: {e['source']}")

# ---------------------------------------------------------------- baseline
SIGNALS = {'entire_home': 'Entire home', 'has_code': 'Shows a licence code', 'code_in_register': 'Code found in register',
           'seasonal_claim': 'Claims seasonal exemption', 'code_not_in_register': 'Code not in register', 'no_code': 'No licence code',
           'code_from_text_only': 'Code only in the text', 'code_nonstandard_format': 'Code in non-standard format', 'code_shared': 'Code shared by listings',
           'several_codes': 'Several codes', 'exact_far_or_size_mismatch': 'Register entry far away / size mismatch', 'active_ltm': 'Active last 12 months',
           'calendar_blocked': 'Calendar fully blocked', 'host_5plus': 'Host with 5+ listings', 'superhost': 'Superhost', 'new_listing': 'New listing (< 1 year)',
           'seasonal_but_short_stays': 'Mid-term claim but short stays', 'host_mid_4_10': 'Mid-size host (4–10)', 'llm_midterm_claim': 'Claude read a mid-term claim'}

with tab_base:
    b = T.compare_baseline()
    lm = b['listing_matching']
    naive_m, ai_m, wrong = lm['naive_exact_field_matches_entire_homes'], lm['giltzai_normalised_matches_entire_homes'], lm['registered_flats_naive_rule_would_wrongly_flag']
    html(f"""<div class="gz-headline"><div class="k">Fewer wasted visits</div>
<div class="h">Reading licence codes the way a person does, GiltzAI recognises <b>{n0(ai_m)}</b> registered entire homes where an exact-match rule sees <b>{n0(naive_m)}</b>:
<span class="gain">{n0(wrong)} homes with a valid register code</span> that the simple rule would wrongly send inspectors to.</div>
<div class="s">Codes are written in many formats (spaces, dashes, missing prefixes, inside the description). The simple rule only accepts a perfect copy of the register code.</div></div>""")
    mx = max(naive_m, ai_m) or 1
    html(f"""<div class="gz-bars">
<div class="gz-bar"><div class="l">Simple rule (exact code match)</div><div class="track"><div class="fill" style="width:{naive_m / mx * 100:.1f}%;background:#94A3B8"></div></div><div class="v">{n0(naive_m)}</div></div>
<div class="gz-bar"><div class="l">GiltzAI code reading</div><div class="track"><div class="fill" style="width:{ai_m / mx * 100:.1f}%;background:#0B5D7A"></div></div><div class="v" style="color:#0B5D7A">{n0(ai_m)}</div></div>
</div>""")
    st.caption('Entire homes recognised as registered (higher = fewer registered flats wrongly flagged).')

    hb = b['hand_label_benchmark']
    _rb = hb.get('rows_by_who', {}) if isinstance(hb, dict) else {}
    html(f'<div class="gz-sec">Checked against labelled listings<small>{hb.get("labelled_rows", "?") if isinstance(hb, dict) else "?"} listings looked up in the register: '
         f'{_rb.get("human", "?")} labelled by hand, {_rb.get("auto", "?")} by a strict rule (same code ≤300 m, similar capacity)</small></div>')
    if isinstance(hb, dict):
        names = {'baseline_exact_field': 'Simple rule: exact licence match', 'ai_prior_rules': 'GiltzAI rules only', 'ai_logreg_loo': 'GiltzAI model (leave-one-out)'}
        trs = ''
        for m, label in names.items():
            if m not in hb:
                continue
            q = hb[m]
            cls = 'ai' if m != 'baseline_exact_field' else ''
            trs += (f'<tr class="{cls}"><td>{label}</td><td class="fa{" zero" if q["fp"] == 0 else ""}">{q["fp"]}</td><td>{q["fn"]}</td>'
                    f'<td>{pct(q["precision"])}</td><td>{pct(q["recall"])}</td><td>{q["f1"]:.2f}</td></tr>')
        c1, c2 = st.columns([1.6, 1], gap='large')
        with c1:
            html(f"""<table class="gz-tbl"><tr><th>Method</th><th class="fa">False alarms<br/><span style="font-weight:500">registered flats flagged</span></th>
<th>Missed</th><th>Precision</th><th>Recall</th><th>F1</th></tr>{trs}</table>""")
        with c2:
            if 'plain_summary' in b:
                html(f'<div class="gz-card" style="font-size:14.5px;line-height:1.5">{escape(b["plain_summary"])}</div>')
        if 'strongest_signals' in b:
            chips = ''.join(f'<span class="gz-sig {"pos" if v > 0 else "neg"}">{"+" if v > 0 else "−"} {escape(SIGNALS.get(k, k.replace("_", " ")))}'
                            f' <span style="opacity:.6">{abs(v):.2f}</span></span>' for k, v in b['strongest_signals'].items())
            html(f'<div class="gz-sec">What the model learned<small>orange = more likely not in register · green = less likely</small></div><div>{chips}</div>')
    else:
        st.info(f'Benchmark {hb}')

    t1, t2, t3 = S.get('ai_text_read_listings'), S.get('ai_text_hotel_like_excluded'), S.get('ai_text_midterm_claims')
    html(f"""<div class="gz-sec">Claude reads what hosts write<small>only declarations are extracted; Claude never decides legality</small></div>
<div class="gz-kpis" style="grid-template-columns:repeat(3,minmax(0,1fr))">
<div class="gz-kpi"><div class="lab">Descriptions read (code not in register, all room types)</div><div class="val">{n0(t1)}</div></div>
<div class="gz-kpi"><div class="lab">Hotels / guesthouses found → excluded</div><div class="val">{n0(t2)}</div></div>
<div class="gz-kpi"><div class="lab">Declared mid-term rentals</div><div class="val">{n0(t3)}</div></div></div>""")

    moved = b['zone_ranking_differences']['barrios_moving_2plus_places_vs_naive']
    if moved:
        html('<div class="gz-sec">Where GiltzAI disagrees with counting listings<small>barrios that move 2+ places (volume ranking)</small></div>')
        mv = pd.DataFrame([{'Barrio': nice(m['zone_name']), 'Rank by listing count': m['rank_naive'], 'GiltzAI rank': m['rank_ai'],
                            'Change': (f"▲ up {m['rank_naive'] - m['rank_ai']}" if m['rank_naive'] > m['rank_ai'] else f"▼ down {m['rank_ai'] - m['rank_naive']}")}
                           for m in moved])
        show_df(mv, height=35 * (len(mv) + 1) + 3, column_config={'Rank by listing count': st.column_config.NumberColumn(format='#%d'),
                                                                   'GiltzAI rank': st.column_config.NumberColumn(format='#%d')})

    # honest caveats: visible but small
    cav = []
    if isinstance(hb, dict):
        rb = hb.get('rows_by_who', {})
        who = ', '.join(f'{v} {k}' for k, v in rb.items()) if isinstance(rb, dict) else str(rb)
        cav.append(f"<b>Small sample:</b> {hb.get('n_eval', '?')} labelled listings used for scoring ({hb.get('labelled_rows')} labelled: {who}; "
                   f"{hb.get('label_corrections', 0)} corrected after register lookup; self-agreement {hb.get('self_agreement')}). Indicative, not definitive.")
    tk = b.get('zone_ranking_topk')
    if tk:
        parts = [f"{k.replace('top', 'top ')}: {v['ai_found']} vs {v['naive_found']} found, {n0(v['ai_listings_to_check'])} vs {n0(v['naive_listings_to_check'])} listings to check"
                 for k, v in tk.items()]
        cav.append('<b>By volume, the top barrios are the same as counting listings</b> (hand-labelled finds, GiltzAI vs naive — ' + '; '.join(parts) +
                   '). The gain comes from reading codes and from efficiency planning (see Inspection plan).')
    if b.get('labelling_caveat'):
        cav.append('<b>Who labelled:</b> ' + escape(b['labelling_caveat']))
    zs = b.get('zone_efficiency_vs_simple_share')
    if zs:
        cav.append(f"<b>Barrio order vs a no-AI share rule:</b> ordering barrios by the share of listings failing an exact code match gives almost the same efficiency order "
                   f"(rank correlation {zs['spearman_rank_correlation']:.2f}). At barrio level the AI adds counts with ranges and the hotel / mid-term exclusions, not a different order.")
    stb = b.get('stability', {})
    cav.append(f"<b>Stability:</b> top 3 identical under 3 assumptions for seasonal-rental claims: {'yes' if stb.get('top3_identical') else 'no'}. "
               f"<b>Model:</b> {escape(str(b.get('model_used')))}. Positive class = “not found in register”; unclear labels excluded.")
    html('<div class="gz-caveat">' + '<br/>'.join(cav) + '</div>')

# ---------------------------------------------------------------- plan
with tab_plan:
    p1, p2, p3 = st.columns(3)
    days = p1.number_input('Inspector-days available', 1, 60, 6, help='e.g. 2 inspectors × 3 days = 6')
    per_day = p2.number_input('Checks per inspector-day', 1, 30, 8, help='A check = reviewing one listing in that barrio')
    strat = p3.selectbox('Strategy', ['hit_rate', 'expected'],
                         format_func={'hit_rate': 'Efficiency first (highest hit rate)', 'expected': 'Volume first (most expected flats)'}.get)
    plan = T.inspection_plan(int(days), int(per_day), strat)
    ai_f, nv_f, g = plan['giltzai_expected_found'] or 0, plan['naive_expected_found'] or 0, plan['gain_percent']
    sh_f = plan.get('simple_share_rule_expected_found') or 0
    gain_txt = f'<span class="gain">{g:+.0f}%</span>' if g is not None else ''
    _fl = plan.get('field_log_checks_used') or 0
    _fl_txt = f' · updated with {_fl} field check{"s" if _fl != 1 else ""} from the Field log' if _fl else ''
    html(f"""<div class="gz-headline"><div class="k">{plan['checks_available']:,} checks ({int(days)} inspector-days × {int(per_day)}){_fl_txt}</div>
<div class="h">Planning by {'efficiency' if strat == 'hit_rate' else 'volume'}, GiltzAI expects to find <b>{n1(ai_f)}</b> unregistered flats vs <b>{n1(nv_f)}</b> going where most listings are {gain_txt}</div>
<div class="s">Same budget, same model scoring every plan: the difference is only <i>where</i> the checks go. Honest check: a no-AI rule that orders barrios by the share of listings
without an exact code match reaches <b>{n1(sh_f)}</b>, so most of this gain comes from planning by efficiency. The AI's own gain is upstream: {n0(S['entire_homes_matched_register'] - S['entire_homes_naive_exact_match'])} registered homes not sent to inspection, hotels and mid-term declarations read, and ranges.</div></div>""")
    mx = max(ai_f, nv_f, sh_f) or 1
    html(f"""<div class="gz-bars">
<div class="gz-bar"><div class="l">GiltzAI plan</div><div class="track"><div class="fill" style="width:{ai_f / mx * 100:.1f}%;background:#0B5D7A"></div></div><div class="v" style="color:#0B5D7A">{n1(ai_f)}</div></div>
<div class="gz-bar"><div class="l">Naive plan (most listings first)</div><div class="track"><div class="fill" style="width:{nv_f / mx * 100:.1f}%;background:#94A3B8"></div></div><div class="v">{n1(nv_f)}</div></div>
<div class="gz-bar"><div class="l" style="font-weight:500;color:#526077">No-AI share rule (reference)</div><div class="track"><div class="fill" style="width:{sh_f / mx * 100:.1f}%;background:#CBD5E1"></div></div><div class="v" style="color:#526077;font-size:15px">{n1(sh_f)}</div></div>
</div>""")

    def plan_df(rows_):
        d = pd.DataFrame([{'Order': i + 1, 'Barrio': nice(x['barrio']), 'Checks': x['checks'], 'Expected finds': x['expected_found']} for i, x in enumerate(rows_)])
        return d if len(d) else pd.DataFrame(columns=['Order', 'Barrio', 'Checks', 'Expected finds'])
    cfg = {'Order': st.column_config.NumberColumn(width='small'), 'Expected finds': st.column_config.NumberColumn(format='%.1f')}
    l, r_ = st.columns(2, gap='large')
    with l:
        html('<div class="gz-sec" style="margin-top:6px">GiltzAI plan<small>go here, in this order</small></div>')
        gp = plan_df(plan['giltzai_plan'])
        show_df(gp, height=35 * (len(gp) + 1) + 3, column_config=cfg)
    with r_:
        html('<div class="gz-sec" style="margin-top:6px">Naive plan<small>by number of listings</small></div>')
        npd = plan_df(plan['naive_plan_by_listing_count'])
        show_df(npd, height=35 * (len(npd) + 1) + 3, column_config=cfg)
    st.download_button('Download GiltzAI plan (CSV)', gp.to_csv(index=False).encode('utf-8'), 'giltzai_inspection_plan.csv', 'text/csv')
    _ck = plan['checks_available'] or 1
    html(f'<div class="gz-card" style="border-left:6px solid #047857"><b>Pilot indicator · how to know if GiltzAI is right</b><br/>'
         f'Log every check (barrio, found in REATE yes/no). The model predicts <b>{ai_f / _ck * 100:.0f}%</b> of checks in this plan find a flat missing from the register '
         f'(vs {nv_f / _ck * 100:.0f}% going where most listings are). If the field rate is clearly lower, the ranking is wrong for that barrio: '
         f'add the results as new labels and retrain before the next plan.</div>')
    html('<div class="gz-caveat">Expected finds are GiltzAI model estimates per barrio (the same model scores every plan), not a list of flats. '
         'A check = reviewing one listing in that barrio. Every case still requires an official inspection.</div>')

# ---------------------------------------------------------------- route (barrio-level field route)
DAY_COL = [[11, 93, 122], [217, 119, 6], [124, 58, 237], [4, 120, 87], [190, 18, 60]]
with tab_route:
    ze_all = T._load()['ze']
    pub_names = ze_all[ze_all.publishable.astype(bool)].zone_name.tolist()
    all_names = sorted(ze_all.zone_name.tolist(), key=lambda z: nice(z))
    c1, c2, c3, c4, c5 = st.columns([1.4, .8, .8, .9, 1.1])
    start = c1.selectbox('Start from (each day)', all_names, index=all_names.index('ANTIGUA') if 'ANTIGUA' in all_names else 0, format_func=nice)
    r_days = c2.number_input('Days', 1, 5, 1)
    r_hours = c3.number_input('Hours per day', 2, 10, 6)
    r_mpc = c4.number_input('Minutes per check', 5, 90, 20, help='Time to review one listing on site, incl. finding it')
    r_mode = c5.radio('Travel', ['car', 'bike', 'walk'], horizontal=True, format_func={'car': 'Car', 'bike': 'Bike', 'walk': 'Walk'}.get)
    first = st.checkbox(f'Check {nice(start)} first (even if its hit rate is lower)', value=False)
    rp = T.route_plan(start, int(r_days), float(r_hours), int(r_mpc), r_mode, 8, first)
    if 'error' in rp:
        st.warning(rp['error'])
    else:
        hr_cmp, nv_cmp = rp['comparison_hit_rate_order_ignoring_travel'], rp['comparison_most_listings_first']
        gain = (rp['expected_finds'] / nv_cmp['expected_finds'] - 1) * 100 if nv_cmp['expected_finds'] else None
        html(f"""<div class="gz-headline"><div class="k">{int(r_days)} day{'s' if r_days > 1 else ''} from {escape(nice(start))} · {int(r_hours)} h/day · {int(r_mpc)} min per check · {r_mode}</div>
<div class="h">GiltzAI route: <b>{n1(rp['expected_finds'])}</b> expected finds in <b>{rp['checks']}</b> checks, <b>{rp['travel_minutes']}</b> min travelling
{f'<span class="gain">{gain:+.0f}%</span>' if gain is not None else ''} vs going where most listings are ({n1(nv_cmp['expected_finds'])})</div>
<div class="s">At every step it goes to the barrio with most expected finds per minute (travel between barrio centres + time per check), using the hit rates updated by the
Field log ({rp['field_log_used']} checks logged). Ordering by hit rate and ignoring travel gives {n1(hr_cmp['expected_finds'])} finds with {hr_cmp['travel_minutes']} min travelling.
Barrio level only: inside each barrio inspectors pick listings with their usual procedure.</div></div>""")
        C = T._centroids()
        gj_r = json.load(open(DATA / 'zones.geojson', encoding='utf-8'))
        for ft in gj_r['features']:
            ft['properties'] = {'zone_name': ft['properties']['zone_name'],
                                'fill': [226, 232, 240, 120] if ft['properties']['zone_name'] not in pub_names else [241, 245, 249, 200],
                                'tip': escape(nice(ft['properties']['zone_name']))}
        paths, pts, lbl = [], [], []
        for d in rp['route']:
            col = DAY_COL[(d['day'] - 1) % len(DAY_COL)]
            seq = [start] + [x['barrio'] for x in d['stops']] + [start]
            paths.append({'path': [[C[z][1], C[z][0]] for z in seq], 'color': col, 'tip': f"Day {d['day']}"})
            for i, x in enumerate(d['stops'], 1):
                la, lo = C[x['barrio']]
                pts.append({'pos': [lo, la], 'color': col, 'tip': f"Day {d['day']} · stop {i}: {escape(nice(x['barrio']))} · {x['checks']} checks · ~{x['expected_finds']} finds"})
                lbl.append({'pos': [lo, la], 'text': f"{d['day']}.{i}" if r_days > 1 else str(i)})
        sla, slo = C[start]
        layers = [pdk.Layer('GeoJsonLayer', gj_r, stroked=True, filled=True, get_fill_color='properties.fill', get_line_color=[100, 116, 139, 160],
                            line_width_min_pixels=1, pickable=True),
                  pdk.Layer('PathLayer', paths, get_path='path', get_color='color', width_min_pixels=4, get_width=5, pickable=True),
                  pdk.Layer('ScatterplotLayer', pts, get_position='pos', get_fill_color='color', get_radius=170, radius_min_pixels=13, pickable=True,
                            stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=2),
                  pdk.Layer('ScatterplotLayer', [{'pos': [slo, sla], 'tip': f'Start: {escape(nice(start))}'}], get_position='pos', get_fill_color=[15, 23, 42],
                            get_radius=140, radius_min_pixels=9, pickable=True),
                  pdk.Layer('TextLayer', lbl, get_position='pos', get_text='text', get_size=13, get_color=[255, 255, 255, 255], font_weight=700)]
        deck_r = pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=43.303, longitude=-1.99, zoom=11.6),
                          tooltip={'html': '{tip}', 'style': {'backgroundColor': '#0F172A', 'color': 'white', 'fontSize': '13px', 'borderRadius': '8px'}})
        lc, rc = st.columns([1.5, 1], gap='large')
        with lc:
            if _NEW_ST:
                st.pydeck_chart(deck_r, height=470)
            else:
                try:
                    st.pydeck_chart(deck_r, use_container_width=True, height=470)
                except TypeError:
                    st.pydeck_chart(deck_r, use_container_width=True)
        with rc:
            for d in rp['route']:
                col = DAY_COL[(d['day'] - 1) % len(DAY_COL)]
                rows_ = ''.join(f"<tr><td>{i}</td><td>{escape(nice(x['barrio']))}</td><td>{x['travel_min']} min</td><td>{x['checks']}</td><td>{x['expected_finds']}</td></tr>"
                                for i, x in enumerate(d['stops'], 1))
                html(f"""<div class="gz-card" style="border-left:6px solid rgb({col[0]},{col[1]},{col[2]})"><b>Day {d['day']}</b> · start {escape(nice(start))}
<table style="width:100%;font-size:14px;margin-top:6px"><tr style="color:#526077"><td>#</td><td>Barrio</td><td>Travel</td><td>Checks</td><td>Exp. finds</td></tr>{rows_}</table>
<div style="color:#526077;font-size:13px;margin-top:4px">Back to start: {d['travel_back_to_start_min']} min</div></div>""")
        html('<div class="gz-caveat">Travel times are estimates (straight line between barrio centres × 1.3, urban speeds: car 15 km/h incl. parking, bike 12, walk 4.5). '
             'Expected finds come from the GiltzAI model, updated by the Field log. Every case still requires an official inspection.</div>')

# ---------------------------------------------------------------- field log (inspector feedback loop)
def _rerun():
    (getattr(st, 'rerun', None) or getattr(st, 'experimental_rerun'))()


if st.session_state.get('toast'):
    st.toast(st.session_state.pop('toast'))
RES_LABEL = {'not_in_register': 'Not in the register', 'registered': 'Registered (legal)', 'closed_or_inactive': 'Closed / inactive', 'out_of_scope': 'Hotel or mid-term (out of scope)'}
with tab_log:
    html("""<div class="gz-headline"><div class="k">The plan learns from the field</div>
<div class="h">Log what each check found. GiltzAI updates that barrio's hit rate and re-plans.</div>
<div class="s">Bayesian update: the model's hit rate counts as 10 checks, so every real check moves it. The Inspection plan and Route tabs use the updated rates.
Only barrio + result + count are stored: never addresses, codes or flat details.</div></div>""")
    fl, fr = st.columns([1, 1.6], gap='large')
    with fl:
        with st.form('log_form', clear_on_submit=True):
            lb = st.selectbox('Barrio', pub_names, format_func=nice)
            lres = st.radio('Result', list(RES_LABEL), format_func=RES_LABEL.get)
            ln = st.number_input('Number of checks with this result', 1, 30, 1)
            sent = st.form_submit_button('Log result', type='primary')
        if sent:
            st.session_state['last_log'] = T.log_check(lb, lres, int(ln))
            _rerun()   # earlier tabs (plan, route) re-render with the updated hit rates
        msg = st.session_state.get('last_log')
        if msg:
            if 'error' in msg:
                st.warning(msg['error'])
            else:
                st.success(f"Logged {msg['logged']['checks']} × {RES_LABEL[msg['logged']['result']]} in {nice(msg['logged']['barrio'])}: "
                           f"hit rate {pct(msg['hit_rate_before'])} → {pct(msg['hit_rate_after'])}.")
        b1, b2 = st.columns(2)
        if b1.button('Undo last entry'):
            st.session_state['last_log'] = None
            u = T.undo_last_check()
            st.session_state['toast'] = u.get('error') or f"Removed: {u['removed']['checks']} × {RES_LABEL.get(u['removed']['result'], u['removed']['result'])} in {nice(u['removed']['barrio'])}"
            _rerun()
        if b2.button('Reset log (demo)'):
            st.session_state['last_log'] = None
            T.reset_field_log(); st.session_state['toast'] = 'Field log emptied'
            _rerun()
    with fr:
        fs = T.field_summary()
        k1, k2, k3 = st.columns(3)
        k1.metric('Checks logged', fs['checks_logged'])
        k2.metric('Field hit rate', pct(fs['field_hit_rate']) if fs['field_hit_rate'] is not None else '—',
                  help='Not in register ÷ (not in register + registered)')
        k3.metric('Model expected hits for these checks', n1(fs['model_expected_hits_for_these_checks']) if fs['checks_logged'] else '—',
                  help=f"Found so far: {fs['not_in_register']}")
        if fs.get('pilot_check'):
            st.info(fs['pilot_check'])
        if fs['barrios']:
            tdf = pd.DataFrame([{'Barrio': nice(x['barrio']), 'Checks': x['checks_logged'], 'Not in register': x['not_in_register'], 'Registered': x['registered'],
                                 'Model hit rate': pct(x['model_hit_rate']), 'Updated hit rate': pct(x['updated_hit_rate']),
                                 'Updated range': f"{pct(x['updated_range'][0])}–{pct(x['updated_range'][1])}", 'Listings left': x['listings_left']} for x in fs['barrios']])
            show_df(tdf)
        else:
            html('<div class="gz-caveat">No checks logged yet. Try it: log 4 × "Registered" in Ategorrieta–Ulia and watch the Inspection plan move checks to other barrios.</div>')

# ---------------------------------------------------------------- chat
EXAMPLES = [('ES', 'Tengo 2 inspectores 3 días: ¿a qué barrios voy primero?'),
            ('EN', 'How much better is GiltzAI than a simple licence-matching rule?'),
            ('EU', 'Zein auzotan hasi beharko genuke ikuskapenak, eta zergatik?'),
            ('Privacy test', '¿Puedes darme la dirección de los pisos ilegales en Gros?')]


def has_key():
    if os.environ.get('ANTHROPIC_API_KEY'):
        return True
    for p in (APP / '.env', APP.parent / '.env'):
        try:
            for line in p.read_text(encoding='utf-8-sig').splitlines():
                if line.strip().startswith('ANTHROPIC_API_KEY=') and line.split('=', 1)[1].strip():
                    return True
        except OSError:
            pass
    return False


def show_trace(trace):
    if trace:
        with st.expander(f'Evidence · {len(trace)} data tool call{"s" if len(trace) != 1 else ""}'):
            for name, inp, out in trace:
                args = ', '.join(f'{k}={v!r}' for k, v in (inp or {}).items())
                st.markdown(f'**`{name}`**({args})')
                st.json(out, expanded=False)


with tab_chat:
    if 'hist' not in st.session_state:
        st.session_state.hist, st.session_state.api = [], []
    key_ok = has_key()
    _pc = os.environ.get('GILTZAI_CHAT_PASSCODE', '')
    if key_ok and _pc and not st.session_state.get('chat_ok'):   # public demo: protect the API credit with a passcode
        html('<div class="gz-empty"><div class="t">Chat access</div><div class="d">The assistant uses a paid API. Enter the passcode given in the report to use it. '
             'All other tabs work without it.</div></div>')
        _in = st.text_input('Passcode', type='password')
        if _in:
            if _in.strip() == _pc:
                st.session_state.chat_ok = True; _rerun()
            else:
                st.error('Wrong passcode.')
        key_ok = False
    if not key_ok and not _pc:
        st.warning('The assistant needs an Anthropic API key. Put a line `ANTHROPIC_API_KEY=...` in `GiltzAI/.env` and restart the app. '
                   'The map, comparison and inspection-plan tabs work without it.')
    if not st.session_state.hist:
        html("""<div class="gz-empty"><div class="t">Ask about the barrios, in Spanish, Basque or English</div>
<div class="d">GiltzAI answers with numbers from its data tools only: open “Evidence” under each answer to see which tool produced each figure.
It works at zone level and will not discuss individual flats or hosts.</div></div>""")
    cols = st.columns(len(EXAMPLES))
    clicked = None
    for c, (tag, q_) in zip(cols, EXAMPLES):
        if c.button(f'{tag} · {q_}', key=f'ex_{tag}', disabled=not key_ok):
            clicked = q_
    msgs = st.container()
    with msgs:
        for role, text, trace in st.session_state.hist:
            with st.chat_message(role):
                st.markdown(text)
                show_trace(trace)
    q = st.chat_input('Ask GiltzAI… e.g. ¿Qué barrio tiene más pisos sin registrar?', disabled=not key_ok) or clicked
    if q:
        st.session_state.hist.append(('user', q, None))
        with msgs:
            with st.chat_message('user'):
                st.markdown(q)
            with st.chat_message('assistant'):
                with st.spinner('Checking the data…'):
                    try:
                        from agent import ask
                        ans, trace, st.session_state.api = ask(q, st.session_state.api)
                    except Exception as e:
                        ans, trace = None, []
                        print(f'[app] assistant error: {type(e).__name__}')
                        st.error(f'The assistant could not answer ({type(e).__name__}). Check the API key and the internet connection.')
                if ans:
                    st.markdown(ans)
                    show_trace(trace)
        if ans:
            st.session_state.hist.append(('assistant', ans, trace))
        else:
            st.session_state.hist.pop()

# ---------------------------------------------------------------- footer
with st.expander('About the data, method and privacy'):
    st.markdown(f"""
- **Data:** Inside Airbnb Euskadi ({S.get('snapshot_listings')}), REATE tourist-flat register, Gobierno Vasco ({S.get('snapshot_register')}), Donostia city census ({S.get('snapshot_city_census')}) and official barrio boundaries (open data).
- **Method:** licence-code reading + Claude reading descriptions → register matching → probability per listing (learned from hand labels) → {MR.get('runs', 1000):,} simulations per barrio ({MR.get('jitter_m', 150)} m location blur) → ranges.
- **Privacy:** zone level only; hosts hashed; barrios with fewer than 5 listings hidden; no addresses or listing ids anywhere in the app.
- **Meaning:** “unregistered” = not found in the official register, not “illegal”. Every case needs an official inspection.
- **Problem context** (documents, not model output): {T.PROBLEM_CONTEXT['src_motion']}; {T.PROBLEM_CONTEXT['src_press']}.
- **Running it after the pilot:** {' '.join(T.data_sources()['operations_and_scale'])}
- **Not used:** guest ratings (same median, 4.8, for matched and unmatched listings); Booking/Vrbo (no public data).
- **Licences & credits:** Inside Airbnb (CC BY 4.0) · Gobierno Vasco – Open Data Euskadi, REATE (CC BY 4.0) · Ayuntamiento de Donostia – datos abiertos (CC BY-SA 3.0; GiltzAI's derived barrio tables are shared under the same licence) · basemap © OpenStreetMap contributors © CARTO.
- **AI use:** the assistant runs on Claude (Anthropic); Claude also read listing descriptions and helped write the code and documents. Team GiltzAI: Carlos Tesoriero, Marcos Tesoriero.
""")
html('<div class="gz-foot">GiltzAI · Team GiltzAI · Donostia Meeting Minds 2026 · City category · prototype. Figures computed from the data bundle at run time. '
     'Data: Inside Airbnb (CC BY 4.0) · Open Data Euskadi (CC BY 4.0) · Donostia datos abiertos (CC BY-SA 3.0) · map © OpenStreetMap © CARTO · built with Claude (Anthropic).</div>')
