"""GiltzAI - AI text reading step (Claude). Reads the description of every listing WITHOUT a valid register code and returns
structured signals. It never decides legality; it only extracts what the host declares.
Input : out/_llm_input.csv (listing_id, name, description, minimum_nights, ...)   Output: out/llm_text_signals.csv
Run   : python3 scripts/llm_read_descriptions.py   (needs ANTHROPIC_API_KEY)"""
import os, sys, json, time, re
import pandas as pd, anthropic

IN = sys.argv[1] if len(sys.argv) > 1 else 'out/_llm_input.csv'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'out/llm_text_signals.csv'
MODEL = os.environ.get('GILTZAI_MODEL', 'claude-haiku-5-5')
client = anthropic.Anthropic()
SYSTEM = """You read Airbnb listing texts from Donostia / San Sebastián and extract ONLY what the text explicitly states. Do not guess.
For each listing return an object with:
- "id": the id given
- "midterm_or_non_tourist_claim": true if the text says it is a temporary / seasonal / mid-term / non-tourist rental (e.g. "alquiler de temporada", "temporary non-touristic", "Real Decreto 1312/2024", "minimum one month", "contract required", "for students/workers"), else false
- "hotel_like": true if the text says it is a hotel, hostel, aparthotel, student residence, co-living building or similar commercial establishment (not a private home), else false
- "licence_text": true if the text mentions a licence / registration number or that the flat is registered, else false
- "code": any tourist registration code written in the text, normalised like ESS01234 or LSS00123 (letters + 5 digits), or null
- "evidence": a quote of at most 12 words supporting the strongest true field, or "" if all false
Return ONLY a JSON array, no prose."""


def run_batch(rows):
    items = [{'id': str(r.listing_id), 'title': str(r['name'])[:150], 'text': str(r.description)[:1500]} for _, r in rows.iterrows()]
    for attempt in range(4):
        try:
            resp = client.messages.create(model=MODEL, max_tokens=3000, system=SYSTEM,
                                          messages=[{'role': 'user', 'content': json.dumps(items, ensure_ascii=False)}])
            txt = ''.join(b.text for b in resp.content if b.type == 'text')
            txt = txt[txt.find('['): txt.rfind(']') + 1]
            return json.loads(txt), resp.usage
        except Exception as e:
            print('retry', attempt, type(e).__name__, str(e)[:120]); time.sleep(2 + 3 * attempt)
    return [], None


d = pd.read_csv(IN, dtype={'listing_id': str})
from concurrent.futures import ThreadPoolExecutor
batches = [d.iloc[i:i + 10] for i in range(0, len(d), 10)]
out, tin, tout, done = [], 0, 0, 0
with ThreadPoolExecutor(max_workers=6) as ex:
    for res, usage in ex.map(run_batch, batches):
        out += res; done += 1
        if usage:
            tin += usage.input_tokens; tout += usage.output_tokens
        print(f'batch {done}/{len(batches)}', flush=True)
r = pd.DataFrame(out).rename(columns={'id': 'listing_id'})
r['listing_id'] = r.listing_id.astype(str)
r['code'] = r.code.where(r.code.astype(str).str.fullmatch(r'[EHLK]SS\d{5}'), None)
r = d[['listing_id']].merge(r.drop_duplicates('listing_id'), on='listing_id', how='left')
r['llm_model'] = MODEL
r.to_csv(OUT, index=False)
print(f'\nread {r.midterm_or_non_tourist_claim.notna().sum()}/{len(d)} | tokens in {tin} out {tout}')
print(r[['midterm_or_non_tourist_claim', 'hotel_like', 'licence_text']].apply(lambda s: s.fillna(False).astype(bool).sum()).to_dict(), '| codes found:', r.code.notna().sum())
