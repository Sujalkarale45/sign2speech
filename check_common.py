import pandas as pd
import json

df = pd.read_csv('D:/signvoice/data/raw/asl-signs/train.csv')
google_signs = set(df['sign'].unique())

with open('D:/signvoice/data/raw/wlasl-processed/WLASL_v0.3.json') as f:
    wlasl = json.load(f)
wlasl_signs = set(e['gloss'].lower() for e in wlasl)

common = google_signs.intersection(wlasl_signs)

results = []
for s in common:
    g_count = len(df[df['sign'] == s])
    w_count = next(
        (len(e['instances']) for e in wlasl if e['gloss'].lower() == s), 0
    )
    results.append((s, g_count, w_count, g_count + w_count))

results.sort(key=lambda x: x[3], reverse=True)

print('Total common signs:', len(common))
print()
print('Top 20 by combined sample count:')
print('Sign                 | Google | WLASL | Total')
print('-' * 50)
for s, g, w, t in results[:20]:
    print(f'{s:<20} | {g:>6} | {w:>5} | {t:>5}')
