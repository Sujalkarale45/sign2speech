from pathlib import Path
import json

videos = sorted(Path('data/raw/wlasl-processed/videos').glob('*.mp4'))
print('Actual filenames on disk:')
for v in videos[:10]:
    print('  ' + v.name)

with open('data/raw/wlasl-processed/WLASL_v0.3.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print('\nJSON expects these video_ids (first 10):')
for entry in data[:5]:
    for inst in entry['instances'][:2]:
        print(f"  {str(inst['video_id']).zfill(5)}.mp4  gloss={entry['gloss']}")
