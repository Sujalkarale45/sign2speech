import pandas as pd
from pathlib import Path

df = pd.read_csv('data/raw/metadata/how2sign_val.csv', sep='\t')
print('Total clips:', len(df))

video_dir = Path('data/raw/videos/val')
found = 0
missing = 0
for _, row in df.iterrows():
    name = str(row['SENTENCE_NAME']).strip()
    p    = video_dir / (name + '.mp4')
    if p.exists():
        found += 1
    else:
        missing += 1

print('Videos found  :', found)
print('Videos missing:', missing)

print()
print('Sample SENTENCE_NAME:', df['SENTENCE_NAME'].iloc[0])

videos = list(video_dir.glob('*.mp4'))
if videos:
    print('Sample video file   :', videos[0].name)
else:
    print('No mp4 files in val folder')
