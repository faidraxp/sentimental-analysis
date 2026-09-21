import numpy as np 
import pandas as pd

from pathlib import Path
import soundfile as sf
import librosa
dataset_path = Path(
    r"C:\Users\faidr\Documents\sentimental_analysis\Acted Emotional Speech Dynamic Database"
)

X = []
y = []
filenames = []
bad_files = list()
for emotion_dir in dataset_path.iterdir():
    if not emotion_dir.is_dir():
        continue

    emotion = emotion_dir.name

    for wav_path in emotion_dir.glob("*.wav"):
        try:
            audio, sample_rate = sf.read(wav_path, dtype="float32")
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sample_rate,
                n_mfcc=13
            )

            # Average each MFCC over time
            mfcc_mean = np.mean(mfcc, axis=1)

            X.append(mfcc_mean)
            y.append(emotion)
            filenames.append(wav_path.name)

        except sf.LibsndfileError as e:
            print(f"Skipping: {wav_path}")
            print(f"Reason: {e}")
            bad_files.append(wav_path)

columns = [f"mfcc_{i}" for i in range(1, 14)]

df = pd.DataFrame(X, columns=columns)
df["emotion"] = y
df["filename"] = filenames
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("dataset.csv", index=False)
