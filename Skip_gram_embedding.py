# w2v_skipgram_embed.py
import os
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

DATA_PATH = r"C:\Users\User\OneDrive - Mongolian University of Science and Technology\Documents\vscode\NLP\texts_clean (1).csv"
ART_DIR   = r"C:/Users/User/OneDrive - Mongolian University of Science and Technology/Documents/vscode/NLP/artifacts"

os.makedirs(ART_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
texts  = df["text"].astype(str).tolist()
labels = df["label"].astype(int).values

sentences = [t.split() for t in texts]

print("[INFO] Training Word2Vec (SKIP-GRAM)...")
skip_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    sg=1,          # ✅ SKIP-GRAM
    workers=2
)

def sentence_embedding(sentence, model):
    vecs = [model.wv[w] for w in sentence.split() if w in model.wv]
    if len(vecs) == 0:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)

print("[INFO] Building sentence embeddings...")
X = np.vstack([sentence_embedding(t, skip_model) for t in texts]).astype(np.float32)

np.save(os.path.join(ART_DIR, "w2v_skipgram_embeddings.npy"), X)
np.save(os.path.join(ART_DIR, "w2v_skipgram_labels.npy"), labels)
skip_model.save(os.path.join(ART_DIR, "w2v_skipgram.model"))

print("[DONE]")
print("Embeddings:", X.shape)
