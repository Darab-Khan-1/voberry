import pickle
import uuid
from pathlib import Path
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex

# ---- Settings ----
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
METRIC = "angular"
NUM_TREES = 10

VDB_DIR = Path("data")
VDB_DIR.mkdir(exist_ok=True)
PARAGRAPHS_FILE = VDB_DIR / "paragraphs.pkl"
INDEX_FILE = VDB_DIR / "index.annoy"
METADATA_FILE = VDB_DIR / "metadata.pkl"

# ---- Example paragraphs ----
raw_paragraphs = [
    "LiveKit enables real-time video, audio, and data streaming for developers.",
    "Rooms in LiveKit are virtual spaces where participants can join and communicate.",
    "Tracks are media sources, such as audio or video, published by participants.",
    "Simulcast in LiveKit allows streaming at multiple resolutions for better adaptability.",
    "LiveKit's server SDK provides APIs for managing rooms, participants, and tracks."
]

# ---- Step 1: Embed paragraphs ----
model = SentenceTransformer(EMBEDDING_MODEL)
embeddings = model.encode(raw_paragraphs, convert_to_tensor=False)

# ---- Step 2: Create UUID mapping ----
uuid_map = {str(uuid.uuid4()): para for para in raw_paragraphs}

# ---- Step 3: Create Annoy index ----
embedding_dim = len(embeddings[0])  # Use a distinct name to avoid confusion
index = AnnoyIndex(embedding_dim, metric=METRIC)
userdata_map = {}

for i, (uid, embedding) in enumerate(zip(uuid_map.keys(), embeddings)):
    index.add_item(i, embedding)
    userdata_map[i] = uid

index.build(NUM_TREES)
index.save(str(INDEX_FILE))

# ---- Step 4: Save paragraph UUID mapping ----
with open(PARAGRAPHS_FILE, "wb") as para_file:
    pickle.dump(uuid_map, para_file)

# ---- Step 5: Save metadata ----
metadata = {
    "f": embedding_dim,
    "metric": METRIC,
    "userdata": userdata_map
}

with open(METADATA_FILE, "wb") as meta_file:
    pickle.dump(metadata, meta_file)

print("✅ index.annoy, paragraphs.pkl, and metadata.pkl generated in ./data/")
