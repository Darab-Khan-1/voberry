import uuid
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# ---- Settings ----
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "livekit_docs"
RAW_TEXT_FILE = Path("data/raw_data.txt")

# ---- Load and clean paragraphs ----
if not RAW_TEXT_FILE.exists():
    raise FileNotFoundError(f"{RAW_TEXT_FILE} not found.")

with open(RAW_TEXT_FILE, "r", encoding="utf-8") as f:
    full_text = f.read()

paragraphs = [p.strip()
              for p in full_text.split("\n\n") if len(p.strip()) > 50]
if not paragraphs:
    raise ValueError("No valid paragraphs found.")

# ---- Load embedding model ----
model = SentenceTransformer(EMBEDDING_MODEL)

# ---- Encode paragraphs ----
embeddings = model.encode(paragraphs, convert_to_tensor=False)

# ---- Connect to Qdrant (assumes local Qdrant is running) ----
qdrant = QdrantClient(host="localhost", port=6333)

# ---- Create/reset collection ----
qdrant.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=len(embeddings[0]),
        distance=Distance.COSINE
    )
)

# ---- Prepare and upload data ----
points = []
for paragraph, vector in zip(paragraphs, embeddings):
    uid = str(uuid.uuid4())
    points.append(
        PointStruct(
            id=uid,
            vector=vector,
            payload={
                "text": paragraph,
                "source": "raw_data.txt"  # optional metadata
            }
        )
    )

qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

print(f"✅ Uploaded {len(points)} paragraphs to Qdrant.")
