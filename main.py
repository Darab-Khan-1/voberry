#!/usr/bin/env python3
import logging
import pickle
from pathlib import Path
from typing import Literal, Any, Optional
from collections.abc import Iterable
from dataclasses import dataclass
from dotenv import load_dotenv
import annoy
from sentence_transformers import SentenceTransformer  # Added for embeddings

from livekit.agents import (
    JobContext,
    WorkerOptions,
    cli,
    RunContext,
    RoomInputOptions,
    Agent,
    AgentSession,
    function_tool,
    ChatMessage,
)
from livekit.plugins import deepgram, silero, cartesia, openai
from livekit.plugins.turn_detector.english import EnglishModel

# Load .env variables
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rag-agent")

# ---- Configuration ----
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VDB_DIR = Path(__file__).parent / "data"
DATA_FILE = "paragraphs.pkl"
INDEX_FILE = "index.annoy"

# ---- Annoy + Metadata ----
Metric = Literal["angular", "euclidean", "manhattan", "hamming", "dot"]

@dataclass
class _FileData:
    f: int
    metric: Metric
    userdata: dict[int, Any]

@dataclass
class Item:
    i: int
    userdata: Any
    vector: list[float]

@dataclass
class QueryResult:
    userdata: Any
    distance: float

class AnnoyIndex:
    def __init__(self, index: annoy.AnnoyIndex, filedata: _FileData) -> None:
        self._index = index
        self._filedata = filedata

    @classmethod
    def load(cls, path: str) -> "AnnoyIndex":
        p = Path(path)
        metadata_path = p / "metadata.pkl"

        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        index = annoy.AnnoyIndex(metadata["f"], metadata["metric"])
        index.load(str(p / "index.annoy"))

        filedata = _FileData(f=metadata["f"], metric=metadata["metric"], userdata=metadata["userdata"])
        return cls(index, filedata)

    def query(self, vector: list[float], n: int, search_k: int = -1) -> list[QueryResult]:
        ids = self._index.get_nns_by_vector(
            vector, n, search_k=search_k, include_distances=True
        )
        return [
            QueryResult(userdata=self._filedata.userdata[i], distance=distance)
            for i, distance in zip(*ids)
        ]

# ---- Agent ----
class RAGEnrichedAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
                 You are a helpful voice assistant for given website provided to you, able to answer user questions.
                Keep answers casual, TTS-friendly, and avoid long formalities or markdown.
                When searching documentation, use the given website provided to you docs_search function.
                make sure that you act like you are talking with actual user in production.
                If the user asks anythingother than given website provided to you, apologize him and tell them to ask about given website provided to you only.
                examples: user asks "What is the population of pakistan" you will respond "I can assist you with given website provided to you related queries only. Please avoid asking irrelevant question.
                If the response generates an eror 404 or any other error, just say that "Cannot answer the question as this is inconsistant with the knowledge base. Don't return the error.
            """
        )
        self._embedding_model = None
        self._annoy_index = None
        self._paragraphs_by_uuid = {}
        self._seen_results = set()
        
        self._initialize_rag()

    def _initialize_rag(self) -> bool:
        """Initialize RAG components, return success status"""
        if not VDB_DIR.exists():
            logger.error(f"RAG database directory not found: {VDB_DIR}")
            return False

        data_path = VDB_DIR / DATA_FILE
        if not data_path.exists():
            logger.error(f"Paragraph data not found: {data_path}")
            return False

        try:
            # Initialize embedding model
            self._embedding_model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
            
            # Load vector index
            self._annoy_index = AnnoyIndex.load(str(VDB_DIR))
            
            # Load paragraph data
            with open(data_path, "rb") as f:
                self._paragraphs_by_uuid = pickle.load(f)
                
            logger.info("RAG system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG: {e}")
            return False

    def _generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for query text"""
        if not self._embedding_model:
            raise RuntimeError("Embedding model not initialized")
        return self._embedding_model.encode(text, convert_to_tensor=False).tolist()

    @function_tool(name="livekit_docs_search")
    async def livekit_docs_search(self, query: str) -> str:  # Changed signature
        """Search knowledge base for information regarding the user query using the given query.
        
        Args:
            query: The search query to look up in retrieved documentation
        """
        try:
            if not self._annoy_index or not self._paragraphs_by_uuid:
                return "My knowledge base is not currently available."

            # Generate proper embedding for the query
            query_vector = self._generate_embedding(query)

            all_results = self._annoy_index.query(query_vector, n=5)
            new_results = [r for r in all_results if r.userdata not in self._seen_results]

            if not new_results:
                return "No new results found."
            new_results = new_results[:2]

            context_parts = []
            for result in new_results:
                self._seen_results.add(result.userdata)
                paragraph = self._paragraphs_by_uuid.get(result.userdata, "")
                if paragraph:
                    source = "Unknown source"
                    if "from [" in paragraph:
                        source = paragraph.split("from [")[1].split("]")[0]
                        paragraph = paragraph.split("]")[1].strip()
                    context_parts.append(f"Source: {source}\nContent: {paragraph}\n")

            return "\n\n".join(context_parts) if context_parts else "No usable paragraph found."

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return "Could not find any relevant information for that query."

    async def on_message(self, message: ChatMessage):
        search_result = await self.livekit_docs_search(query=message.text)
        logger.error(f"")
        # If result contains content, respond
        if "Content:" in search_result:
            await self.session.send_text(search_result)
        else:
            # Otherwise, block unrelated answers
            await self.session.say("Sorry, I can't answer that. My knowledge is limited to context provided to me via documentation.")
            # await super().on_message(message)

    async def on_enter(self):
        if self._annoy_index:
            greeting = "Hi! I can help you today?"
        else:
            greeting = "Hello! I'm having trouble accessing my knowledge base, but I can still try to help."
        await self.session.say(greeting)

        #await self.session.generate_reply(instructions=greeting)

# ---- Entrypoint ----
async def entrypoint(ctx: JobContext):
    await ctx.connect()
    
    session = AgentSession(
        stt=deepgram.STT(
            model="nova-2",
            language="en-US"
        ),
        llm=openai.LLM.with_ollama(
            model="llama3.2:1b",
            base_url="http://localhost:11434/v1",
            temperature=0.8
        ),
        tts=cartesia.TTS(
            model="sonic-english",
            voice="39b376fc-488e-4d0c-8b37-e00b72059fdd",
            speed=0.5,
            emotion=["curiosity:highest", "positivity:high"]
        ),
        vad=silero.VAD.load()
    )

    await session.start(
        agent=RAGEnrichedAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions()
    )

# ---- Main ----
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))