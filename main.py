#!/usr/bin/env python3
import logging
from pathlib import Path
from typing import Any
from dataclasses import dataclass
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from livekit.agents import (
    JobContext,
    WorkerOptions,
    cli,
    RoomInputOptions,
    Agent,
    AgentSession,
    function_tool,
    ChatMessage,
)
from livekit.plugins import deepgram, silero, cartesia, openai

# Load .env variables
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rag-agent")

# ---- Configuration ----
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
QDRANT_COLLECTION_NAME = "livekit_docs"
QDRANT_URL = "http://localhost:6333"

# ---- Agent ----


@dataclass
class QueryResult:
    id: str
    payload: dict[str, Any]
    score: float


class RAGEnrichedAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
                You are a helpful voice assistant for farmdar, able to answer user questions.
                Keep answers casual, TTS-friendly, and avoid long formalities or markdown.
                When searching documentation, You have access to only one tool: livekit_docs_search. Do not invent or call any other tool.
                Make sure you act like you're talking to a real user in production.
                If the user asks anything unrelated to farmdar, politely refuse.
                Example: If user asks What is the population of Pakistan? say: I can assist you with farmdar related queries only.
                - If the user says a product name, feature, or term like 'AgriChain', 'CropScan', or 'NDVI', you should assume they want to search the documentation, even if they don't say 'quote' or 'search'.
                - Trigger the livekit_docs_search tool automatically when a term sounds like a product or technical concept.
                - Avoid asking users to say 'quote' or 'search for' — infer their intent from phrasing and term capitalization.
                - Answer in a casual, human-like tone,
            """
        )
        self._embedding_model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
        self._client = QdrantClient(url=QDRANT_URL)
        self._seen_results = set()

        self._initialize_qdrant()

    def _initialize_qdrant(self) -> None:
        if not self._client.collection_exists(collection_name=QDRANT_COLLECTION_NAME):
            self._client.recreate_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
        logger.info("Qdrant collection is ready.")

    def _generate_embedding(self, text: str) -> list[float]:
        return self._embedding_model.encode(text, convert_to_tensor=False).tolist()

    @function_tool(name="livekit_docs_search")
    async def livekit_docs_search(self, query: str) -> str:
        try:
            query_vector = self._generate_embedding(query)
            results = self._client.search(
                collection_name=QDRANT_COLLECTION_NAME,
                query_vector=query_vector,
                limit=5
            )

            new_results = [
                QueryResult(id=str(res.id), payload=res.payload,
                            score=res.score)
                for res in results if res.payload and res.id not in self._seen_results
            ]

            if not new_results:
                return "No new results found."

            self._seen_results.update(r.id for r in new_results)
            new_results = new_results[:2]

            context_parts = []
            for result in new_results:
                paragraph = result.payload.get("text", "")
                source = result.payload.get("source", "Unknown source")
                context_parts.append(
                    f"Source: {source}\nContent: {paragraph}\n")

            return "\n\n".join(context_parts) if context_parts else "No usable paragraph found."

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return "Could not find any relevant information for that query."

    async def on_message(self, message: ChatMessage):
        search_result = await self.livekit_docs_search(query=message.text)

        if "Content:" in search_result:
            reply = await self.session.generate_reply(
                instructions=self.instructions,
                text=f"User asked: {message.text}\n\nRelevant farmdar Documentation:\n{search_result}"
            )
            await self.session.say(reply)
        else:
            await self.session.say("Sorry, I can't answer that. My knowledge is limited to farmdar documentation.")

    async def on_enter(self):
        await self.session.say("Hi! I can help with farmdar questions. What would you like to know?")


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
            temperature=0.7,

        ),
        tts=cartesia.TTS(
            model="sonic-english",
            voice="39b376fc-488e-4d0c-8b37-e00b72059fdd",
            speed=0.4,
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
