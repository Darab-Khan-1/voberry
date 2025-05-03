import pickle
import uuid
import logging
from pathlib import Path
from typing import List, Optional, Union, Literal, Callable, Any
from collections.abc import Iterable
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import annoy

from livekit.agents import tokenize

logger = logging.getLogger("rag-builder")

# RAG Index Types and Classes
Metric = Literal["angular", "euclidean", "manhattan", "hamming", "dot"]
ANNOY_FILE = "index.annoy"
METADATA_FILE = "metadata.pkl"

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
        index_path = p / ANNOY_FILE
        metadata_path = p / METADATA_FILE

        with open(metadata_path, "rb") as f:
            metadata: _FileData = pickle.load(f)

        index = annoy.AnnoyIndex(metadata.f, metadata.metric)
        index.load(str(index_path))
        return cls(index, metadata)

    @property
    def size(self) -> int:
        return self._index.get_n_items()

    def items(self) -> Iterable[Item]:
        for i in range(self._index.get_n_items()):
            item = Item(
                i=i,
                userdata=self._filedata.userdata[i],
                vector=self._index.get_item_vector(i),
            )
            yield item

    def query(
        self, vector: list[float], n: int, search_k: int = -1
    ) -> list[QueryResult]:
        ids = self._index.get_nns_by_vector(
            vector, n, search_k=search_k, include_distances=True
        )
        return [
            QueryResult(userdata=self._filedata.userdata[i], distance=distance)
            for i, distance in zip(*ids)
        ]

class IndexBuilder:
    def __init__(self, f: int, metric: Metric) -> None:
        self._index = annoy.AnnoyIndex(f, metric)
        self._filedata = _FileData(f=f, metric=metric, userdata={})
        self._i = 0

    def save(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        index_path = p / ANNOY_FILE
        metadata_path = p / METADATA_FILE
        self._index.save(str(index_path))
        with open(metadata_path, "wb") as f:
            pickle.dump(self._filedata, f)

    def build(self, trees: int = 50, jobs: int = -1) -> AnnoyIndex:
        self._index.build(n_trees=trees, n_jobs=jobs)
        return AnnoyIndex(self._index, self._filedata)

    def add_item(self, vector: list[float], userdata: Any) -> None:
        self._index.add_item(self._i, vector)
        self._filedata.userdata[self._i] = userdata
        self._i += 1

class SentenceChunker:
    def __init__(
        self,
        *,
        max_chunk_size: int = 120,
        chunk_overlap: int = 30,
        paragraph_tokenizer: Callable[
            [str], list[str]
        ] = tokenize.basic.tokenize_paragraphs,
        sentence_tokenizer: tokenize.SentenceTokenizer = tokenize.basic.SentenceTokenizer(),
        word_tokenizer: tokenize.WordTokenizer = tokenize.basic.WordTokenizer(
            ignore_punctuation=False
        ),
    ) -> None:
        self._max_chunk_size = max_chunk_size
        self._chunk_overlap = chunk_overlap
        self._paragraph_tokenizer = paragraph_tokenizer
        self._sentence_tokenizer = sentence_tokenizer
        self._word_tokenizer = word_tokenizer

    def chunk(self, *, text: str) -> list[str]:
        chunks = []
        buf_words: list[str] = []
        for paragraph in self._paragraph_tokenizer(text):
            last_buf_words: list[str] = []

            for sentence in self._sentence_tokenizer.tokenize(text=paragraph):
                for word in self._word_tokenizer.tokenize(text=sentence):
                    reconstructed = self._word_tokenizer.format_words(
                        buf_words + [word]
                    )

                    if len(reconstructed) > self._max_chunk_size:
                        while (
                            len(self._word_tokenizer.format_words(last_buf_words))
                            > self._chunk_overlap
                        ):
                            last_buf_words = last_buf_words[1:]

                        new_chunk = self._word_tokenizer.format_words(
                            last_buf_words + buf_words
                        )
                        chunks.append(new_chunk)
                        last_buf_words = buf_words
                        buf_words = []

                    buf_words.append(word)

            if buf_words:
                while (
                    len(self._word_tokenizer.format_words(last_buf_words))
                    > self._chunk_overlap
                ):
                    last_buf_words = last_buf_words[1:]

                new_chunk = self._word_tokenizer.format_words(
                    last_buf_words + buf_words
                )
                chunks.append(new_chunk)
                buf_words = []

        return chunks

class RAGBuilder:
    def __init__(
        self,
        index_path: Union[str, Path],
        data_path: Union[str, Path],
        embeddings_model_name: str = "all-MiniLM-L6-v2",
        metric: str = "angular",
        device: str = "cpu",
        batch_size: int = 32,
    ):
        self._index_path = Path(index_path)
        self._data_path = Path(data_path)
        self._model = SentenceTransformer(embeddings_model_name, device=device)
        self._embeddings_dimension = self._model.get_sentence_embedding_dimension()
        self._metric = metric
        self._batch_size = batch_size

    def _clean_content(self, text: str) -> str:
        lines = text.split('\n')
        cleaned_lines = []
        
        skip_patterns = [
            'Docs', 'Search', 'GitHub', 'Slack', 'Sign in',
            'Home', 'AI Agents', 'Telephony', 'Recipes', 'Reference',
            'On this page', 'Get started with LiveKit today',
            'Content from https://docs.livekit.io/'
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if any(pattern in line for pattern in skip_patterns):
                continue
                
            if line.startswith('http') or line.startswith('[') or line.endswith(']'):
                continue
                
            cleaned_lines.append(line)
            
        return '\n'.join(cleaned_lines)

    def _create_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, convert_to_tensor=False).tolist()

    async def build_from_texts(self, texts: List[str], show_progress: bool = True) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._data_path.parent.mkdir(parents=True, exist_ok=True)

        idx_builder = IndexBuilder(f=self._embeddings_dimension, metric=self._metric)

        cleaned_texts = []
        for text in texts:
            cleaned = self._clean_content(text)
            if cleaned:
                cleaned_texts.append(cleaned)

        paragraphs_by_uuid = {str(uuid.uuid4()): text for text in cleaned_texts}
        texts_to_embed = list(paragraphs_by_uuid.values())

        if show_progress:
            pbar = tqdm(total=len(texts_to_embed), desc="Creating embeddings")

        for i in range(0, len(texts_to_embed), self._batch_size):
            batch = texts_to_embed[i:i + self._batch_size]
            embeddings = self._create_embeddings(batch)
            
            for j in range(len(batch)):
                p_uuid = list(paragraphs_by_uuid.keys())[i + j]
                idx_builder.add_item(embeddings[j], p_uuid)
                
                if show_progress:
                    pbar.update(1)

        if show_progress:
            pbar.close()

        logger.info(f"Building index at {self._index_path}")
        idx_builder.build()
        idx_builder.save(str(self._index_path))

        logger.info(f"Saving paragraph data to {self._data_path}")
        with open(self._data_path, "wb") as f:
            pickle.dump(paragraphs_by_uuid, f)

    async def build_from_file(self, file_path: Union[str, Path], show_progress: bool = True) -> None:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        with open(file_path, "r") as f:
            raw_data = f.read()

        paragraphs = tokenize.basic.tokenize_paragraphs(raw_data)
        await self.build_from_texts(paragraphs, show_progress)

    @classmethod
    async def create_from_file(
        cls,
        file_path: Union[str, Path],
        index_path: Union[str, Path],
        data_path: Union[str, Path],
        **kwargs,
    ) -> "RAGBuilder":
        builder = cls(index_path, data_path, **kwargs)
        await builder.build_from_file(file_path)
        return builder