from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings


DEFAULT_EMBED_MODEL = os.environ.get("OPENAI_EMBEDDINGS", "text-embedding-3-small")
DEFAULT_CHAT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


@dataclass
class RAGConfig:
    persist_dir: str = "chroma_db"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    k: int = 4
    chat_model: str = DEFAULT_CHAT_MODEL
    embed_model: str = DEFAULT_EMBED_MODEL


def _embeddings(model: str):
    if model.startswith("local:"):
        # Allow "local:all-MiniLM-L6-v2" or "local:sentence-transformers/all-MiniLM-L6-v2"
        name = model.split("local:", 1)[1].strip() or "sentence-transformers/all-MiniLM-L6-v2"
        if name == "all-MiniLM-L6-v2":
            name = "sentence-transformers/all-MiniLM-L6-v2"
        return HuggingFaceEmbeddings(model_name=name)
    # OpenAI path:
    return OpenAIEmbeddings(model=model)


def build_vector_store(
    texts_with_meta: List[Tuple[str, Dict[str, Any]]],
    cfg: RAGConfig
) -> Chroma:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap
    )
    documents: List[Document] = []
    for text, meta in texts_with_meta:
        if not text:
            continue
        for chunk in splitter.split_text(text):
            documents.append(Document(page_content=chunk, metadata=meta))

    vs = Chroma.from_documents(
        documents=documents,
        embedding=_embeddings(cfg.embed_model),
        persist_directory=cfg.persist_dir,
    )
    vs.persist()
    return vs


def load_vector_store(cfg: RAGConfig) -> Chroma:
    return Chroma(
        embedding_function=_embeddings(cfg.embed_model),
        persist_directory=cfg.persist_dir,
    )


def retrieve(vs: Chroma, query: str, k: int) -> List[Document]:
    retriever = vs.as_retriever(search_kwargs={"k": k})
    return retriever.get_relevant_documents(query)


def generate_answer(query: str, contexts: List[Document], cfg: RAGConfig) -> str:
    llm = ChatOpenAI(model=cfg.chat_model, temperature=0)

    sources_lines = []
    for i, d in enumerate(contexts, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        label = f"[{i}] {src}" + (f" (p.{page})" if page else "")
        sources_lines.append(label)

    context_block = "\n\n".join(
        f"[{i+1}] {doc.page_content}" for i, doc in enumerate(contexts)
    )

    system = (
        "use ONLY the provided context to answer. "
        "If the answer is not in the context, say you don't know. Cite like [1], [2]."
    )
    prompt = f"""SYSTEM:
{system}

CONTEXT:
{context_block}

QUESTION:
{query}

ANSWER (with citations):"""
    resp = llm.invoke(prompt)
    content = getattr(resp, "content", str(resp))
    if sources_lines:
        content += "\n\nSources:\n" + "\n".join(sources_lines)
    return content
