import os
from pathlib import Path
from typing import Optional, Dict, Any, List

import requests
import numpy as np
import torch
import chromadb
from chromadb.config import Settings

from fastapi import FastAPI
from pydantic import BaseModel
from FlagEmbedding import FlagReranker


# ---- Config ----
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "index/chroma"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_docs_ollama")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:7b-instruct")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

TOPK_RECALL = int(os.getenv("TOPK_RECALL", "12"))
TOPK_CONTEXT = int(os.getenv("TOPK_CONTEXT", "4"))


app = FastAPI(title="Agent Service (Ollama embeddings + Ollama LLM)")


class AskRequest(BaseModel):
    question: str
    where: Optional[Dict[str, Any]] = None
    topk_context: Optional[int] = None


class AskResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]


def ollama_embed(text: str) -> List[float]:
    url = f"{OLLAMA_HOST}/api/embeddings"
    payload = {"model": OLLAMA_EMBED_MODEL, "prompt": text}
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    emb = data.get("embedding", None)
    if emb is None:
        raise RuntimeError(f"Ollama embeddings response missing 'embedding': {data}")
    return emb


def ollama_generate(prompt: str) -> str:
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": OLLAMA_LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "top_p": 0.9}
    }
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return (data.get("response", "") or "").strip()


def build_prompt(question: str, contexts: List[str]) -> str:
    ctx = "\n\n".join([f"[证据{i+1}]\n{c}" for i, c in enumerate(contexts)])
    return f"""你是一个严谨的知识库问答助手。只能根据给定证据回答，禁止凭空编造。
如果证据不足以回答，请明确说“证据不足”，并说明缺少什么信息。
回答必须引用证据编号，例如：引用：[1][3]。

问题：{question}

证据：
{ctx}

请用中文作答：
"""


def doc_to_context(doc: str, md: Dict[str, Any]) -> str:
    src = md.get("pdf_name") or md.get("source") or md.get("file") or ""
    page = md.get("page", "")
    dtype = md.get("type", "")
    chunk = md.get("chunk_idx", "")
    img = md.get("image_path", "")

    header = f"(source={src}, page={page}, type={dtype}, chunk={chunk})"
    if img:
        header += f"\n(image_path={img})"
    return header + "\n" + doc


# ---- Chroma + reranker (loaded once) ----
client = None
col = None
reranker = None


@app.on_event("startup")
def startup():
    global client, col, reranker

    if not CHROMA_DIR.exists():
        raise RuntimeError(f"Chroma dir not found: {CHROMA_DIR}")

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False)
    )
    col = client.get_collection(CHROMA_COLLECTION)

    use_fp16 = torch.cuda.is_available()
    reranker = FlagReranker(RERANK_MODEL, use_fp16=use_fp16)

    # preflight check
    _ = ollama_embed("ping")
    _ = ollama_generate("只回答OK")


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    q = req.question.strip()
    if not q:
        return {"answer": "问题为空。", "citations": []}

    # 1) query embedding via Ollama
    q_emb = np.asarray(ollama_embed(q), dtype=np.float32).tolist()

    # 2) Chroma query (native)
    res = col.query(
        query_embeddings=[q_emb],
        n_results=TOPK_RECALL,
        where=req.where,
        include=["documents", "metadatas", "distances"]
    )

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    if not docs:
        return {"answer": "没有召回到任何证据。", "citations": []}

    candidates = [doc_to_context(d, m or {}) for d, m in zip(docs, metas)]

    # 3) rerank
    pairs = [[q, c] for c in candidates]
    scores = reranker.compute_score(pairs)
    ranked = sorted(zip(scores, candidates, metas), key=lambda x: x[0], reverse=True)

    k_ctx = req.topk_context or TOPK_CONTEXT
    top = ranked[:k_ctx]
    contexts = [t[1] for t in top]

    # 4) generate via Ollama
    prompt = build_prompt(q, contexts)
    answer = ollama_generate(prompt)

    citations = []
    for i, (score, _ctx, md) in enumerate(top, 1):
        md = md or {}
        citations.append({
            "evidence_id": i,
            "score": float(score),
            "pdf_name": md.get("pdf_name") or md.get("source") or "",
            "page": md.get("page", -1),
            "type": md.get("type", ""),
            "chunk_idx": md.get("chunk_idx", -1),
            "image_path": md.get("image_path", ""),
        })

    return {"answer": answer, "citations": citations}
