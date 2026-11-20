"""Hybrid retrieval: dense (bge-small) + BM25 per query -> RRF -> cross-encoder rerank -> MMR."""
from __future__ import annotations

import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from . import config

log = logging.getLogger("retrieval")

STOP = set("""a an the and or of to in on for with at by from is are was were be been it its this that
these those as what which who whom how do does did can could would should i you we they he she me my
your our their about any some there here than then so if not no yes please tell show give find""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


@dataclass
class Chunk:
    rowid: int
    source: str
    kind: str
    url: str
    title: str
    text: str


class KnowledgeBase:
    def __init__(self):
        self.ready = False
        self._lock = threading.Lock()

    def load(self):
        from fastembed import TextEmbedding
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        t = time.time()
        con = sqlite3.connect(config.DATA_DIR / "kb.db", check_same_thread=False)
        rows = con.execute("SELECT rowid, source, kind, url, title, text FROM chunks ORDER BY rowid").fetchall()
        self.chunks = [Chunk(*r) for r in rows]
        self.vecs = np.load(config.DATA_DIR / "vectors.npy").astype(np.float32)
        assert len(self.vecs) == len(self.chunks), "vectors.npy out of sync with kb.db"
        self.bm25 = BM25Okapi([tokenize(f"{c.title or ''} {c.text}") for c in self.chunks])
        self.embedder = TextEmbedding(config.EMBED_MODEL, cache_dir=str(config.MODEL_DIR),
                                      threads=config.ONNX_THREADS)
        self.reranker = TextCrossEncoder(config.RERANK_MODEL, cache_dir=str(config.MODEL_DIR),
                                         threads=config.ONNX_THREADS)
        self.db = con
        self.ready = True
        log.info("KB loaded: %d chunks in %.1fs", len(self.chunks), time.time() - t)

    # ---------------------------------------------------------------- stages
    def _embed_queries(self, queries: list[str]) -> np.ndarray:
        q = np.array(list(self.embedder.query_embed(queries)), dtype=np.float32)
        return q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)

    def _allowed(self, sources: set[str] | None) -> np.ndarray | None:
        if not sources:
            return None
        return np.array([c.source in sources for c in self.chunks])

    def hybrid(self, queries: list[str], sources: set[str] | None = None) -> list[list[int]]:
        """One ranked list per (query, retriever)."""
        k = config.RETRIEVE_PER_QUERY
        mask = self._allowed(sources)
        lists = []
        qv = self._embed_queries(queries)
        dense = qv @ self.vecs.T
        for i, q in enumerate(queries):
            d = dense[i].copy()
            if mask is not None:
                d[~mask] = -1
            lists.append(list(np.argsort(-d)[:k]))
            s = self.bm25.get_scores(tokenize(q))
            if mask is not None:
                s[~mask] = -1
            top = np.argsort(-s)[:k]
            lists.append([j for j in top if s[j] > 0])
        return lists

    @staticmethod
    def rrf(lists: list[list[int]], k: int = 60) -> list[int]:
        score: dict[int, float] = {}
        for lst in lists:
            for rank, idx in enumerate(lst):
                score[idx] = score.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return sorted(score, key=score.get, reverse=True)

    def rerank(self, query: str, idxs: list[int]) -> list[tuple[int, float]]:
        docs = [f"{self.chunks[i].title or ''}. {self.chunks[i].text}"[:1200] for i in idxs]
        scores = list(self.reranker.rerank(query, docs, batch_size=4))  # small batches keep RAM low
        return sorted(zip(idxs, scores), key=lambda x: x[1], reverse=True)

    def mmr(self, ranked: list[tuple[int, float]], n: int) -> list[tuple[int, float]]:
        """Maximal marginal relevance over reranker scores, plus a per-URL cap."""
        if not ranked:
            return []
        rel = np.array([s for _, s in ranked], dtype=np.float32)
        rel = (rel - rel.min()) / (rel.max() - rel.min() + 1e-9)
        V = self.vecs[[i for i, _ in ranked]]
        chosen: list[int] = []
        per_url: dict[str, int] = {}
        while len(chosen) < min(n, len(ranked)):
            best, best_val = None, -1e9
            for j in range(len(ranked)):
                if j in chosen:
                    continue
                url = self.chunks[ranked[j][0]].url
                if per_url.get(url, 0) >= config.MAX_PER_URL:
                    continue
                sim = max((float(V[j] @ V[c]) for c in chosen), default=0.0)
                if sim > 0.92:          # near-duplicate of something already chosen
                    continue
                val = config.MMR_LAMBDA * rel[j] - (1 - config.MMR_LAMBDA) * sim
                if val > best_val:
                    best, best_val = j, val
            if best is None:
                break
            chosen.append(best)
            url = self.chunks[ranked[best][0]].url
            per_url[url] = per_url.get(url, 0) + 1
        # back-fill: when one page dominates (a project's FAQs), the URL cap can leave
        # slots empty - fill them with the next-best non-duplicate chunks.
        for j in range(len(ranked)):
            if len(chosen) >= n:
                break
            if j in chosen:
                continue
            if all(float(V[j] @ V[c]) <= 0.92 for c in chosen):
                chosen.append(j)
        return [ranked[j] for j in chosen]

    # ---------------------------------------------------------------- public
    def search(self, question: str, variants: list[str], sources: set[str] | None = None) -> dict:
        t0 = time.time()
        queries = [question] + [v for v in variants if v and v.lower() != question.lower()][:3]
        with self._lock:
            lists = self.hybrid(queries, sources)
            fused = self.rrf(lists)[: config.FUSED_KEEP]
            ranked = self.rerank(question, fused)[: config.RERANK_KEEP]
            final = self.mmr(ranked, config.FINAL_CHUNKS)
        hits = [{"chunk": self.chunks[i], "score": float(s)} for i, s in final]
        return {"hits": hits, "trace": {
            "queries": queries, "raw_hits": sum(len(l) for l in lists),
            "fused": len(fused), "reranked": len(ranked), "final": len(hits),
            "ms": int((time.time() - t0) * 1000)}}


KB = KnowledgeBase()
