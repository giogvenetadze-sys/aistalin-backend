# search_api.py — AiStalin Hybrid Search API v1.3.0
# ✅ gemini-embedding-001 (768-dim) — matches embed_generator.py
# ✅ asyncio.to_thread — embed_content სინქრონული, event loop არ იბლოკება
# ✅ FTS 'simple' config — fts_tokens 'simple'-ით არის generated
# ✅ title ენის მიხედვით — title_ka / title_ru / title_en
# ✅ mode validation — hybrid / vector / fts

import os
import asyncio
import asyncpg
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

DATABASE_URL   = os.getenv("DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ✅ gemini-embedding-001 — ზუსტად ის მოდელი, რომლითაც ბაზა შეივსო
EMBED_MODEL = "models/gemini-embedding-001"
TOP_K       = 10
RRF_K       = 60

genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="AiStalin Hybrid Search API", version="1.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db_pool: asyncpg.Pool = None

@app.on_event("startup")
async def startup():
    global db_pool
    db_pool = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=2,
        max_size=10,
        command_timeout=60   # ✅ production: query 60s-ზე მეტს არ ელოდება
    )
    print("✅ DB Pool ready")

    # ✅ feedback ცხრილი — პირველ გაშვებაზე ავტომატურად შეიქმნება
    async with db_pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id         SERIAL PRIMARY KEY,
                name       TEXT         NOT NULL DEFAULT 'ანონიმი',
                message    TEXT         NOT NULL,
                created_at TIMESTAMPTZ  NOT NULL DEFAULT NOW()
            );
        """)
    print("✅ feedback table ready")

@app.on_event("shutdown")
async def shutdown():
    await db_pool.close()

# ── Request / Response Models ────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    language: str = "en"      # "en", "ka", "ru"
    top_k: int = TOP_K
    mode: str = "hybrid"      # "hybrid", "vector", "fts"

class ChunkResult(BaseModel):
    chunk_id: int
    work_id: int
    title: str
    chunk_text: str
    language: str
    volume_num: Optional[int]
    score: float
    rank: int

class SearchResponse(BaseModel):
    query: str
    mode: str
    results: list[ChunkResult]
    total: int

# ── Embedding ────────────────────────────────────────────────────────────
async def get_query_embedding(text: str) -> list[float]:
    """
    ✅ asyncio.to_thread — genai.embed_content სინქრონულია.
    thread-ში გაშვება event loop-ს არ ბლოკავს.
    სხვა requests ელოდებენ რომ embed_content დასრულდეს —
    to_thread ამ ლოდინს background thread-ში გადააქვს.
    """
    def _embed():
        return genai.embed_content(
            model=EMBED_MODEL,
            content=text,
            task_type="retrieval_query",
            output_dimensionality=768   # ✅ ბაზა 768-dim
        )
    result = await asyncio.to_thread(_embed)
    return result["embedding"]

# ── Vector Search ────────────────────────────────────────────────────────
async def vector_search(
    query_embedding: list[float],
    language: str,
    top_k: int
) -> list[dict]:
    """
    pgvector cosine distance search.
    <=> = cosine distance | 1 - distance = similarity score
    """
    embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

    sql = """
        SELECT
            c.id         AS chunk_id,
            c.work_id    AS work_id,
            CASE
                WHEN $2 = 'ka' THEN w.title_ka
                WHEN $2 = 'ru' THEN w.title_ru
                ELSE                w.title_en
            END          AS title,
            c.chunk_text,
            c.language,
            w.volume_num,
            1 - (c.embedding <=> $1::vector) AS score
        FROM aistalin_chunks c
        JOIN aistalin_works  w ON w.id = c.work_id
        WHERE c.language    = $2
          AND c.embedding  IS NOT NULL
        ORDER BY c.embedding <=> $1::vector
        LIMIT $3
    """

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(sql, embedding_str, language, top_k * 3)  # ✅ *3 → RRF-ს მეტი კანდიდატი

    return [dict(r) for r in rows]

# ── Full-Text Search ─────────────────────────────────────────────────────
async def fts_search(query: str, language: str, top_k: int) -> list[dict]:
    """
    ✅ ts_config = 'simple' — კრიტიკულია!
    fts_tokens სვეტი to_tsvector('simple', ...) -ით არის generated.
    თუ query-ში 'english' ან 'russian' გამოვიყენებთ —
    ლექსემები სხვანაირად დაითვლება, match ვერ მოხდება, 0 შედეგი.
    """
    ts_config = "simple"  # ✅ ყოველთვის simple — ყველა ენისთვის

    sql = f"""
        SELECT
            c.id         AS chunk_id,
            c.work_id    AS work_id,
            CASE
                WHEN $2 = 'ka' THEN w.title_ka
                WHEN $2 = 'ru' THEN w.title_ru
                ELSE                w.title_en
            END          AS title,
            c.chunk_text,
            c.language,
            w.volume_num,
            ts_rank_cd(
                c.fts_tokens,
                plainto_tsquery('{ts_config}', $1)
            ) AS score
        FROM aistalin_chunks c
        JOIN aistalin_works  w ON w.id = c.work_id
        WHERE c.language = $2
          AND c.fts_tokens @@ plainto_tsquery('{ts_config}', $1)
        ORDER BY score DESC
        LIMIT $3
    """

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(sql, query, language, top_k * 3)  # ✅ *3 → RRF-ს მეტი კანდიდატი

    return [dict(r) for r in rows]

# ── RRF Reranking ────────────────────────────────────────────────────────
def reciprocal_rank_fusion(
    vector_results: list[dict],
    fts_results: list[dict],
    top_k: int
) -> list[dict]:
    """
    RRF Score = 1/(60 + rank_vector) + 1/(60 + rank_fts)
    ორივე სიაში მაღლა — საბოლოო სიაში მაღლა.
    """
    scores     = {}
    chunk_data = {}

    for rank, r in enumerate(vector_results, 1):
        cid             = r["chunk_id"]
        scores[cid]     = scores.get(cid, 0) + 1.0 / (RRF_K + rank)
        chunk_data[cid] = r

    for rank, r in enumerate(fts_results, 1):
        cid         = r["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (RRF_K + rank)
        if cid not in chunk_data:
            chunk_data[cid] = r

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)

    results = []
    for rank, cid in enumerate(sorted_ids[:top_k], 1):
        item          = chunk_data[cid].copy()
        item["score"] = round(scores[cid], 6)
        item["rank"]  = rank
        results.append(item)

    return results

# ── Endpoints ────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    async with db_pool.acquire() as conn:
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM aistalin_chunks WHERE embedding IS NOT NULL"
        )
    return {
        "status":       "ok",
        "version":      "1.3.0",
        "chunks_ready": count
    }

@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    # ✅ Validation
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty")
    if req.language not in ["en", "ka", "ru"]:
        raise HTTPException(400, "language must be 'en', 'ka', or 'ru'")
    if req.mode not in ["hybrid", "vector", "fts"]:
        raise HTTPException(400, "mode must be 'hybrid', 'vector', or 'fts'")

    results = []

    if req.mode == "vector":
        emb  = await get_query_embedding(req.query)
        raw  = await vector_search(emb, req.language, req.top_k)
        results = [
            {**r, "score": round(r["score"], 6), "rank": i + 1}
            for i, r in enumerate(raw[:req.top_k])
        ]

    elif req.mode == "fts":
        raw  = await fts_search(req.query, req.language, req.top_k)
        results = [
            {**r, "rank": i + 1}
            for i, r in enumerate(raw[:req.top_k])
        ]

    else:  # hybrid — vector + FTS პარალელურად
        emb = await get_query_embedding(req.query)
        vector_res, fts_res = await asyncio.gather(
            vector_search(emb, req.language, req.top_k),
            fts_search(req.query, req.language, req.top_k)
        )
        results = reciprocal_rank_fusion(vector_res, fts_res, req.top_k)

    return SearchResponse(
        query=req.query,
        mode=req.mode,
        results=results,
        total=len(results)
    )


# ── Chat Models ──────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    language: str = "en"
    top_k: int = 5

class SourceItem(BaseModel):
    chunk_id: int
    work_id: int
    title: str
    volume_num: Optional[int]
    score: float

class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceItem]

SYSTEM_INSTRUCTION = """შენ ხარ სტალინის ისტორიული არქივის ციფრული ასისტენტი.
უპასუხე მომხმარებლის კითხვას მკაცრად მხოლოდ მოწოდებულ Context-ზე დაყრდნობით.
იყავი მოკლე და კონკრეტული (მაქსიმუმ 3-4 აბზაცი).
არ გამოიყენო გარე ცოდნა.
თუ Context-ში პასუხი არ არის, უთხარი რომ არქივში ეს ინფორმაცია არ იძებნება.
პასუხის ბოლოს მიუთითე გამოყენებული წყაროები (მაგ: ტომი, სათაური)."""

CHAT_MODEL = "models/gemini-2.5-flash"

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty")
    if req.language not in ["en", "ka", "ru"]:
        raise HTTPException(400, "language must be 'en', 'ka', or 'ru'")

    emb = await get_query_embedding(req.query)
    vector_res, fts_res = await asyncio.gather(
        vector_search(emb, req.language, req.top_k),
        fts_search(req.query, req.language, req.top_k)
    )
    chunks = reciprocal_rank_fusion(vector_res, fts_res, req.top_k)

    if not chunks:
        return ChatResponse(
            query=req.query,
            answer="არქივში ამ კითხვაზე შესაბამისი ინფორმაცია ვერ მოიძებნა.",
            sources=[]
        )

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[{i}] სათაური: {chunk['title']} | ტომი: {chunk.get('volume_num', '?')}\n"
            f"{chunk['chunk_text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""Context (სტალინის ტექსტებიდან):\n\n{context}\n\n---\n\nმომხმარებლის კითხვა: {req.query}\n\nგთხოვ უპასუხო კითხვას მხოლოდ ზემოთ მოწოდებული Context-ის საფუძველზე."""

    def _generate():
        model = genai.GenerativeModel(
            model_name=CHAT_MODEL,
            system_instruction=SYSTEM_INSTRUCTION,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=2000,
                temperature=0.2,
            )
        )
        return model.generate_content(prompt)

    response = await asyncio.to_thread(_generate)
    answer = response.text.strip()

    sources = [
        SourceItem(
            chunk_id=c["chunk_id"],
            work_id=c["work_id"],
            title=c["title"],
            volume_num=c.get("volume_num"),
            score=c["score"]
        )
        for c in chunks
    ]

    return ChatResponse(query=req.query, answer=answer, sources=sources)


# ── Volume Endpoint ──────────────────────────────────────────────────────
@app.get("/volume/{volume_num}")
async def get_volume_content(volume_num: int, language: str = "ka"):
    """
    Returns full content of a volume: all chunks grouped by article title.
    language: "ka" (default) | "en" | "ru"
    Used by the frontend reader modal.
    """
    if language not in ["en", "ka", "ru"]:
        raise HTTPException(400, "language must be 'en', 'ka', or 'ru'")

    sql = """
        SELECT
            CASE
                WHEN $2 = 'ka' THEN w.title_ka
                WHEN $2 = 'ru' THEN w.title_ru
                ELSE                w.title_en
            END          AS title,
            c.chunk_text,
            c.id         AS chunk_id
        FROM aistalin_chunks c
        JOIN aistalin_works  w ON w.id = c.work_id
        WHERE w.volume_num = $1
          AND c.language   = $2
        ORDER BY c.id ASC
    """

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(sql, volume_num, language)

    if not rows:
        return {"volume": volume_num, "chapters": [], "error": "Volume not found or not yet loaded"}

    chapters = [{"title": r["title"], "chunk_text": r["chunk_text"]} for r in rows]
    return {"volume": volume_num, "language": language, "chapters": chapters}


# ══════════════════════════════════════════════════════════════
#  FEEDBACK ENDPOINT
#  POST /feedback  — saves user message to feedback table
# ══════════════════════════════════════════════════════════════

class FeedbackRequest(BaseModel):
    name: str = "ანონიმი"
    message: str

@app.post("/feedback", status_code=201)
async def submit_feedback(req: FeedbackRequest):
    msg = req.message.strip()
    if not msg:
        raise HTTPException(400, "message cannot be empty")
    if len(msg) > 4000:
        raise HTTPException(400, "message too long (max 4000 chars)")
    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO feedback (name, message) VALUES ($1, $2)",
            req.name.strip() or "ანონიმი",
            msg
        )
    return {"status": "ok", "message": "feedback saved"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("search_api:app", host="0.0.0.0", port=8000, reload=True)
