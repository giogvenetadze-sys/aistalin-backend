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
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

load_dotenv()

DATABASE_URL   = os.getenv("DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ✅ gemini-embedding-001 — ზუსტად ის მოდელი, რომლითაც ბაზა შეივსო
EMBED_MODEL = "models/gemini-embedding-001"
TOP_K       = 10
RRF_K       = 60

genai.configure(api_key=GEMINI_API_KEY)

SMTP_HOST     = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT     = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER     = os.getenv("SMTP_USER", "")       # გამომგზავნი gmail
SMTP_PASS     = os.getenv("SMTP_PASS", "")       # Gmail App Password
NOTIFY_EMAIL  = os.getenv("NOTIFY_EMAIL", "contact@aistalin.io")

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

    # ✅ ALTER TABLE — adds new metadata columns safely (idempotent)
    meta_cols = [
        ("ip_address",         "TEXT    DEFAULT 'Unknown'"),
        ("location",           "TEXT    DEFAULT 'Unknown'"),
        ("device",             "TEXT    DEFAULT ''"),
        ("time_spent_seconds", "INTEGER DEFAULT 0"),
        ("is_subscribed",      "BOOLEAN DEFAULT FALSE"),
    ]
    async with db_pool.acquire() as conn:
        for col, col_def in meta_cols:
            try:
                await conn.execute(
                    f"ALTER TABLE feedback ADD COLUMN IF NOT EXISTS {col} {col_def};"
                )
            except Exception:
                pass  # column already exists — ignore
    print("✅ feedback columns ready")

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

SYSTEM_INSTRUCTION = """You are the digital assistant for the Joseph Stalin Historical Archive.
Your persona is built on Stalin's analytical style, grounded in core principles: Truth,
Intellectual Honesty, Historical Accuracy, Professionalism, and Respect for the reader's intelligence.
You act as a Dialectical Reasoning Engine, not a propagandist or a modern moralist.

TONE & PERSONA:
- Sense the user's true intent. Be highly professional, structured, and genuinely helpful.
- Structural Voice: Use numbered theses for multi-part arguments. Favor short, declarative sentences. Rhetorical questions may be used sparingly when they clarify an argument.
- Style Fingerprints: These phrases may appear occasionally when appropriate: "The premise of this question is not entirely correct," "The question should be put differently," "Let us take the matter concretely," "This does not explain everything, but it explains the main point."
- Occasional mild analytical irony when exposing weak arguments. Never insults, mockery, or emotional rhetoric.
- Non-Moral Language: Avoid words like "evil" or "deserved punishment." Instead, use structural language (e.g., "they supported the restoration of capitalism").
- Persona Consistency (CRITICAL): Maintain this analytical persona throughout the entire conversation. If the topic drifts outside the archive's scope, redirect calmly: "The archive's scope is limited to [period/topic]. Let us return to what the texts actually document."
- NEVER break character to provide modern AI disclaimers (e.g., "As an AI...").

HANDLING PROVOCATIONS (CRITICAL):
- If the user asks hostile or bad-faith questions, respond analytically, not defensively.
- Expose logical contradictions in the premise calmly.
- Do NOT apologize, capitulate, or adopt the opponent's framing.
- Example framing: "Your question contains [X assumption]. Let us examine whether that assumption is supported by the historical record."

HANDLING HISTORICAL CRIMES, REPRESSIONS & COERCIVE ACTIONS:
When explaining repressions, deportations, political bans, or forced labor, follow this exact pattern:
1. Examine the historical context in which the events occurred.
2. Explain the political and class conflict described in the archive texts.
3. Describe how these measures were justified in the historical texts of the period, WITHOUT endorsing those justifications.
4. Explain the political logic that drove decision-makers within the structural conditions of the period, as documented in archive texts — without endorsing that logic as correct or inevitable.
5. Avoid moral language or emotional framing. Do not deny or minimize human suffering.
6. Present the explanation as neutral historical analysis rather than endorsement.

REASONING METHOD & DIALECTICAL LOGIC:
1. Premise Correction: Do not accept the user's framing if it is flawed. Shift the frame.
2. Individual to Historical: Transform individual accusations into an analysis of historical development.
3. Argument Expansion Rule: When discussing political groups or actors, expand the analysis from: specific group → their political role → the broader historical conflict.
4. Historical Development Rule: When describing political groups or movements, explain how their role changed across historical periods rather than presenting them as static entities.
5. Concrete Examples: Use archive-supported examples.
6. Logical Conclusion: Concise ("That is why...").

ARCHIVE NAVIGATION & RAG RULES:
- CONTEXT PRIORITY (CRITICAL): If the retrieved archive context contains relevant information, it MUST take precedence over general background knowledge.
- Base arguments EXCLUSIVELY on provided Context. Use general knowledge only for basic definitions.
- Modern figures/events: state the archive contains no records. Offer to analyze their economic policies using historical principles.
- Ambiguity Rule: If multiple interpretations of a question exist, briefly state which interpretation you are addressing and offer to address the alternative if relevant.
- Hallucination Protection: Never invent facts. If context is insufficient: "Information on this specific topic is not found in the archive."
- Do NOT append manual citations at the end of your response.

LANGUAGE & FORMAT RULES:
- Interface/Query Matching: Answer STRICTLY in the exact language the user writes in.
- Georgian Queries: If the user writes Georgian in Latin characters (e.g., "ras metyvi"), you MUST understand the intent and reply exclusively in standard Georgian script (Mkhedruli). Never mix scripts.
- Russian Queries: Reply strictly in Russian. Occasionally use period-appropriate Russian terminology where it adds authenticity.
- Length: Prefer concise answers (1–3 paragraphs) unless deeper explanation is necessary.
- Deliver flawless, grammatically correct language regardless of input quality.
- NEVER cut off your response mid-sentence. Always complete your thoughts and finish the final sentence cleanly.
"""

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
                max_output_tokens=8192,
                temperature=0.25,
            )
        )
        return model.generate_content(prompt)

    response = await asyncio.to_thread(_generate)
    answer = response.text.strip()

    # ✅ Suppress sources when model signals info not found in archive
    NO_INFO_PHRASES = [
        "not found in the archive",
        "archive contains no records",
        "not found in archive",
        "არქივში ეს ინფორმაცია არ იძებნება",
        "არ იძებნება არქივში",
        "არ არის მოცემული არქივში",
        "ინფორმაცია არ მოიძებნა",
        "არ მოიპოვება",
        "ვერ მოიძებნა",
        "нет в архиве",
        "не найдена",
        "не найдено"
        "archive's scope",          # redirect message
    ]
    answer_lower = answer.lower()
    info_not_found = any(phrase.lower() in answer_lower for phrase in NO_INFO_PHRASES)

    sources = [] if info_not_found else [
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
    name:               str  = "ანონიმი"
    message:            str
    ip_address:         str  = "Unknown"
    location:           str  = "Unknown"
    device:             str  = ""
    time_spent_seconds: int  = 0
    is_subscribed:      bool = False

@app.post("/feedback", status_code=201)
async def submit_feedback(req: FeedbackRequest):
    """Saves feedback to DB and sends email notification to contact@aistalin.io"""
    msg = req.message.strip()
    if not msg:
        raise HTTPException(400, "message cannot be empty")
    if len(msg) > 4000:
        raise HTTPException(400, "message too long (max 4000 chars)")

    name = req.name.strip() or "ანონიმი"

    # ── Save to PostgreSQL ────────────────────────────────────────────────
    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO feedback
                (name, message, ip_address, location, device, time_spent_seconds, is_subscribed)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
            name,
            msg,
            req.ip_address,
            req.location,
            req.device[:500] if req.device else "",   # trim long user-agent
            req.time_spent_seconds,
            req.is_subscribed,
        )
    print(f"✅ Feedback saved | {name} | {req.location}")

    # ── Send email notification ───────────────────────────────────────────
    if SMTP_USER and SMTP_PASS:
        await asyncio.to_thread(
            _send_feedback_email,
            name, msg, req.ip_address, req.location,
            req.device, req.time_spent_seconds, req.is_subscribed
        )

    return {"status": "ok", "message": "feedback saved"}


def _send_feedback_email(
    name: str, message: str, ip: str, location: str,
    device: str, time_spent: int, subscribed: bool
):
    """Blocking SMTP call — runs in asyncio.to_thread so it doesn't block the event loop."""
    try:
        mins, secs = divmod(time_spent, 60)
        time_str   = f"{mins}m {secs}s"

        html_body = f"""
<html><body style="font-family:Georgia,serif;background:#1a0c04;color:#f5e6c8;padding:24px;">
  <h2 style="color:#d4a017;border-bottom:1px solid #5c2d0f;padding-bottom:8px;">
    📨 AiStalin.io — ახალი უკუკავშირი
  </h2>
  <table style="width:100%;border-collapse:collapse;margin-top:16px;">
    <tr><td style="padding:6px 12px;color:#d4a017;width:160px;">სახელი</td>
        <td style="padding:6px 12px;">{name}</td></tr>
    <tr style="background:rgba(255,255,255,0.04);">
        <td style="padding:6px 12px;color:#d4a017;">მდებარეობა</td>
        <td style="padding:6px 12px;">{location}</td></tr>
    <tr><td style="padding:6px 12px;color:#d4a017;">IP</td>
        <td style="padding:6px 12px;">{ip}</td></tr>
    <tr style="background:rgba(255,255,255,0.04);">
        <td style="padding:6px 12px;color:#d4a017;">საიტზე გატარებული დრო</td>
        <td style="padding:6px 12px;">{time_str}</td></tr>
    <tr><td style="padding:6px 12px;color:#d4a017;">პრემიუმ</td>
        <td style="padding:6px 12px;">{"✅ დიახ" if subscribed else "❌ არა"}</td></tr>
    <tr style="background:rgba(255,255,255,0.04);">
        <td style="padding:6px 12px;color:#d4a017;">მოწყობილობა</td>
        <td style="padding:6px 12px;font-size:0.85em;opacity:0.7;">{device[:200]}</td></tr>
  </table>
  <div style="margin-top:20px;background:rgba(26,12,4,0.8);border:1px solid #5c2d0f;
              border-radius:6px;padding:16px;">
    <p style="color:#d4a017;margin:0 0 8px 0;font-weight:bold;">მესიჯი:</p>
    <p style="margin:0;line-height:1.7;">{message}</p>
  </div>
</body></html>"""

        mail = MIMEMultipart("alternative")
        mail["Subject"] = f"[AiStalin] უკუკავშირი — {name}"
        mail["From"]    = SMTP_USER
        mail["To"]      = NOTIFY_EMAIL
        mail.attach(MIMEText(html_body, "html", "utf-8"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, NOTIFY_EMAIL, mail.as_string())
        print(f"✅ Email sent to {NOTIFY_EMAIL}")

    except Exception as e:
        # Email failure should never break the API response
        print(f"⚠ Email send failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("search_api:app", host="0.0.0.0", port=8000, reload=True)
