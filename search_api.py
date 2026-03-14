# search_api.py -- AiStalin Hybrid Search API v2.0.0
# v2.0: JWT auth, HttpOnly session tokens, velocity check, daily limits
# SYSTEM_INSTRUCTION, _generate(), RAG logic -- UNTOUCHED

import os, asyncio, asyncpg, uuid, datetime, secrets
import urllib.request, urllib.error, json as _json
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv
from typing import Optional
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import bcrypt as _bcrypt  # Direct bcrypt — no passlib (passlib incompatible with Python 3.13)
from jose import JWTError, jwt

load_dotenv()

DATABASE_URL    = os.getenv("DATABASE_URL")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
JWT_SECRET      = os.getenv("JWT_SECRET", "change-this-in-production")
JWT_ALGORITHM   = "HS256"
JWT_EXPIRE_DAYS = 30
EMBED_MODEL      = "models/gemini-embedding-001"
TOP_K            = 10
RRF_K            = 60
VELOCITY_SECONDS = int(os.getenv("VELOCITY_SECONDS", "5"))
FREE_DAILY_LIMIT = int(os.getenv("FREE_DAILY_LIMIT", "3"))
COOKIE_NAME      = "aistalin_session"
COOKIE_DAYS      = 365

genai.configure(api_key=GEMINI_API_KEY)
SMTP_HOST    = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT    = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER    = os.getenv("SMTP_USER", "")
SMTP_PASS    = os.getenv("SMTP_PASS", "")
NOTIFY_EMAIL = os.getenv("NOTIFY_EMAIL", "contact@aistalin.io")

# ── Password hashing — direct bcrypt (rounds=12, work factor) ────────────
def _hash_password(password: str) -> str:
    """Hash a password with bcrypt. Input validated to ≤72 bytes before call."""
    salt = _bcrypt.gensalt(rounds=12)
    return _bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

def _verify_password(plain: str, hashed: str) -> bool:
    """Constant-time bcrypt verify. Returns False on any error."""
    try:
        return _bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False
bearer_scheme = HTTPBearer(auto_error=False)

app = FastAPI(title="AiStalin Hybrid Search API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                       # ✅ works everywhere (HTTP+HTTPS)
    allow_methods=["*"],
    allow_headers=["*", "X-Session-Token"],    # allow custom session header
    expose_headers=["X-Session-Token"],        # frontend can read this from response
)
db_pool: asyncpg.Pool = None


# == STARTUP =================================================================
@app.on_event("startup")
async def startup():
    global db_pool
    db_pool = await asyncpg.create_pool(
        DATABASE_URL, min_size=2, max_size=10, command_timeout=60
    )
    print("OK DB Pool ready")
    await _create_tables()


async def _create_tables():
    stmts = [
        """CREATE TABLE IF NOT EXISTS feedback (
            id SERIAL PRIMARY KEY, name TEXT NOT NULL DEFAULT 'anon',
            message TEXT NOT NULL, ip_address TEXT DEFAULT 'Unknown',
            location TEXT DEFAULT 'Unknown', device TEXT DEFAULT '',
            time_spent_seconds INTEGER DEFAULT 0,
            is_subscribed BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )""",
        """CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY, email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL, role TEXT NOT NULL DEFAULT 'user',
            is_premium BOOLEAN NOT NULL DEFAULT FALSE,
            premium_until TIMESTAMPTZ, ip_address TEXT DEFAULT 'Unknown',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )""",
        """CREATE TABLE IF NOT EXISTS chat_history (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
            session_token TEXT NOT NULL, user_query TEXT NOT NULL,
            ai_response TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )""",
        """CREATE INDEX IF NOT EXISTS idx_chat_session
            ON chat_history(session_token, created_at DESC)""",
        """CREATE TABLE IF NOT EXISTS daily_limits (
            id SERIAL PRIMARY KEY,
            session_token TEXT NOT NULL,
            ip_address    TEXT NOT NULL,
            date          DATE NOT NULL DEFAULT CURRENT_DATE,
            query_count   INT  NOT NULL DEFAULT 1,
            UNIQUE (session_token, date)
        )""",
        """CREATE INDEX IF NOT EXISTS idx_daily_session ON daily_limits(session_token, date)""",
        """CREATE INDEX IF NOT EXISTS idx_daily_ip ON daily_limits(ip_address, date)""",
        # Password reset tokens — 1-hour expiry, single-use
        """CREATE TABLE IF NOT EXISTS password_reset_tokens (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
            token TEXT NOT NULL UNIQUE,
            expires_at TIMESTAMPTZ NOT NULL,
            used BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )""",
        """CREATE INDEX IF NOT EXISTS idx_prt_token ON password_reset_tokens(token)""",

        # User bookmarks — up to 7 for free, unlimited for premium
        """CREATE TABLE IF NOT EXISTS user_bookmarks (
            id          SERIAL PRIMARY KEY,
            user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            title       TEXT NOT NULL,
            volume_num  INTEGER,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (user_id, title, volume_num)
        )""",
        """CREATE INDEX IF NOT EXISTS idx_bk_user ON user_bookmarks(user_id)""",

        # User notes — simple text entries, date-stamped
        """CREATE TABLE IF NOT EXISTS user_notes (
            id         SERIAL PRIMARY KEY,
            user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            note_text  TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )""",
        """CREATE INDEX IF NOT EXISTS idx_notes_user ON user_notes(user_id, created_at DESC)""",
    ]
    async with db_pool.acquire() as conn:
        for s in stmts:
            await conn.execute(s)
    print("OK All tables ready")


@app.on_event("shutdown")
async def shutdown():
    await db_pool.close()


# == JWT =====================================================================
def create_jwt(uid: int, email: str, role: str, is_premium: bool) -> str:
    exp = datetime.datetime.utcnow() + datetime.timedelta(days=JWT_EXPIRE_DAYS)
    return jwt.encode({"sub": str(uid), "email": email, "role": role,
                       "is_premium": is_premium, "exp": exp},
                      JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_jwt(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError:
        return None

async def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)
) -> Optional[dict]:
    if not creds:
        return None
    return decode_jwt(creds.credentials)


# == AUTH ENDPOINTS ==========================================================
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

@app.post("/register", status_code=201)
async def register(req: RegisterRequest, request: Request):
    # ── Byte-level validation (bcrypt hard-limit = 72 bytes, not chars) ──
    # Georgian/Cyrillic chars = 2-3 bytes in UTF-8, so we MUST check bytes.
    pw_bytes = req.password.encode("utf-8")
    if len(pw_bytes) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    if len(pw_bytes) > 72:
        raise HTTPException(
            400,
            "Password too long — use Latin characters (bcrypt max = 72 bytes; Georgian chars = 3 bytes each)"
        )
    pw_hash = _hash_password(req.password)
    ip = request.headers.get("X-Forwarded-For", request.client.host or "Unknown").split(",")[0].strip()
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO users (email,password_hash,ip_address) VALUES ($1,$2,$3) RETURNING id,email,role,is_premium",
                req.email.lower(), pw_hash, ip
            )
    except asyncpg.UniqueViolationError:
        raise HTTPException(409, "Email already registered")
    return {"access_token": create_jwt(row["id"],row["email"],row["role"],row["is_premium"]),
            "token_type": "bearer"}

@app.post("/login")
async def login(req: LoginRequest):
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id,email,password_hash,role,is_premium,premium_until FROM users WHERE email=$1",
            req.email.lower()
        )
    if not row or not _verify_password(req.password, row["password_hash"]):
        raise HTTPException(401, "Invalid email or password")
    is_premium = row["is_premium"]
    if is_premium and row["premium_until"]:
        pu = row["premium_until"]
        if pu.tzinfo is None:
            pu = pu.replace(tzinfo=datetime.timezone.utc)
        if pu < datetime.datetime.now(datetime.timezone.utc):
            async with db_pool.acquire() as conn:
                await conn.execute("UPDATE users SET is_premium=FALSE WHERE id=$1", row["id"])
            is_premium = False
    return {"access_token": create_jwt(row["id"],row["email"],row["role"],is_premium),
            "token_type": "bearer", "is_premium": is_premium}


# == PASSWORD RESET ENDPOINTS ==============================================
class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

FRONTEND_URL = os.getenv("FRONTEND_URL", "https://aistalin.io")

@app.post("/forgot-password", status_code=200)
async def forgot_password(req: ForgotPasswordRequest):
    """
    Generate a reset token and send an email with a reset link.
    Always returns 200 — never reveals if the email exists (security).
    """
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, email FROM users WHERE email=$1", req.email.lower()
        )
    if not row:
        # Return success anyway — don't leak whether email is registered
        return {"status": "ok"}

    token = secrets.token_urlsafe(32)
    expires = datetime.datetime.utcnow() + datetime.timedelta(hours=1)

    async with db_pool.acquire() as conn:
        # Invalidate any previous unused tokens for this user
        await conn.execute(
            "UPDATE password_reset_tokens SET used=TRUE WHERE user_id=$1 AND used=FALSE",
            row["id"]
        )
        await conn.execute(
            "INSERT INTO password_reset_tokens (user_id, token, expires_at) VALUES ($1, $2, $3)",
            row["id"], token, expires
        )

    reset_link = f"{FRONTEND_URL}/?reset={token}"

    if SMTP_USER and SMTP_PASS:
        await asyncio.to_thread(_send_reset_email, row["email"], reset_link)

    print(f"✅ Reset token generated for {row['email']} | link: {reset_link}")
    return {"status": "ok"}


# RESEND_API_KEY env var — get free key from resend.com (3000 emails/month free)
# Railway hobby plan BLOCKS outbound SMTP (port 587/465) — HTTP API is the only option
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
RESEND_FROM    = os.getenv("RESEND_FROM", "AiStalin <noreply@aistalin.io>")

def _send_reset_email(to_email: str, reset_link: str):
    """Send password reset email via Resend HTTP API (works on Railway hobby plan).
    Railway blocks SMTP ports; HTTP API has no such restriction.
    Get free API key at resend.com → 3000 emails/month free.
    """
    if not RESEND_API_KEY:
        print(f"⚠ RESEND_API_KEY not set — reset link: {reset_link}")
        return

    html_body = f"""
<html><body style="font-family:Georgia,serif;background:#1a0c04;color:#f5e6c8;padding:24px;max-width:560px;margin:0 auto;">
  <h2 style="color:#d4a017;border-bottom:1px solid #5c2d0f;padding-bottom:8px;">
    🔑 AiStalin.io — პაროლის განახლება
  </h2>
  <p style="margin-top:16px;">გამარჯობა,</p>
  <p style="margin-top:8px;opacity:0.85;line-height:1.7;">
    მოვიდა მოთხოვნა თქვენი ანგარიშის პაროლის განახლებაზე.
    თუ ეს თქვენ გამოგზავნეთ, დააჭირეთ ღილაკს:
  </p>
  <div style="text-align:center;margin:32px 0;">
    <a href="{reset_link}"
       style="background:#d4a017;color:#1a0c04;padding:14px 32px;
              border-radius:6px;text-decoration:none;font-weight:bold;
              font-size:1rem;font-family:Georgia,serif;display:inline-block;">
      პაროლის განახლება
    </a>
  </div>
  <p style="opacity:0.65;font-size:0.85rem;line-height:1.7;">
    ბმული მოქმედებს <strong>1 საათის</strong> განმავლობაში.<br>
    თუ ეს მოთხოვნა არ გამოგზავნიათ, უბრალოდ უგულებელყავით ეს წერილი.
  </p>
  <p style="opacity:0.35;font-size:0.75rem;margin-top:24px;border-top:1px solid #5c2d0f;padding-top:12px;">
    aistalin.io — სტალინის ციფრული არქივი
  </p>
</body></html>"""

    payload = _json.dumps({
        "from":    RESEND_FROM,
        "to":      [to_email],
        "subject": "AiStalin — პაროლის განახლება",
        "html":    html_body,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.resend.com/emails",
        data=payload,
        headers={
            "Authorization": f"Bearer {RESEND_API_KEY}",
            "Content-Type":  "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = _json.loads(resp.read())
            print(f"✅ Reset email sent via Resend to {to_email} | id: {result.get('id')}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"⚠ Resend error {e.code}: {body}")
    except Exception as e:
        print(f"⚠ Reset email failed: {e}")


@app.post("/reset-password", status_code=200)
async def reset_password(req: ResetPasswordRequest):
    """
    Verify a reset token and update the user's password.
    Token is single-use and expires after 1 hour.
    """
    # Byte-level password validation (bcrypt hard limit)
    pw_bytes = req.new_password.encode("utf-8")
    if len(pw_bytes) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    if len(pw_bytes) > 72:
        raise HTTPException(400, "Password too long (max 72 bytes)")

    now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT prt.id, prt.user_id, prt.expires_at, prt.used
               FROM password_reset_tokens prt
               WHERE prt.token = $1""",
            req.token
        )

    if not row:
        raise HTTPException(400, "Invalid reset link — request a new one")
    if row["used"]:
        raise HTTPException(400, "This link has already been used")

    expires = row["expires_at"]
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=datetime.timezone.utc)
    if expires < now:
        raise HTTPException(400, "Reset link expired — request a new one")

    new_hash = _hash_password(req.new_password)

    async with db_pool.acquire() as conn:
        await conn.execute(
            "UPDATE users SET password_hash=$1 WHERE id=$2",
            new_hash, row["user_id"]
        )
        await conn.execute(
            "UPDATE password_reset_tokens SET used=TRUE WHERE id=$1",
            row["id"]
        )

    print(f"✅ Password reset for user_id={row['user_id']}")
    return {"status": "ok", "message": "Password updated successfully"}


# == BOOKMARKS ENDPOINTS ====================================================
FREE_BK_LIMIT = 7   # free users; premium = unlimited

class BookmarkIn(BaseModel):
    title:      str
    volume_num: Optional[int] = None

@app.get("/bookmarks")
async def list_bookmarks(user: dict = Depends(get_current_user)):
    if not user:
        raise HTTPException(401, "Login required")
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, title, volume_num, created_at FROM user_bookmarks "
            "WHERE user_id=$1 ORDER BY created_at DESC",
            int(user["sub"])
        )
    return {"bookmarks": [
        {"id": r["id"], "title": r["title"],
         "volume_num": r["volume_num"],
         "date": r["created_at"].strftime("%d %b %Y")}
        for r in rows
    ]}

@app.post("/bookmarks", status_code=201)
async def add_bookmark(bk: BookmarkIn, user: dict = Depends(get_current_user)):
    if not user:
        raise HTTPException(401, "Login required")
    uid = int(user["sub"])
    is_premium = user.get("is_premium", False)

    async with db_pool.acquire() as conn:
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM user_bookmarks WHERE user_id=$1", uid
        )
        if not is_premium and count >= FREE_BK_LIMIT:
            raise HTTPException(403, f"Bookmark limit reached ({FREE_BK_LIMIT}/7). Upgrade to Premium.")
        try:
            row = await conn.fetchrow(
                "INSERT INTO user_bookmarks (user_id, title, volume_num) "
                "VALUES ($1,$2,$3) RETURNING id, created_at",
                uid, bk.title.strip(), bk.volume_num
            )
        except Exception:
            raise HTTPException(409, "Already bookmarked")
    return {"id": row["id"], "date": row["created_at"].strftime("%d %b %Y")}

@app.delete("/bookmarks/{bk_id}", status_code=200)
async def delete_bookmark(bk_id: int, user: dict = Depends(get_current_user)):
    if not user:
        raise HTTPException(401, "Login required")
    async with db_pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM user_bookmarks WHERE id=$1 AND user_id=$2",
            bk_id, int(user["sub"])
        )
    if result == "DELETE 0":
        raise HTTPException(404, "Bookmark not found")
    return {"status": "deleted"}


# == NOTES ENDPOINTS =========================================================
NOTE_MAX = 30

class NoteIn(BaseModel):
    note_text: str

@app.get("/notes")
async def list_notes(user: dict = Depends(get_current_user)):
    if not user:
        raise HTTPException(401, "Login required")
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, note_text, created_at FROM user_notes "
            "WHERE user_id=$1 ORDER BY created_at DESC LIMIT $2",
            int(user["sub"]), NOTE_MAX
        )
    return {"notes": [
        {"id": r["id"], "text": r["note_text"],
         "date": r["created_at"].strftime("%d %b %Y · %H:%M")}
        for r in rows
    ]}

@app.post("/notes", status_code=201)
async def add_note(note: NoteIn, user: dict = Depends(get_current_user)):
    if not user:
        raise HTTPException(401, "Login required")
    text = note.note_text.strip()
    if not text:
        raise HTTPException(400, "Note cannot be empty")
    if len(text) > 5000:
        raise HTTPException(400, "Note too long (max 5000 chars)")
    uid = int(user["sub"])
    async with db_pool.acquire() as conn:
        # Auto-trim oldest if over limit
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM user_notes WHERE user_id=$1", uid
        )
        if count >= NOTE_MAX:
            await conn.execute(
                "DELETE FROM user_notes WHERE id = ("
                "SELECT id FROM user_notes WHERE user_id=$1 "
                "ORDER BY created_at ASC LIMIT 1)", uid
            )
        row = await conn.fetchrow(
            "INSERT INTO user_notes (user_id, note_text) VALUES ($1,$2) "
            "RETURNING id, created_at",
            uid, text
        )
    return {"id": row["id"], "date": row["created_at"].strftime("%d %b %Y · %H:%M")}

@app.delete("/notes/{note_id}", status_code=200)
async def delete_note(note_id: int, user: dict = Depends(get_current_user)):
    if not user:
        raise HTTPException(401, "Login required")
    async with db_pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM user_notes WHERE id=$1 AND user_id=$2",
            note_id, int(user["sub"])
        )
    if result == "DELETE 0":
        raise HTTPException(404, "Note not found")
    return {"status": "deleted"}


# == SEARCH MODELS ===========================================================
class SearchRequest(BaseModel):
    query: str; language: str = "en"; top_k: int = TOP_K; mode: str = "hybrid"

class ChunkResult(BaseModel):
    chunk_id: int; work_id: int; title: str; chunk_text: str
    language: str; volume_num: Optional[int]; score: float; rank: int

class SearchResponse(BaseModel):
    query: str; mode: str; results: list[ChunkResult]; total: int


# == EMBEDDING (unchanged) ===================================================
async def get_query_embedding(text: str) -> list[float]:
    def _embed():
        return genai.embed_content(model=EMBED_MODEL, content=text,
                                   task_type="retrieval_query", output_dimensionality=768)
    return (await asyncio.to_thread(_embed))["embedding"]


# == VECTOR SEARCH (unchanged) ===============================================
async def vector_search(query_embedding: list[float], language: str, top_k: int) -> list[dict]:
    embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
    sql = """
        SELECT c.id AS chunk_id, c.work_id,
            CASE WHEN $2='ka' THEN w.title_ka WHEN $2='ru' THEN w.title_ru ELSE w.title_en END AS title,
            c.chunk_text, c.language, w.volume_num,
            1 - (c.embedding <=> $1::vector) AS score
        FROM aistalin_chunks c JOIN aistalin_works w ON w.id=c.work_id
        WHERE c.language=$2 AND c.embedding IS NOT NULL
        ORDER BY c.embedding <=> $1::vector LIMIT $3
    """
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(sql, embedding_str, language, top_k * 3)
    return [dict(r) for r in rows]


# == FULL-TEXT SEARCH (unchanged) ============================================
async def fts_search(query: str, language: str, top_k: int) -> list[dict]:
    ts = "simple"
    sql = f"""
        SELECT c.id AS chunk_id, c.work_id,
            CASE WHEN $2='ka' THEN w.title_ka WHEN $2='ru' THEN w.title_ru ELSE w.title_en END AS title,
            c.chunk_text, c.language, w.volume_num,
            ts_rank_cd(c.fts_tokens, plainto_tsquery('{ts}', $1)) AS score
        FROM aistalin_chunks c JOIN aistalin_works w ON w.id=c.work_id
        WHERE c.language=$2 AND c.fts_tokens @@ plainto_tsquery('{ts}', $1)
        ORDER BY score DESC LIMIT $3
    """
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(sql, query, language, top_k * 3)
    return [dict(r) for r in rows]


# == RRF RERANKING (unchanged) ===============================================
def reciprocal_rank_fusion(vr: list[dict], fr: list[dict], top_k: int) -> list[dict]:
    scores = {}; data = {}
    for rank, r in enumerate(vr, 1):
        cid = r["chunk_id"]; scores[cid] = scores.get(cid,0) + 1.0/(RRF_K+rank); data[cid] = r
    for rank, r in enumerate(fr, 1):
        cid = r["chunk_id"]; scores[cid] = scores.get(cid,0) + 1.0/(RRF_K+rank)
        if cid not in data: data[cid] = r
    out = []
    for rank, cid in enumerate(sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k], 1):
        item = data[cid].copy(); item["score"] = round(scores[cid],6); item["rank"] = rank
        out.append(item)
    return out


# == HEALTH & SEARCH (unchanged) =============================================
@app.get("/health")
async def health_check():
    async with db_pool.acquire() as conn:
        count = await conn.fetchval("SELECT COUNT(*) FROM aistalin_chunks WHERE embedding IS NOT NULL")
    return {"status": "ok", "version": "2.0.0", "chunks_ready": count}

@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    if not req.query.strip(): raise HTTPException(400, "Query cannot be empty")
    if req.language not in ["en","ka","ru"]: raise HTTPException(400, "language must be en/ka/ru")
    if req.mode not in ["hybrid","vector","fts"]: raise HTTPException(400, "invalid mode")
    if req.mode == "vector":
        emb = await get_query_embedding(req.query)
        raw = await vector_search(emb, req.language, req.top_k)
        results = [{**r,"score":round(r["score"],6),"rank":i+1} for i,r in enumerate(raw[:req.top_k])]
    elif req.mode == "fts":
        raw = await fts_search(req.query, req.language, req.top_k)
        results = [{**r,"rank":i+1} for i,r in enumerate(raw[:req.top_k])]
    else:
        emb = await get_query_embedding(req.query)
        v,f = await asyncio.gather(vector_search(emb,req.language,req.top_k),
                                   fts_search(req.query,req.language,req.top_k))
        results = reciprocal_rank_fusion(v, f, req.top_k)
    return SearchResponse(query=req.query, mode=req.mode, results=results, total=len(results))


# == CHAT MODELS ============================================================
class ChatRequest(BaseModel):
    query: str
    language: str = 'en'
    top_k: int = 5

class SourceItem(BaseModel):
    chunk_id: int; work_id: int; title: str
    volume_num: Optional[int]; score: float

class ChatResponse(BaseModel):
    query: str; answer: str; sources: list[SourceItem]


# == SYSTEM INSTRUCTION (unchanged) =========================================
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
"""

CHAT_MODEL = "models/gemini-2.5-flash"

# == _generate (unchanged) =================================================
def _generate(prompt: str):
    model = genai.GenerativeModel(
        model_name=CHAT_MODEL,
        system_instruction=SYSTEM_INSTRUCTION,
        generation_config=genai.types.GenerationConfig(max_output_tokens=8192, temperature=0.25)
    )
    return model.generate_content(prompt)


# == CHAT ENDPOINT v2 ========================================================
NO_INFO_PHRASES = [
    "not found in the archive", "archive contains no records", "not found in archive",
    "archived scope",
    "არქივში ეს ინფორმაცია არ იძებნება",
    "არ იძებნება არქივში",
    "არ არის მოცემული არქივში",
    "ინფორმაცია არ მოიძებნა",
    "არ მოიპოვება",
    "ვერ მოიძებნა",
    "нет в архиве",
    "не найдена",
    "не найдено",
]


@app.post("/chat")
async def chat(
    req:          ChatRequest,
    request:      Request,
    response:     Response,
    current_user: Optional[dict] = Depends(get_current_user),
):
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty")
    if req.language not in ["en", "ka", "ru"]:
        raise HTTPException(400, "language must be en/ka/ru")

    # 1. Session token — header takes priority (works on HTTP+HTTPS)
    #    Frontend stores it in localStorage and sends as X-Session-Token
    session_token = (
        request.headers.get("X-Session-Token") or   # from localStorage
        request.cookies.get(COOKIE_NAME) or          # from HttpOnly cookie (HTTPS)
        None
    )
    is_new_session = not bool(session_token)
    if is_new_session:
        session_token = str(uuid.uuid4())

    # 2. Client IP
    client_ip = request.headers.get("X-Forwarded-For", request.client.host or "Unknown").split(",")[0].strip()

    # 3. Velocity check
    async with db_pool.acquire() as conn:
        last_ts = await conn.fetchval(
            "SELECT created_at FROM chat_history WHERE session_token=$1 ORDER BY created_at DESC LIMIT 1",
            session_token
        )
    if last_ts:
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        if not last_ts.tzinfo:
            last_ts = last_ts.replace(tzinfo=datetime.timezone.utc)
        elapsed = (now_utc - last_ts).total_seconds()
        if elapsed < VELOCITY_SECONDS:
            raise HTTPException(429, f"Too fast. Wait {int(VELOCITY_SECONDS - elapsed) + 1}s.")

    # 4. Premium / Guest
    is_premium = bool(current_user and current_user.get("is_premium"))
    user_id    = int(current_user["sub"]) if current_user else None

    def _attach_session(resp, req=request):
        # Always return session token in header (works HTTP + HTTPS)
        resp.headers["X-Session-Token"] = session_token
        resp.headers["Access-Control-Expose-Headers"] = "X-Session-Token"
        # Also set cookie — secure only on HTTPS (auto-detected)
        is_https = req.url.scheme == "https"
        if is_new_session:
            resp.set_cookie(
                key=COOKIE_NAME, value=session_token,
                httponly=True,
                secure=is_https,                    # ✅ False on HTTP (dev), True on HTTPS (prod)
                samesite="none" if is_https else "lax",
                max_age=60 * 60 * 24 * COOKIE_DAYS
            )
        return resp

    if not is_premium:
        today = datetime.date.today()
        async with db_pool.acquire() as conn:
            # session count — exact match on session_token+date
            ses_cnt = await conn.fetchval(
                "SELECT query_count FROM daily_limits WHERE session_token=$1 AND date=$2",
                session_token, today) or 0
            # IP count — sum across ALL sessions from this IP today
            ip_cnt = await conn.fetchval(
                "SELECT COALESCE(SUM(query_count),0) FROM daily_limits WHERE ip_address=$1 AND date=$2",
                client_ip, today) or 0
        if ses_cnt >= FREE_DAILY_LIMIT or ip_cnt >= FREE_DAILY_LIMIT:
            return _attach_session(JSONResponse(
                content={"status": "limit_reached", "message": "Daily free limit reached. Please subscribe."},
                status_code=200
            ))

    # 5. RAG retrieval
    emb = await get_query_embedding(req.query)
    vr, fr = await asyncio.gather(
        vector_search(emb, req.language, req.top_k),
        fts_search(req.query, req.language, req.top_k)
    )
    chunks = reciprocal_rank_fusion(vr, fr, req.top_k)

    if not chunks:
        answer = "არქივში ამ კითხვაზე შესაბამისი ინფორმაცია ვერ მოიძებნა."
    else:
        context = "\n\n---\n\n".join(
            f"[{i}] სათაური: {c['title']} | ტომი: {c.get('volume_num','?')}\n{c['chunk_text']}"
            for i, c in enumerate(chunks, 1)
        )
        prompt = (
            "Context (სტალინის ტექსტებიდან):\n\n" + context +
            "\n\n---\n\nმომხმარებლის კითხვა: " + req.query +
            "\n\nგთხოვ უპასუხო კითხვას მხოლოდ ზემოთ მოწოდებული Context-ის საფუძვლზე."
        )
        gen = await asyncio.to_thread(_generate, prompt)
        answer = gen.text.strip()

    info_not_found = any(p.lower() in answer.lower() for p in NO_INFO_PHRASES)
    sources = [] if info_not_found else [
        SourceItem(chunk_id=c["chunk_id"], work_id=c["work_id"], title=c["title"],
                   volume_num=c.get("volume_num"), score=c["score"])
        for c in chunks
    ]

    # Save history
    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO chat_history (user_id,session_token,user_query,ai_response) VALUES ($1,$2,$3,$4)",
            user_id, session_token, req.query, answer
        )

    # Increment daily limits — single row per (session_token, date)
    # IP is stored in the same row; SUM(query_count) WHERE ip_address=X gives total per IP
    if not is_premium:
        today = datetime.date.today()
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO daily_limits (session_token, ip_address, date, query_count)
                VALUES ($1, $2, $3, 1)
                ON CONFLICT (session_token, date)
                DO UPDATE SET query_count = daily_limits.query_count + 1
            """, session_token, client_ip, today)

    data = ChatResponse(query=req.query, answer=answer, sources=sources)
    return _attach_session(JSONResponse(content=data.dict()))


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
