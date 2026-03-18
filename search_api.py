# search_api.py -- AiStalin Hybrid Search API v2.0.0
# v2.0: JWT auth, HttpOnly session tokens, velocity check, daily limits
# SYSTEM_INSTRUCTION, _generate(), RAG logic -- UNTOUCHED

import os, asyncio, asyncpg, uuid, datetime, secrets, hashlib, hmac, time
from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests
import urllib.request, urllib.error, json as _json
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
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
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")  # Get from Google Cloud Console → APIs & Services → Credentials

# ── Paddle configuration ─────────────────────────────────────────────────────
# Paddle dashboard → Developer Tools → Notifications → Notification settings
PADDLE_WEBHOOK_SECRET = os.getenv("PADDLE_WEBHOOK_SECRET", "")  # Notification secret key
PADDLE_PRICE_ID       = os.getenv("PADDLE_PRICE_ID", "")        # pri_xxxx — your $5/mo price
PADDLE_ENV            = os.getenv("PADDLE_ENV", "sandbox")       # "sandbox" | "production"

# ── Admin access ─────────────────────────────────────────────────────────────
# Only this email can access future admin endpoints (e.g. /admin/*)
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "")  # Set in Railway Variables

# In-memory store for short-lived admin dashboard tokens
# { token_str: { "email": str, "expires": float } }
# Tokens expire in 60 seconds — single use for redirect only
_admin_dash_tokens: dict = {}
JWT_ALGORITHM   = "HS256"
JWT_EXPIRE_DAYS = 30
EMBED_MODEL      = "models/gemini-embedding-001"
TOP_K            = 10
RRF_K            = 60
VELOCITY_SECONDS = int(os.getenv("VELOCITY_SECONDS", "5"))
FREE_DAILY_LIMIT = int(os.getenv("FREE_DAILY_LIMIT", "3"))

# ── Chat RAG constants (single source of truth) ───────────────────────────
# Change these two lines only — no other edits needed to tune chat quality.
CHAT_TOP_K          = 4   # fused chunks passed to generation (was 5)
CHAT_SOURCE_LIMIT   = 3   # max sources returned in response
PREMIUM_MEMORY_DAYS = 10  # how far back to look for premium chat history
PREMIUM_MEMORY_TURNS = 5  # max turns injected directly into prompt

# Minimum RRF score for a chunk to be considered "genuinely relevant".
# RRF scores range roughly 0.01–0.033. Below this threshold the retrieval
# matched something weakly (e.g. greetings, meta-questions) and sources
# should not be shown even if chunks were returned.
RRF_SOURCE_THRESHOLD = 0.016
COOKIE_NAME      = "aistalin_session"
COOKIE_DAYS      = 365

# ── Query cache (in-memory, per process) ─────────────────────────────────
# Caches final (answer, sources) for identical query+language pairs.
# TTL: 6 hours. Max: 256 entries (LRU-eviction by insertion order).
# Does NOT cache "not found" answers — those can change as corpus grows.
import re as _re
from collections import OrderedDict as _OD

CACHE_TTL_SEC  = 6 * 3600   # 6 hours
CACHE_MAX_SIZE = 256

_query_cache: _OD = _OD()   # key → {"answer": str, "sources": list, "ts": float}

def _cache_key(query: str, language: str) -> str:
    """Normalise query to maximise cache hits for near-identical inputs."""
    return language + ":" + _re.sub(r"\s+", " ", query.strip().lower())

def _cache_get(key: str):
    entry = _query_cache.get(key)
    if not entry:
        return None
    if time.time() - entry["ts"] > CACHE_TTL_SEC:
        _query_cache.pop(key, None)
        return None
    # Move to end (LRU touch)
    _query_cache.move_to_end(key)
    return entry

def _cache_set(key: str, answer: str, sources: list):
    if key in _query_cache:
        _query_cache.move_to_end(key)
    _query_cache[key] = {"answer": answer, "sources": sources, "ts": time.time()}
    if len(_query_cache) > CACHE_MAX_SIZE:
        _query_cache.popitem(last=False)   # evict oldest

# ── Trivial / greeting query detector ────────────────────────────────────
# Queries below MIN_SUBSTANTIVE_CHARS or matching GREETING_RE bypass RAG
# entirely — no embedding call, no DB hit, no sources returned.
MIN_SUBSTANTIVE_CHARS = 15

_GREETINGS = {
    # Georgian
    "სალამი","გამარჯობა","მოგესალმებით","გამარჯობათ","მარჯობა",
    "გამარჯვება","ჰეი","ჰი","სალამ",
    # English
    "hi","hello","hey","greetings","good morning","good evening",
    "good afternoon","good day","howdy","yo","sup","what's up","whats up",
    # Russian
    "привет","здравствуйте","здравствуй","добрый день","добрый вечер",
    "доброе утро","добро пожаловать","хай","приветствую",
}

_GREETING_PATTERN = _re.compile(
    r"^\s*(" + "|".join(_re.escape(g) for g in _GREETINGS) + r")[!?.\s]*$",
    _re.IGNORECASE | _re.UNICODE
)

_GREETING_REPLIES = {
    "ka": "მოგესალმებით. რით შემიძლია დაგეხმაროთ?",
    "en": "Greetings. How may I assist you?",
    "ru": "Приветствую. Чем могу помочь?",
}

def _is_trivial(query: str) -> bool:
    """True for greetings and very short non-substantive inputs."""
    q = query.strip()
    if len(q) < MIN_SUBSTANTIVE_CHARS:
        return bool(_GREETING_PATTERN.match(q))
    return False

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
        # Premium memory query: WHERE user_id=$1 AND created_at>=$2 — needs this index
        """CREATE INDEX IF NOT EXISTS idx_chat_user_date
            ON chat_history(user_id, created_at DESC)
            WHERE user_id IS NOT NULL""",
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

        # Paddle payment transactions log
        """CREATE TABLE IF NOT EXISTS paddle_transactions (
            id              SERIAL PRIMARY KEY,
            paddle_event_id TEXT NOT NULL UNIQUE,
            user_id         INTEGER REFERENCES users(id) ON DELETE SET NULL,
            email           TEXT,
            amount_usd      NUMERIC(10,2),
            status          TEXT,
            raw_payload     TEXT,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )""",
        """CREATE INDEX IF NOT EXISTS idx_paddle_user ON paddle_transactions(user_id)""",

        # Site settings — key/value store for admin-configurable options
        """CREATE TABLE IF NOT EXISTS site_settings (
            key        TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )""",

        # Daily quotes — managed via admin dashboard, shown on the homepage
        """CREATE TABLE IF NOT EXISTS daily_quotes (
            id         SERIAL PRIMARY KEY,
            text_ka    TEXT NOT NULL,
            text_en    TEXT,
            text_ru    TEXT,
            source     TEXT,
            quote_date DATE,
            is_active  BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )""",
        """CREATE INDEX IF NOT EXISTS idx_dq_active ON daily_quotes(is_active)""",
        """CREATE UNIQUE INDEX IF NOT EXISTS idx_dq_date ON daily_quotes(quote_date)
           WHERE quote_date IS NOT NULL""",
    ]

    # Create all tables FIRST
    async with db_pool.acquire() as conn:
        for s in stmts:
            await conn.execute(s)

    # ── Schema migrations (idempotent — safe to re-run on every deploy) ───
    async with db_pool.acquire() as conn:
        for migration in [
            "ALTER TABLE daily_quotes ADD COLUMN IF NOT EXISTS quote_date DATE",
            """CREATE UNIQUE INDEX IF NOT EXISTS idx_dq_date ON daily_quotes(quote_date)
               WHERE quote_date IS NOT NULL""",
        ]:
            try:
                await conn.execute(migration)
            except Exception as _me:
                print(f"Migration note (safe to ignore): {_me}")

    # THEN seed default settings (table now exists)
    async with db_pool.acquire() as conn:
        defaults = [
            # Audio volumes (0-100 scale, divided by 100 for actual JS volume)
            ("bg_music_volume",   "13"),
            ("lamp_volume",       "55"),
            ("book_volume",       "30"),
            ("ring_volume",       "39"),
            ("mutesfx_volume",    "50"),
            # Chat UI
            ("chat_height_px",    "400"),
            ("chat_scroll_px",    "38"),   # chat messages panel scroll speed
            ("reader_scroll_px",  "55"),   # reader body panel scroll speed
            ("ui_lang_default",   "ka"),
            # Donate settings
            ("usdt_address",      "TNBNAYUxbw9LNqbmRjb8EAJ3gL9d19Sbb5"),
            ("usdt_qr_url",       ""),     # custom QR image URL (empty = auto-generate)
            ("paddle_link_5",     ""),     # direct Paddle/Stripe link for $5
            ("paddle_link_10",    ""),     # direct Paddle/Stripe link for $10
        ]
        for k, v in defaults:
            await conn.execute(
                "INSERT INTO site_settings (key, value) VALUES ($1,$2) ON CONFLICT (key) DO NOTHING",
                k, v
            )
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



# == GOOGLE SSO ENDPOINT ====================================================
class GoogleAuthRequest(BaseModel):
    credential: str  # Google Identity Services JWT token

@app.post("/auth/google", status_code=200)
async def auth_google(req: GoogleAuthRequest, request: Request):
    """
    Verify a Google Identity Services JWT token.
    - If email exists in DB → return our app JWT (login)
    - If email is new      → create user, return JWT (register)
    No password needed — Google has already authenticated the user.
    """
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(500, "Google OAuth not configured (GOOGLE_CLIENT_ID missing)")

    # ── 1. Verify the Google token ─────────────────────────────────────────
    try:
        id_info = google_id_token.verify_oauth2_token(
            req.credential,
            google_requests.Request(),
            GOOGLE_CLIENT_ID,
            clock_skew_in_seconds=10,  # tolerate slight clock drift
        )
    except ValueError as e:
        raise HTTPException(401, f"Invalid Google token: {e}")

    email = id_info.get("email", "").lower()
    if not email:
        raise HTTPException(400, "Google token does not contain an email")

    # ── 2. Check if user exists, else create ───────────────────────────────
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, email, role, is_premium, premium_until FROM users WHERE email=$1",
            email
        )

        if row is None:
            # New user — insert with google_sso placeholder password
            ip = request.headers.get(
                "X-Forwarded-For", request.client.host or "Unknown"
            ).split(",")[0].strip()
            row = await conn.fetchrow(
                "INSERT INTO users (email, password_hash, ip_address) "
                "VALUES ($1, 'google_sso', $2) "
                "RETURNING id, email, role, is_premium",
                email, ip
            )
            is_new = True
        else:
            is_new = False

    # ── 3. Handle premium expiry (same logic as /login) ────────────────────
    is_premium = row["is_premium"]
    if not is_new and is_premium and row.get("premium_until"):
        pu = row["premium_until"]
        if pu.tzinfo is None:
            pu = pu.replace(tzinfo=datetime.timezone.utc)
        if pu < datetime.datetime.now(datetime.timezone.utc):
            async with db_pool.acquire() as conn:
                await conn.execute(
                    "UPDATE users SET is_premium=FALSE WHERE id=$1", row["id"]
                )
            is_premium = False

    # ── 4. Return our app JWT ───────────────────────────────────────────────
    token = create_jwt(row["id"], row["email"], row["role"], is_premium)
    print(f"{'🆕' if is_new else '✅'} Google SSO: {email} | new={is_new}")

    return {
        "access_token": token,
        "token_type":   "bearer",
        "is_premium":   is_premium,
        "is_new_user":  is_new,  # frontend can show welcome modal for new users
    }


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

    # Send email via Resend HTTP API (SMTP blocked on Railway hobby plan)
    if RESEND_API_KEY:
        await asyncio.to_thread(_send_reset_email, row["email"], reset_link)
    else:
        print(f"⚠ RESEND_API_KEY not set. Add it in Railway Variables. Reset link: {reset_link}")

    print(f"✅ Reset token generated for {row['email']} | link: {reset_link}")
    return {"status": "ok"}


# RESEND_API_KEY env var — get free key from resend.com (3000 emails/month free)
# Railway hobby plan BLOCKS outbound SMTP (port 587/465) — HTTP API is the only option
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
RESEND_FROM    = os.getenv("RESEND_FROM", "AiStalin <noreply@aistalin.io>")

def _resend_post(from_addr: str, to_email: str, subject: str, html_body: str) -> dict:
    """Make one POST to Resend API. Returns response dict or raises urllib.error.HTTPError."""
    payload = _json.dumps({
        "from":    from_addr,
        "to":      [to_email],
        "subject": subject,
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
    with urllib.request.urlopen(req, timeout=10) as resp:
        return _json.loads(resp.read())


def _send_reset_email(to_email: str, reset_link: str):
    """Send password reset email via Resend HTTP API.
    Strategy:
      1. Try RESEND_FROM (custom domain, e.g. noreply@aistalin.io)
      2. If error 1010 (domain not verified for this API key), 
         auto-fallback to onboarding@resend.dev — always works.
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

    subject = "AiStalin — პაროლის განახლება"

    # Attempt 1: Try configured RESEND_FROM (custom domain noreply@aistalin.io)
    try:
        result = _resend_post(RESEND_FROM, to_email, subject, html_body)
        print(f"✅ Email sent ({RESEND_FROM}) → {to_email} | id: {result.get('id')}")
        return
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        # Error 1010 = "from" domain not verified for this API key scope
        # Auto-fallback to Resend built-in test address — works with ANY valid key
        if e.code == 403 and "1010" in raw:
            print(f"⚠ 1010: domain not authorized for this key → fallback to onboarding@resend.dev")
            try:
                result = _resend_post("onboarding@resend.dev", to_email, subject, html_body)
                print(f"✅ Email sent (onboarding@resend.dev) → {to_email} | id: {result.get('id')}")
                return
            except Exception as e2:
                print(f"⚠ Fallback failed: {e2}")
        else:
            print(f"⚠ Resend error {e.code}: {raw}")
    except Exception as e:
        print(f"⚠ Email failed: {type(e).__name__}: {e}")


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




@app.get("/me")
async def get_me(user: dict = Depends(get_current_user)):
    """Returns current user info + fresh JWT if premium status changed."""
    if not user:
        raise HTTPException(401, "Not authenticated")
    uid = int(user["sub"])
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, email, role, is_premium, premium_until FROM users WHERE id=$1", uid
        )
    if not row:
        raise HTTPException(404, "User not found")
    is_premium = row["is_premium"]
    if is_premium and row["premium_until"]:
        pu = row["premium_until"]
        if pu.tzinfo is None:
            pu = pu.replace(tzinfo=datetime.timezone.utc)
        if pu < datetime.datetime.now(datetime.timezone.utc):
            async with db_pool.acquire() as conn:
                await conn.execute("UPDATE users SET is_premium=FALSE WHERE id=$1", uid)
            is_premium = False
    result = {"id": row["id"], "email": row["email"],
              "role": row["role"], "is_premium": is_premium}
    # Fresh JWT if premium changed (frontend uses this to unlock chat)
    if is_premium != user.get("is_premium", False):
        result["new_token"] = create_jwt(row["id"], row["email"], row["role"], is_premium)
    return result

# == PADDLE PAYMENT ENDPOINTS ==============================================
# Paddle Billing (v2 API) uses HMAC-SHA256 for webhook verification.
# Docs: https://developer.paddle.com/webhooks/signature-verification

def _verify_paddle_signature(raw_body: bytes, signature_header: str, secret: str) -> bool:
    """
    Verify Paddle webhook using HMAC-SHA256.
    Header format: ts=TIMESTAMP;h1=SIGNATURE
    """
    if not secret:
        # No secret configured — reject all webhooks (secure default)
        return False
    try:
        parts = dict(p.split("=", 1) for p in signature_header.split(";"))
        timestamp = parts.get("ts", "")
        signature = parts.get("h1", "")
        if not timestamp or not signature:
            return False
        # Paddle signed payload = "ts:raw_body"
        signed_payload = f"{timestamp}:{raw_body.decode('utf-8')}".encode()
        expected = hmac.new(
            secret.encode("utf-8"),
            signed_payload,
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, signature)
    except Exception as e:
        print(f"⚠ Paddle sig verification error: {e}")
        return False


class PaddleCheckoutSession(BaseModel):
    """Frontend calls this to get customer data to pre-fill Paddle checkout."""
    pass  # all data comes from JWT


@app.get("/paddle/config")
async def get_paddle_config(user: dict = Depends(get_current_user)):
    """
    Returns Paddle config + user data for the frontend checkout.
    Frontend uses this to initialise Paddle.js with the correct price
    and pre-filled customer info.
    """
    if not PADDLE_PRICE_ID:
        raise HTTPException(500, "Paddle not configured (PADDLE_PRICE_ID missing)")

    uid   = int(user["sub"]) if user else None
    email = user.get("email", "") if user else ""

    return {
        "price_id":   PADDLE_PRICE_ID,
        "environment": PADDLE_ENV,        # "sandbox" | "production"
        "customer": {
            "email": email,
        },
        "custom_data": {
            "user_id": str(uid) if uid else "",
            "email":   email,
        },
    }


@app.post("/paddle-webhook", status_code=200)
async def paddle_webhook(request: Request):
    """
    Receives Paddle webhook notifications.
    Verifies signature, then updates the user's premium status.

    Paddle event types we care about:
      transaction.completed  → payment succeeded, grant premium
      subscription.activated → subscription started
      subscription.cancelled → subscription cancelled (optional: downgrade)
    """
    raw_body = await request.body()
    sig_header = request.headers.get("Paddle-Signature", "")

    # ── Security: verify signature ────────────────────────────────────────
    if not _verify_paddle_signature(raw_body, sig_header, PADDLE_WEBHOOK_SECRET):
        print(f"⚠ Paddle webhook: invalid signature | sig={sig_header[:40]}")
        raise HTTPException(400, "Invalid webhook signature")

    try:
        payload = _json.loads(raw_body.decode("utf-8"))
    except Exception:
        raise HTTPException(400, "Invalid JSON payload")

    event_type = payload.get("event_type", "")
    event_id   = payload.get("notification_id") or payload.get("event_id", str(uuid.uuid4()))

    print(f"📦 Paddle event: {event_type} | id: {event_id}")

    # ── Only process successful payment events ─────────────────────────────
    if event_type not in ("transaction.completed", "subscription.activated", "subscription.updated"):
        # Return 200 so Paddle doesn't retry non-payment events
        return {"status": "ignored", "event_type": event_type}

    # ── Extract user identification from custom_data ───────────────────────
    # We pass custom_data.user_id and custom_data.email in the checkout
    data        = payload.get("data", {})
    custom_data = data.get("custom_data") or {}
    email       = (custom_data.get("email") or
                   data.get("customer", {}).get("email", "")).lower()
    user_id_str = custom_data.get("user_id", "")

    # ── Extract billing details ────────────────────────────────────────────
    # Paddle v2 transaction amounts
    details   = data.get("details", {})
    totals    = details.get("totals", {})
    amount    = totals.get("grand_total", 0)
    currency  = data.get("currency_code", "USD")
    status    = data.get("status", event_type)

    # Convert cents to dollars (Paddle uses lowest currency unit)
    try:
        amount_usd = float(amount) / 100
    except (TypeError, ValueError):
        amount_usd = 0.0

    # ── Deduplicate — same event_id should only be processed once ──────────
    async with db_pool.acquire() as conn:
        existing = await conn.fetchval(
            "SELECT id FROM paddle_transactions WHERE paddle_event_id=$1", event_id
        )
        if existing:
            print(f"ℹ️ Duplicate Paddle event {event_id} — skipping")
            return {"status": "already_processed"}

    # ── Find user in DB ────────────────────────────────────────────────────
    user_row = None
    async with db_pool.acquire() as conn:
        # Try by user_id first (most reliable), then fall back to email
        if user_id_str and user_id_str.isdigit():
            user_row = await conn.fetchrow(
                "SELECT id, email FROM users WHERE id=$1", int(user_id_str)
            )
        if not user_row and email:
            user_row = await conn.fetchrow(
                "SELECT id, email FROM users WHERE email=$1", email
            )

    if not user_row:
        # Log the transaction even if we can't find the user
        print(f"⚠ Paddle webhook: user not found | email={email} user_id={user_id_str}")
        async with db_pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO paddle_transactions "
                "(paddle_event_id, user_id, email, amount_usd, status, raw_payload) "
                "VALUES ($1, NULL, $2, $3, $4, $5)",
                event_id, email, amount_usd, "user_not_found", raw_body.decode()[:5000]
            )
        return {"status": "user_not_found"}

    # ── Grant premium — 30 days from now ─────────────────────────────────
    premium_until = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=30)

    async with db_pool.acquire() as conn:
        await conn.execute(
            "UPDATE users SET is_premium=TRUE, premium_until=$1 WHERE id=$2",
            premium_until, user_row["id"]
        )
        # Log successful transaction
        await conn.execute(
            "INSERT INTO paddle_transactions "
            "(paddle_event_id, user_id, email, amount_usd, status, raw_payload) "
            "VALUES ($1, $2, $3, $4, $5, $6)",
            event_id, user_row["id"], user_row["email"],
            amount_usd, "premium_granted", raw_body.decode()[:5000]
        )

    print(f"✅ Premium granted | user_id={user_row['id']} email={user_row['email']} "
          f"amount=${amount_usd:.2f} until={premium_until.date()}")

    return {"status": "ok", "premium_granted": True}


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



# ── Admin guard dependency ─────────────────────────────────────────────────
async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """Dependency that allows only ADMIN_EMAIL through.
    Usage: @app.get("/admin/...") async def handler(admin=Depends(require_admin))
    """
    if not user:
        raise HTTPException(401, "Authentication required")
    if not ADMIN_EMAIL:
        raise HTTPException(503, "Admin not configured (ADMIN_EMAIL env var missing)")
    if user.get("email", "").lower() != ADMIN_EMAIL.lower():
        raise HTTPException(403, "Admin access only")
    return user


# == LEGAL PAGES (HTML) ======================================================
# Self-contained HTML pages served directly by the API.
# No separate static file server needed — works on Railway as-is.

_LEGAL_CSS = """
  body{margin:0;background:#0d0603;color:#d8c9a8;font-family:'Georgia',serif;line-height:1.85}
  .wrap{max-width:760px;margin:0 auto;padding:3rem 1.5rem 5rem}
  h1{font-family:'Playfair Display',serif;font-size:2rem;color:#d4a017;
     border-bottom:1px solid rgba(92,45,15,.5);padding-bottom:.75rem;margin-bottom:2rem}
  h2{font-family:'Playfair Display',serif;font-size:1.15rem;color:#c8941a;margin-top:2rem}
  p,li{font-size:.97rem;opacity:.88}
  ul{padding-left:1.5rem}
  a{color:#d4a017;text-decoration:none}a:hover{text-decoration:underline}
  .back{display:inline-block;margin-bottom:2rem;font-size:.85rem;
        padding:.45rem 1rem;border:1px solid rgba(92,45,15,.6);
        border-radius:4px;color:rgba(212,184,150,.7);transition:all .2s}
  .back:hover{border-color:#d4a017;color:#d4a017}
  .footer-links{margin-top:3rem;padding-top:1rem;
    border-top:1px solid rgba(92,45,15,.4);font-size:.8rem;
    opacity:.55;display:flex;gap:1.5rem;flex-wrap:wrap}
  .footer-links a{color:rgba(212,184,150,.7)}
  strong{color:#e0c880}
  .lang-bar{display:flex;gap:.5rem;margin-bottom:1.75rem}
  .lang-btn{padding:.3rem .85rem;border:1px solid rgba(92,45,15,.55);border-radius:3px;
            background:transparent;color:rgba(212,184,150,.6);font-family:Georgia,serif;
            font-size:.8rem;cursor:pointer;text-decoration:none;transition:all .2s}
  .lang-btn:hover,.lang-btn.active{border-color:#d4a017;color:#d4a017;background:rgba(212,160,23,.07)}
"""

# Legal page labels per language
_LEGAL_LABELS = {
    "ka": {
        "back":    "← AiStalin.io",
        "terms":   "მომსახურების პირობები",
        "privacy": "კონფიდენციალურობა",
        "refund":  "თანხის დაბრუნება",
    },
    "en": {
        "back":    "← AiStalin.io",
        "terms":   "Terms of Service",
        "privacy": "Privacy Policy",
        "refund":  "Refund Policy",
    },
    "ru": {
        "back":    "← AiStalin.io",
        "terms":   "Условия использования",
        "privacy": "Политика конфиденциальности",
        "refund":  "Политика возврата",
    },
}

def _legal_page(title: str, body_html: str, lang: str = "ka", page: str = "terms") -> str:
    """Render a self-contained legal HTML page with language switcher."""
    lb   = _LEGAL_LABELS.get(lang, _LEGAL_LABELS["ka"])
    base = page  # "terms" | "privacy" | "refund"
    def _btn(l, flag, label):
        active = " active" if l == lang else ""
        return f'<a class="lang-btn{active}" href="/{base}?lang={l}">{flag} {label}</a>'
    lang_bar = (
        '<div class="lang-bar">\n'
        + _btn("ka", "🇬🇪", "ქართული") + "\n"
        + _btn("en", "🇬🇧", "English")  + "\n"
        + _btn("ru", "🇷🇺", "Русский")  + "\n"
        + '</div>'
    )
    footer_links = (
        f'<a href="/terms?lang={lang}">{lb["terms"]}</a>\n'
        f'  <a href="/privacy?lang={lang}">{lb["privacy"]}</a>\n'
        f'  <a href="/refund?lang={lang}">{lb["refund"]}</a>\n'
        f'  <a href="mailto:contact@aistalin.io">contact@aistalin.io</a>'
    )
    return f"""<!doctype html><html lang="{lang}"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — AiStalin.io</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Cormorant+Garamond:ital,wght@0,400;1,400&display=swap" rel="stylesheet">
<style>{{_LEGAL_CSS}}</style>
</head><body><div class="wrap">
<a href="https://aistalin.io" class="{lb["back"][:2]}back">&#8592; AiStalin.io</a>
{lang_bar}
{body_html}
<div class="footer-links">
  {footer_links}
</div>
</div></body></html>"""


@app.get("/terms", response_class=HTMLResponse)
async def terms_page(lang: str = "ka"):
    if lang not in ("ka", "en", "ru"): lang = "ka"

    bodies = {
        "ka": """
<h1>მომსახურების პირობები</h1>
<p style="opacity:.55;font-size:.82rem;margin-bottom:2rem">ბოლო განახლება: 2025 წელი</p>
<p>კეთილი იყოს თქვენი მობრძანება AiStalin-ზე. ამ ვებსაიტისა და მასთან დაკავშირებული სერვისების გამოყენებით თქვენ ეთანხმებით ქვემოთ მოცემულ პირობებს. თუ ამ პირობებს არ ეთანხმებით, გთხოვთ არ გამოიყენოთ ჩვენი სერვისი.</p>
<h2>1. სერვისის აღწერა</h2>
<p>AiStalin წარმოადგენს ონლაინ ისტორიულ ბიბლიოთეკასა და AI-ზე დაფუძნებულ საძიებო პლატფორმას. სისტემა მომხმარებლებს აძლევს შესაძლებლობას:</p>
<ul><li>დაათვალიერონ ისტორიული ტექსტები</li><li>გამოიყენონ უფასო საძიებო სისტემა</li><li>გამოიყენონ AI ჩატი ტექსტების მოძებნაში</li></ul>
<p>სერვისის ნაწილი ხელმისაწვდომია უფასოდ, ხოლო AI ჩატის სრული ფუნქციონალი ხელმისაწვდომია პრემიუმ წევრობის ფარგლებში.</p>
<h2>2. კონტენტის წყაროები</h2>
<p>ვებსაიტზე ხელმისაწვდომი ისტორიული ტექსტები აღებულია საჯარო არქივებიდან:</p>
<ul><li><strong>Marxists Internet Archive</strong> (marxists.org)</li><li><strong>RevolutionaryDemocracy.org</strong></li></ul>
<p>AiStalin არ აცხადებს საკუთრების უფლებას ამ ტექსტებზე. პლატფორმა წარმოადგენს ტექნოლოგიურ ინსტრუმენტს ამ არქივებში მასალის მოძიებისთვის.</p>
<h2>3. ტექსტების გამოყენება</h2>
<p>ამ არქივებიდან მიღებული მასალები გავრცელებულია თავისუფლად. მომხმარებლებს შეუძლიათ ტექსტების კითხვა, ციტირება და კოპირება იმ პირობებით, რომლებიც დაშვებულია ორიგინალი არქივების მიერ.</p>
<h2>4. ანგარიშის შექმნა</h2>
<p>ზოგიერთი ფუნქციის გამოყენებისთვის შეიძლება საჭირო გახდეს ანგარიშის შექმნა. მომხმარებელი პასუხისმგებელია ანგარიშის ინფორმაციის სისწორეზე, პაროლის უსაფრთხოებაზე და ანგარიშის გამოყენებაზე.</p>
<h2>5. გადახდები და გამოწერა</h2>
<p>პლატფორმის ზოგიერთი ფუნქცია ხელმისაწვდომია ფასიანი გამოწერის საშუალებით. გადახდები მუშავდება მესამე მხარის გადახდის სისტემის — <strong>Paddle</strong>-ის მეშვეობით. გამოწერა შეიძლება განახლდეს ავტომატურად ბილინგის პერიოდის დასრულების შემდეგ.</p>
<h2>6. AI პასუხები</h2>
<p>AI ჩატი წარმოადგენს საინფორმაციო ინსტრუმენტს. პასუხები ეფუძნება ისტორიულ ტექსტებს, მაგრამ შეიძლება იყოს არასრული. მომხმარებელმა მნიშვნელოვანი გადაწყვეტილებები არ უნდა მიიღოს მხოლოდ ამ სისტემის პასუხებზე დაყრდნობით.</p>
<h2>7. პასუხისმგებლობის შეზღუდვა</h2>
<p>სერვისი მიეწოდება არსებული მდგომარეობით. AiStalin არ აგებს პასუხს არაპირდაპირ ან შემთხვევით ზიანზე სერვისის გამოყენების შედეგად.</p>
<h2>8. ცვლილებები</h2>
<p>ეს პირობები შეიძლება განახლდეს. განახლებული ვერსია გამოქვეყნდება ამ გვერდზე.</p>
<h2>9. კონტაქტი</h2>
<p><a href="mailto:contact@aistalin.io">contact@aistalin.io</a></p>
""",
        "en": """
<h1>Terms of Service</h1>
<p style="opacity:.55;font-size:.82rem;margin-bottom:2rem">Last updated: 2025</p>
<p>Welcome to AiStalin. By using this website and its associated services, you agree to the terms set out below. If you do not agree, please do not use our service.</p>
<h2>1. Description of Service</h2>
<p>AiStalin is an online historical library and AI-powered research platform. The system allows users to:</p>
<ul><li>Browse historical texts</li><li>Use a free full-text search engine</li><li>Use the AI chat assistant to explore texts</li></ul>
<p>Part of the service is available free of charge. The full AI chat functionality is available under a Premium membership.</p>
<h2>2. Content Sources</h2>
<p>The historical texts available on this website are sourced from public archives:</p>
<ul><li><strong>Marxists Internet Archive</strong> (marxists.org)</li><li><strong>RevolutionaryDemocracy.org</strong></li></ul>
<p>AiStalin does not claim ownership of these texts. The platform is a technological tool for locating material within these archives.</p>
<h2>3. Use of Texts</h2>
<p>Materials derived from these archives are freely distributed. Users may read, quote and copy texts subject to the terms permitted by the original archives.</p>
<h2>4. Account Creation</h2>
<p>Some features may require you to create an account. You are responsible for the accuracy of your account information, the security of your password, and all activity under your account.</p>
<h2>5. Payments and Subscriptions</h2>
<p>Some features are available via a paid subscription. Payments are processed by the third-party payment provider <strong>Paddle</strong>. Subscriptions may renew automatically at the end of each billing period.</p>
<h2>6. AI Responses</h2>
<p>The AI chat is an informational tool. Responses are based on historical texts but may be incomplete. Users should not make significant decisions based solely on the system's outputs.</p>
<h2>7. Limitation of Liability</h2>
<p>The service is provided as-is. AiStalin is not liable for indirect or incidental damages resulting from use of the service.</p>
<h2>8. Changes</h2>
<p>These terms may be updated. The updated version will be published on this page.</p>
<h2>9. Contact</h2>
<p><a href="mailto:contact@aistalin.io">contact@aistalin.io</a></p>
""",
        "ru": """
<h1>Условия использования</h1>
<p style="opacity:.55;font-size:.82rem;margin-bottom:2rem">Последнее обновление: 2025 г.</p>
<p>Добро пожаловать на AiStalin. Используя этот сайт и связанные с ним сервисы, вы соглашаетесь с приведёнными ниже условиями. Если вы не принимаете эти условия, пожалуйста, не пользуйтесь нашим сервисом.</p>
<h2>1. Описание сервиса</h2>
<p>AiStalin — онлайн-историческая библиотека и исследовательская платформа на базе ИИ. Система позволяет пользователям:</p>
<ul><li>Просматривать исторические тексты</li><li>Использовать бесплатную поисковую систему</li><li>Использовать ИИ-чат для работы с текстами</li></ul>
<p>Часть сервиса доступна бесплатно. Полная функциональность ИИ-чата доступна в рамках Премиум-подписки.</p>
<h2>2. Источники контента</h2>
<p>Исторические тексты на этом сайте получены из публичных архивов:</p>
<ul><li><strong>Marxists Internet Archive</strong> (marxists.org)</li><li><strong>RevolutionaryDemocracy.org</strong></li></ul>
<p>AiStalin не претендует на владение этими текстами. Платформа является технологическим инструментом для поиска материалов в этих архивах.</p>
<h2>3. Использование текстов</h2>
<p>Материалы из указанных архивов распространяются свободно. Пользователи могут читать, цитировать и копировать тексты на условиях, разрешённых оригинальными архивами.</p>
<h2>4. Создание аккаунта</h2>
<p>Для использования некоторых функций может потребоваться создание аккаунта. Пользователь несёт ответственность за точность данных аккаунта, безопасность пароля и все действия под аккаунтом.</p>
<h2>5. Платежи и подписки</h2>
<p>Часть функций доступна по платной подписке. Платежи обрабатываются сторонней платёжной системой <strong>Paddle</strong>. Подписка может автоматически продлеваться по окончании расчётного периода.</p>
<h2>6. Ответы ИИ</h2>
<p>ИИ-чат является информационным инструментом. Ответы основаны на исторических текстах, но могут быть неполными. Не следует принимать важные решения, опираясь исключительно на ответы системы.</p>
<h2>7. Ограничение ответственности</h2>
<p>Сервис предоставляется «как есть». AiStalin не несёт ответственности за косвенный или случайный ущерб, возникший в результате использования сервиса.</p>
<h2>8. Изменения</h2>
<p>Настоящие условия могут обновляться. Актуальная версия будет опубликована на этой странице.</p>
<h2>9. Контакты</h2>
<p><a href="mailto:contact@aistalin.io">contact@aistalin.io</a></p>
""",
    }
    titles = {"ka": "მომსახურების პირობები", "en": "Terms of Service", "ru": "Условия использования"}
    return _legal_page(titles[lang], bodies[lang], lang=lang, page="terms")


@app.get("/privacy", response_class=HTMLResponse)
async def privacy_page(lang: str = "ka"):
    if lang not in ("ka", "en", "ru"): lang = "ka"

    bodies = {
        "ka": """
<h1>კონფიდენციალურობის პოლიტიკა</h1>
<p style="opacity:.55;font-size:.82rem;margin-bottom:2rem">ბოლო განახლება: 2025 წელი</p>
<p>AiStalin პატივს სცემს მომხმარებლის კონფიდენციალურობას. ეს პოლიტიკა განმარტავს, რა მონაცემებს ვაგროვებთ და როგორ ვიყენებთ მათ.</p>
<h2>1. რა მონაცემებს ვაგროვებთ</h2>
<ul><li>ელფოსტა ანგარიშის შექმნისას</li><li>ჩატის შეტყობინებები (სერვისის გაუმჯობესებისთვის)</li><li>ტექნიკური მონაცემები (IP, ბრაუზერი, მოწყობილობა)</li><li>სისტემის ლოგები უსაფრთხოების მიზნით</li></ul>
<h2>2. როგორ ვიყენებთ მონაცემებს</h2>
<ul><li>მომხმარებლის ანგარიშის სამართავად</li><li>AI ჩატის ფუნქციონირებისთვის</li><li>სისტემის გაუმჯობესებისთვის</li><li>უსაფრთხოების მიზნით</li></ul>
<h2>3. გადახდები</h2>
<p>გადახდები მუშავდება <strong>Paddle</strong>-ის მიერ. ჩვენ არ ვინახავთ სრულ საბარათე მონაცემებს ჩვენს სისტემაში.</p>
<h2>4. Cookies</h2>
<p>ვებსაიტი იყენებს cookies ავტორიზაციისთვის, სესიის სამართავად და გამოცდილების გასაუმჯობესებლად.</p>
<h2>5. მონაცემების გაზიარება</h2>
<p>ჩვენ <strong>არ ვყიდით</strong> მომხმარებლის მონაცემებს. მონაცემები შეიძლება გაზიარდეს მხოლოდ ტექნიკურ სერვის პროვაიდერებთან ან კანონით მოთხოვნილ შემთხვევებში.</p>
<h2>6. კონტენტის წყაროები</h2>
<p>ვებსაიტზე წარმოდგენილი ისტორიული მასალა მიღებულია ღია არქივებიდან (<strong>Marxists Internet Archive</strong>, <strong>RevolutionaryDemocracy.org</strong>).</p>
<h2>7. მონაცემების უსაფრთხოება</h2>
<p>ვიყენებთ შესაბამის ტექნიკურ ზომებს (HTTPS, bcrypt password hashing, JWT tokens) მონაცემების დასაცავად.</p>
<h2>8. ცვლილებები</h2>
<p>ეს პოლიტიკა შეიძლება პერიოდულად განახლდეს.</p>
<h2>9. კონტაქტი</h2>
<p><a href="mailto:contact@aistalin.io">contact@aistalin.io</a></p>
""",
        "en": """
<h1>Privacy Policy</h1>
<p style="opacity:.55;font-size:.82rem;margin-bottom:2rem">Last updated: 2025</p>
<p>AiStalin respects user privacy. This policy explains what data we collect and how we use it.</p>
<h2>1. Data We Collect</h2>
<ul><li>Email address when creating an account</li><li>Chat messages (to improve the service)</li><li>Technical data (IP address, browser, device)</li><li>System logs for security purposes</li></ul>
<h2>2. How We Use Data</h2>
<ul><li>To manage your user account</li><li>To operate the AI chat functionality</li><li>To improve the system</li><li>For security purposes</li></ul>
<h2>3. Payments</h2>
<p>Payments are processed by <strong>Paddle</strong>. We do not store full card details on our systems.</p>
<h2>4. Cookies</h2>
<p>The website uses cookies for authentication, session management, and to improve the user experience.</p>
<h2>5. Data Sharing</h2>
<p>We do <strong>not sell</strong> user data. Data may be shared only with technical service providers or when required by law.</p>
<h2>6. Content Sources</h2>
<p>Historical material on this website is sourced from open archives (<strong>Marxists Internet Archive</strong>, <strong>RevolutionaryDemocracy.org</strong>). These materials are freely available.</p>
<h2>7. Data Security</h2>
<p>We apply appropriate technical measures (HTTPS, bcrypt password hashing, JWT tokens) to protect your data.</p>
<h2>8. Changes</h2>
<p>This policy may be updated periodically.</p>
<h2>9. Contact</h2>
<p><a href="mailto:contact@aistalin.io">contact@aistalin.io</a></p>
""",
        "ru": """
<h1>Политика конфиденциальности</h1>
<p style="opacity:.55;font-size:.82rem;margin-bottom:2rem">Последнее обновление: 2025 г.</p>
<p>AiStalin уважает конфиденциальность пользователей. Настоящая политика разъясняет, какие данные мы собираем и как мы их используем.</p>
<h2>1. Какие данные мы собираем</h2>
<ul><li>Адрес электронной почты при создании аккаунта</li><li>Сообщения в чате (для улучшения сервиса)</li><li>Технические данные (IP-адрес, браузер, устройство)</li><li>Системные журналы в целях безопасности</li></ul>
<h2>2. Как мы используем данные</h2>
<ul><li>Для управления аккаунтом пользователя</li><li>Для работы ИИ-чата</li><li>Для улучшения системы</li><li>В целях безопасности</li></ul>
<h2>3. Платежи</h2>
<p>Платежи обрабатываются <strong>Paddle</strong>. Мы не храним полные данные платёжных карт в наших системах.</p>
<h2>4. Cookies</h2>
<p>Сайт использует cookies для авторизации, управления сессией и улучшения работы сервиса.</p>
<h2>5. Передача данных</h2>
<p>Мы <strong>не продаём</strong> данные пользователей. Данные могут передаваться только техническим поставщикам услуг или в случаях, предусмотренных законодательством.</p>
<h2>6. Источники контента</h2>
<p>Исторические материалы на сайте получены из открытых архивов (<strong>Marxists Internet Archive</strong>, <strong>RevolutionaryDemocracy.org</strong>). Эти материалы находятся в свободном доступе.</p>
<h2>7. Безопасность данных</h2>
<p>Мы применяем соответствующие технические меры (HTTPS, bcrypt-хеширование паролей, JWT-токены) для защиты ваших данных.</p>
<h2>8. Изменения</h2>
<p>Настоящая политика может периодически обновляться.</p>
<h2>9. Контакты</h2>
<p><a href="mailto:contact@aistalin.io">contact@aistalin.io</a></p>
""",
    }
    titles = {"ka": "კონფიდენციალურობის პოლიტიკა", "en": "Privacy Policy", "ru": "Политика конфиденциальности"}
    return _legal_page(titles[lang], bodies[lang], lang=lang, page="privacy")


@app.get("/refund", response_class=HTMLResponse)
async def refund_page(lang: str = "ka"):
    if lang not in ("ka", "en", "ru"): lang = "ka"

    bodies = {
        "ka": """
<h1>თანხის დაბრუნების პოლიტიკა</h1>
<p style="opacity:.55;font-size:.82rem;margin-bottom:2rem">ბოლო განახლება: 2025 წელი</p>
<p>ეს პოლიტიკა განმარტავს თანხის დაბრუნებისა და გამოწერის გაუქმების წესებს AiStalin-ის ფასიანი სერვისებისთვის.</p>
<h2>1. გამოწერის ტიპი</h2>
<p>AiStalin გთავაზობთ <strong>$5/თვე</strong> პრემიუმ გამოწერას, რომელიც იძლევა AI ჩატზე ულიმიტო წვდომას. გადახდები მუშავდება <strong>Paddle</strong>-ის მეშვეობით.</p>
<h2>2. გამოწერის გაუქმება</h2>
<ul><li><strong>ვარიანტი 1:</strong> Paddle-ისგან მიღებული დადასტურების ელფოსტიდან → <em>Manage Subscription</em> ბმულზე დააჭირეთ</li><li><strong>ვარიანტი 2:</strong> მოგვწერეთ <a href="mailto:contact@aistalin.io">contact@aistalin.io</a>-ზე</li></ul>
<p>მიმდინარე ბილინგის პერიოდი ძალაში რჩება მის დასრულებამდე.</p>
<h2>3. თანხის დაბრუნების წესი</h2>
<p>ციფრული სერვისის ბუნებიდან გამომდინარე, უკვე დაწყებული ბილინგის პერიოდის თანხა, როგორც წესი, არ ბრუნდება.</p>
<h2>4. გამონაკლისები</h2>
<ul><li>დუბლირებული გადახდა</li><li>ტექნიკური შეცდომით არასწორი ჩამოჭრა</li><li>სერვისის ხანგრძლივი ტექნიკური გაუმართაობა</li></ul>
<h2>5. მოთხოვნის გაგზავნა</h2>
<p>მოთხოვნაში მიუთითეთ: თქვენი ელფოსტა, გადახდის თარიღი, პრობლემის აღწერა.<br><a href="mailto:contact@aistalin.io">contact@aistalin.io</a></p>
""",
        "en": """
<h1>Refund Policy</h1>
<p style="opacity:.55;font-size:.82rem;margin-bottom:2rem">Last updated: 2025</p>
<p>This policy explains the rules for refunds and subscription cancellations for AiStalin's paid services.</p>
<h2>1. Subscription Type</h2>
<p>AiStalin offers a <strong>$5/month</strong> Premium subscription that provides unlimited access to the AI chat. Payments are processed by <strong>Paddle</strong>.</p>
<h2>2. Cancelling Your Subscription</h2>
<ul><li><strong>Option 1:</strong> In the confirmation email from Paddle → click the <em>Manage Subscription</em> link</li><li><strong>Option 2:</strong> Email us at <a href="mailto:contact@aistalin.io">contact@aistalin.io</a> with a cancellation request</li></ul>
<p>Your current billing period remains active until its end date.</p>
<h2>3. Refund Policy</h2>
<p>Due to the nature of digital services, fees already charged for an active billing period are generally non-refundable.</p>
<h2>4. Exceptions</h2>
<ul><li>Duplicate charge</li><li>Incorrect charge due to a technical error</li><li>Extended service outage</li></ul>
<h2>5. Submitting a Request</h2>
<p>Please include: your email address, the payment date, and a description of the issue.<br><a href="mailto:contact@aistalin.io">contact@aistalin.io</a></p>
""",
        "ru": """
<h1>Политика возврата средств</h1>
<p style="opacity:.55;font-size:.82rem;margin-bottom:2rem">Последнее обновление: 2025 г.</p>
<p>Настоящая политика разъясняет правила возврата средств и отмены подписки на платные сервисы AiStalin.</p>
<h2>1. Тип подписки</h2>
<p>AiStalin предлагает Премиум-подписку за <strong>$5 в месяц</strong>, обеспечивающую неограниченный доступ к ИИ-чату. Платежи обрабатываются через <strong>Paddle</strong>.</p>
<h2>2. Отмена подписки</h2>
<ul><li><strong>Вариант 1:</strong> В подтверждающем письме от Paddle → нажмите ссылку <em>Manage Subscription</em></li><li><strong>Вариант 2:</strong> Напишите нам на <a href="mailto:contact@aistalin.io">contact@aistalin.io</a> с запросом об отмене</li></ul>
<p>Текущий расчётный период остаётся активным до его окончания.</p>
<h2>3. Правила возврата</h2>
<p>В связи с природой цифровых услуг, средства за уже начавшийся расчётный период, как правило, не возвращаются.</p>
<h2>4. Исключения</h2>
<ul><li>Двойное списание</li><li>Ошибочное списание вследствие технической ошибки</li><li>Длительный технический сбой сервиса</li></ul>
<h2>5. Подача заявки</h2>
<p>Укажите: ваш адрес электронной почты, дату платежа и описание проблемы.<br><a href="mailto:contact@aistalin.io">contact@aistalin.io</a></p>
""",
    }
    titles = {"ka": "თანხის დაბრუნების პოლიტიკა", "en": "Refund Policy", "ru": "Политика возврата средств"}
    return _legal_page(titles[lang], bodies[lang], lang=lang, page="refund")


# == ADMIN ENDPOINTS ==========================================================
# All endpoints require ADMIN_EMAIL via require_admin dependency.

# ── One-time dashboard token ─────────────────────────────────────────────
@app.post("/admin/token")
async def get_admin_dash_token(admin: dict = Depends(require_admin)):
    """
    Called by the frontend gear icon click.
    Returns a short-lived (60s) single-use token.
    Frontend opens /admin/dashboard?token=XXX in a new tab.
    This avoids needing to pass JWT in URL params.
    """
    # Clean expired tokens
    now = time.time()
    expired = [k for k, v in _admin_dash_tokens.items() if v["expires"] < now]
    for k in expired:
        del _admin_dash_tokens[k]

    token = secrets.token_urlsafe(32)
    _admin_dash_tokens[token] = {
        "email":   admin["email"],
        "expires": now + 60,   # valid for 60 seconds only
    }
    return {"token": token, "expires_in": 60}


# ── Stats ─────────────────────────────────────────────────────────────────
@app.get("/admin/stats")
async def admin_stats(admin: dict = Depends(require_admin)):
    async with db_pool.acquire() as conn:
        user_count    = await conn.fetchval("SELECT COUNT(*) FROM users")
        premium_count = await conn.fetchval("SELECT COUNT(*) FROM users WHERE is_premium=TRUE")
        chat_today    = await conn.fetchval(
            "SELECT COUNT(*) FROM chat_history WHERE created_at::date = CURRENT_DATE"
        )
        users_today   = await conn.fetchval(
            "SELECT COUNT(*) FROM users WHERE created_at::date = CURRENT_DATE"
        )
        tx_total      = await conn.fetchval(
            "SELECT COALESCE(SUM(amount_usd),0) FROM paddle_transactions WHERE status='premium_granted'"
        )
        tx_month      = await conn.fetchval(
            """SELECT COALESCE(SUM(amount_usd),0) FROM paddle_transactions
               WHERE status='premium_granted'
               AND created_at >= date_trunc('month', CURRENT_DATE)"""
        )
    return {
        "users_total":   user_count,
        "users_premium": premium_count,
        "users_today":   users_today,
        "chats_today":   chat_today,
        "revenue_total": float(tx_total),
        "revenue_month": float(tx_month),
        "admin_email":   admin["email"],
    }


# ── Site Settings CRUD ─────────────────────────────────────────────────────
@app.get("/admin/settings")
async def get_settings(admin: dict = Depends(require_admin)):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("SELECT key, value FROM site_settings ORDER BY key")
    return {r["key"]: r["value"] for r in rows}

class SettingUpdate(BaseModel):
    key:   str
    value: str

@app.post("/admin/settings")
async def update_setting(req: SettingUpdate, admin: dict = Depends(require_admin)):
    async with db_pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO site_settings (key, value, updated_at) VALUES ($1,$2,NOW())
               ON CONFLICT (key) DO UPDATE SET value=$2, updated_at=NOW()""",
            req.key, req.value
        )
    return {"status": "ok", "key": req.key, "value": req.value}


# ── Public settings (read-only, no auth) ──────────────────────────────────
@app.get("/settings/public")
async def get_public_settings():
    """Read-only public endpoint for site settings.
    Frontend uses this to apply admin-configured audio volumes, chat height, etc.
    No authentication required — these are display preferences only.
    Response is cached-friendly (settings change rarely).
    """
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("SELECT key, value FROM site_settings ORDER BY key")
    return {r["key"]: r["value"] for r in rows}


# ── Daily Quote auto-translation via Gemini Flash ────────────────────────────
async def _auto_translate_quote(text_ka: str) -> tuple[str, str]:
    """Translate Georgian Stalin quote → EN + RU using Gemini Flash.
    Returns (text_en, text_ru). Falls back to ("","") on any error."""
    prompt = (
        "You are a professional translator specialising in Soviet-era political texts.\n"
        "Translate the following Georgian quote attributed to Joseph Stalin "
        "into English and Russian.\n\n"
        "Rules:\n"
        "- Preserve the original rhetorical register (disciplined, declarative, political)\n"
        "- Do NOT add quotation marks, annotations, or preamble\n"
        "- Output ONLY valid JSON: {\"en\": \"<English>\", \"ru\": \"<Russian>\"}\n\n"
        f"Georgian original:\n{text_ka}"
    )
    def _call():
        model = genai.GenerativeModel(
            model_name="models/gemini-2.5-flash",
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=512, temperature=0.2,
            )
        )
        return model.generate_content(prompt)
    try:
        gen  = await asyncio.to_thread(_call)
        raw  = gen.text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        import json as _j
        data = _j.loads(raw)
        return data.get("en", ""), data.get("ru", "")
    except Exception as e:
        print(f"⚠ Auto-translate failed: {e}")
        return "", ""


# ── Daily Quotes CRUD ──────────────────────────────────────────────────────
@app.get("/admin/quotes")
async def list_quotes(admin: dict = Depends(require_admin)):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id,text_ka,text_en,text_ru,source,is_active,quote_date,created_at FROM daily_quotes ORDER BY COALESCE(quote_date,'2099-12-31') ASC, id DESC"
        )
    return [{"id":r["id"],"text_ka":r["text_ka"],"text_en":r["text_en"],
             "text_ru":r["text_ru"],"source":r["source"],"is_active":r["is_active"],
             "quote_date": str(r["quote_date"]) if r["quote_date"] else None} for r in rows]

class QuoteCreate(BaseModel):
    text_ka:        str
    text_en:        Optional[str] = None
    text_ru:        Optional[str] = None
    source:         Optional[str] = None
    quote_date:     Optional[str] = None   # "YYYY-MM-DD"
    is_active:      bool          = True
    auto_translate: bool          = False  # if True → Gemini EN+RU

@app.post("/admin/quotes/translate")
async def translate_quote_endpoint(body: dict, admin: dict = Depends(require_admin)):
    """Standalone translate: {"text_ka":"..."} → {"text_en":"...","text_ru":"..."}"""
    text_ka = (body.get("text_ka") or "").strip()
    if not text_ka:
        raise HTTPException(400, "text_ka is required")
    en, ru = await _auto_translate_quote(text_ka)
    return {"text_en": en, "text_ru": ru}


@app.post("/admin/quotes", status_code=201)
async def create_quote(req: QuoteCreate, admin: dict = Depends(require_admin)):
    import datetime as _dt
    en, ru = req.text_en, req.text_ru
    if req.auto_translate and (not en or not ru):
        en, ru = await _auto_translate_quote(req.text_ka)
    qdate = None
    if req.quote_date:
        try:    qdate = _dt.date.fromisoformat(req.quote_date)
        except: raise HTTPException(400, "quote_date must be YYYY-MM-DD")
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """INSERT INTO daily_quotes (text_ka,text_en,text_ru,source,is_active,quote_date)
               VALUES ($1,$2,$3,$4,$5,$6)
               ON CONFLICT (quote_date)
               DO UPDATE SET text_ka=$1,text_en=$2,text_ru=$3,source=$4,is_active=$5
               RETURNING id""",
            req.text_ka, en or None, ru or None, req.source, req.is_active, qdate
        )
    return {"status":"created","id":row["id"],"text_en":en,"text_ru":ru}

@app.put("/admin/quotes/{qid}")
async def update_quote(qid: int, req: QuoteCreate, admin: dict = Depends(require_admin)):
    import datetime as _dt
    en, ru = req.text_en, req.text_ru
    if req.auto_translate and (not en or not ru):
        en, ru = await _auto_translate_quote(req.text_ka)
    qdate = None
    if req.quote_date:
        try:    qdate = _dt.date.fromisoformat(req.quote_date)
        except: raise HTTPException(400, "quote_date must be YYYY-MM-DD")
    async with db_pool.acquire() as conn:
        r = await conn.execute(
            """UPDATE daily_quotes
               SET text_ka=$1,text_en=$2,text_ru=$3,source=$4,is_active=$5,quote_date=$6
               WHERE id=$7""",
            req.text_ka, en or None, ru or None, req.source, req.is_active, qdate, qid
        )
    if r == "UPDATE 0": raise HTTPException(404, "Quote not found")
    return {"status":"updated","text_en":en,"text_ru":ru}

@app.delete("/admin/quotes/{qid}")
async def delete_quote(qid: int, admin: dict = Depends(require_admin)):
    async with db_pool.acquire() as conn:
        r = await conn.execute("DELETE FROM daily_quotes WHERE id=$1", qid)
    if r=="DELETE 0": raise HTTPException(404,"Quote not found")
    return {"status":"deleted"}

# Public endpoint — priority: scheduled date → random active fallback
@app.get("/quotes/today")
async def get_today_quote():
    """
    1. Returns quote where quote_date = today (admin-scheduled).
    2. Falls back to random active quote (day-seeded, consistent within a day).
    3. Returns {"text_ka": null} if table is empty.
    """
    import datetime as _dt, time as _t
    today = _dt.date.today()
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT text_ka,text_en,text_ru,source FROM daily_quotes WHERE quote_date=$1 AND is_active=TRUE",
            today
        )
        if not row:
            rows = await conn.fetch(
                "SELECT text_ka,text_en,text_ru,source FROM daily_quotes WHERE is_active=TRUE ORDER BY id"
            )
            if not rows:
                return {"text_ka": None}
            row = rows[int(_t.time() // 86400) % len(rows)]
    return {"text_ka":row["text_ka"],"text_en":row["text_en"],"text_ru":row["text_ru"],"source":row["source"]}


# ── User Management ────────────────────────────────────────────────────────
@app.get("/admin/users")
async def list_users(
    search: str = "", page: int = 1, per_page: int = 30,
    admin: dict = Depends(require_admin)
):
    offset = (page - 1) * per_page
    async with db_pool.acquire() as conn:
        if search:
            rows = await conn.fetch(
                """SELECT id,email,role,is_premium,premium_until,created_at,ip_address
                   FROM users WHERE email ILIKE $1 ORDER BY id DESC LIMIT $2 OFFSET $3""",
                f"%{search}%", per_page, offset
            )
            total = await conn.fetchval("SELECT COUNT(*) FROM users WHERE email ILIKE $1", f"%{search}%")
        else:
            rows = await conn.fetch(
                """SELECT id,email,role,is_premium,premium_until,created_at,ip_address
                   FROM users ORDER BY id DESC LIMIT $1 OFFSET $2""",
                per_page, offset
            )
            total = await conn.fetchval("SELECT COUNT(*) FROM users")
    return {
        "total": total, "page": page, "per_page": per_page,
        "users": [{
            "id":          r["id"],
            "email":       r["email"],
            "role":        r["role"],
            "is_premium":  r["is_premium"],
            "premium_until": r["premium_until"].isoformat() if r["premium_until"] else None,
            "created_at":  r["created_at"].strftime("%Y-%m-%d"),
            "ip":          r["ip_address"],
        } for r in rows]
    }

class PremiumToggle(BaseModel):
    is_premium: bool
    days:       int = 30  # how many days to grant (if enabling)

@app.post("/admin/users/{uid}/premium")
async def toggle_user_premium(uid: int, req: PremiumToggle, admin: dict = Depends(require_admin)):
    if req.is_premium:
        until = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=req.days)
        async with db_pool.acquire() as conn:
            await conn.execute(
                "UPDATE users SET is_premium=TRUE, premium_until=$1 WHERE id=$2", until, uid
            )
    else:
        async with db_pool.acquire() as conn:
            await conn.execute(
                "UPDATE users SET is_premium=FALSE, premium_until=NULL WHERE id=$1", uid
            )
    return {"status":"ok","user_id":uid,"is_premium":req.is_premium}


# ── Optional admin auth (for browser navigation with ?token=) ─────────────
async def _optional_admin(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> dict:
    """Like get_current_user but returns empty dict instead of raising 401."""
    if not credentials:
        return {}
    result = decode_jwt(credentials.credentials)
    return result or {}


# ── Admin Dashboard HTML ───────────────────────────────────────────────────
@app.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard(
    request: Request,
    admin: dict = Depends(_optional_admin),   # type: ignore[assignment]
):
    """
    Full admin control panel.
    Accepts either:
      - Standard JWT in Authorization header (API calls)
      - ?token=ONE_TIME_TOKEN query param (browser redirect from gear icon)
    """
    email = None

    # Path 1: one-time token from query param (browser navigation)
    qtoken = request.query_params.get("token", "")
    if qtoken:
        now = time.time()
        tok_data = _admin_dash_tokens.pop(qtoken, None)
        if tok_data and tok_data["expires"] > now:
            email = tok_data["email"]
        else:
            return HTMLResponse(
                "<html><body style='background:#0d0603;color:#f5e6c8;font-family:Georgia,serif;"
                "display:flex;align-items:center;justify-content:center;height:100vh;margin:0;'>"
                "<div style='text-align:center'>"
                "<p style='font-size:1.5rem;color:#d4a017;'>⚠ ლინკი ვადაგასულია</p>"
                "<p style='opacity:.6;margin-top:.5rem;'>გთხოვთ კვლავ დააჭიროთ ⚙ ღილაკს</p>"
                "<a href='https://aistalin.io' style='color:#d4a017;margin-top:1rem;display:block;'>← aistalin.io</a>"
                "</div></body></html>",
                status_code=403
            )

    # Path 2: JWT in Authorization header (already validated by Depends)
    if email is None and admin:
        email = admin.get("email", "")

    if not email or email.lower() != ADMIN_EMAIL.lower():
        return HTMLResponse(
            "<html><body style='background:#0d0603;color:#f5e6c8;font-family:Georgia,serif;"
            "display:flex;align-items:center;justify-content:center;height:100vh;margin:0;'>"
            "<div style='text-align:center'>"
            "<p style='font-size:1.5rem;color:#cc1b1b;'>⛔ Access Denied</p>"
            "<p style='opacity:.6;margin-top:.5rem;'>Admin access only</p>"
            "</div></body></html>",
            status_code=403
        )

    # Retrieve the original JWT to embed in dashboard sessionStorage
    # The one-time token was already consumed; we now pass the real JWT
    # so the dashboard can make authenticated API calls
    raw_jwt = ""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        raw_jwt = auth_header[7:]
    # Also try to get from qtoken flow (we stored email, not raw jwt)
    # For qtoken flow, we issue a fresh admin JWT
    if not raw_jwt and email:
        # Create a short-lived JWT for the dashboard tab
        raw_jwt = create_jwt(0, email, "admin", False)

    return _admin_dashboard_html(email, jwt_token=raw_jwt)





def _admin_dashboard_html(admin_email: str, jwt_token: str = "") -> str:
    """Self-contained admin dashboard. JWT passed via sessionStorage on load."""
    api = "https://aistalin-backend-production.up.railway.app"
    return f"""<!doctype html><html lang="ka"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Admin — AiStalin.io</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
<style>
:root{{--bg:#0d0603;--bg2:#140804;--bg3:#1a0c04;--border:rgba(92,45,15,.5);
  --gold:#d4a017;--gold2:rgba(212,160,23,.65);--cream:#f5e6c8;--cream2:rgba(212,184,150,.7);
  --red:#cc1b1b;--green:#4caf50;--r:8px}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--cream2);font-family:'Inter',sans-serif;font-size:14px;min-height:100vh}}
.shell{{display:flex;min-height:100vh}}
.sidebar{{width:220px;flex-shrink:0;background:var(--bg2);border-right:1px solid var(--border);
  padding:1.5rem 0;display:flex;flex-direction:column;position:sticky;top:0;height:100vh;overflow-y:auto}}
.main{{flex:1;padding:2rem;overflow-x:hidden;max-width:100%}}
@media(max-width:768px){{
  .shell{{flex-direction:column}}
  .sidebar{{width:100%;height:auto;position:relative;padding:.5rem;display:flex;flex-direction:row;flex-wrap:wrap;gap:.25rem}}
  .main{{padding:1rem}}
  .sidebar .logo{{width:100%;padding:.5rem;margin-bottom:.25rem}}
  .sidebar-foot{{display:none}}
  .grid-4{{grid-template-columns:repeat(2,1fr)!important}}
  .grid-3{{grid-template-columns:repeat(2,1fr)!important}}
  .grid-2{{grid-template-columns:1fr!important}}
  table th.hide-m,table td.hide-m{{display:none}}
}}
.logo{{padding:0 1.25rem 1rem;border-bottom:1px solid var(--border);margin-bottom:.5rem}}
.logo h1{{font-family:'Playfair Display',serif;font-size:1.05rem;color:var(--gold);letter-spacing:.08em}}
.logo p{{font-size:.68rem;color:rgba(212,184,150,.4);margin-top:.2rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.nav-item{{display:flex;align-items:center;gap:.6rem;padding:.55rem 1.2rem;
  color:var(--cream2);text-decoration:none;cursor:pointer;border:none;background:none;
  width:100%;font-size:.85rem;font-family:'Inter',sans-serif;transition:all .15s;
  border-left:2px solid transparent}}
.nav-item:hover,.nav-item.active{{background:rgba(212,160,23,.08);color:var(--gold);border-left-color:var(--gold)}}
.nav-item i{{width:16px;opacity:.7;font-size:.82rem}}
.sidebar-foot{{margin-top:auto;padding:1rem 1.25rem;border-top:1px solid var(--border);font-size:.72rem;color:rgba(212,184,150,.35)}}
.section{{display:none}}.section.active{{display:block}}
.page-title{{font-family:'Playfair Display',serif;font-size:1.3rem;color:var(--gold);
  margin-bottom:1.5rem;padding-bottom:.6rem;border-bottom:1px solid var(--border)}}
.card{{background:var(--bg3);border:1px solid var(--border);border-radius:var(--r);padding:1.25rem;margin-bottom:1rem}}
.card-title{{font-size:.72rem;text-transform:uppercase;letter-spacing:.1em;color:rgba(212,184,150,.45);margin-bottom:.85rem;display:flex;align-items:center;gap:.5rem}}
.grid-4{{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:1.5rem}}
.grid-3{{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:1rem}}
.grid-2{{display:grid;grid-template-columns:repeat(2,1fr);gap:1rem}}
.stat-card{{background:var(--bg3);border:1px solid var(--border);border-radius:var(--r);padding:1.25rem;text-align:center}}
.stat-num{{font-family:'Playfair Display',serif;font-size:2rem;color:var(--gold);line-height:1}}
.stat-lbl{{font-size:.7rem;color:rgba(212,184,150,.45);margin-top:.3rem;letter-spacing:.06em}}
label{{display:block;font-size:.75rem;color:var(--cream2);margin-bottom:.35rem;opacity:.8}}
input[type=text],input[type=number],textarea,select{{
  width:100%;background:rgba(15,5,2,.7);border:1px solid var(--border);border-radius:5px;
  padding:.5rem .75rem;color:var(--cream);font-family:'Inter',sans-serif;font-size:.88rem;outline:none;transition:border-color .2s}}
input:focus,textarea:focus,select:focus{{border-color:var(--gold)}}
textarea{{resize:vertical;min-height:72px}}
input[type=range]{{width:100%;accent-color:var(--gold);cursor:pointer;margin:.15rem 0}}
.btn{{display:inline-flex;align-items:center;gap:.4rem;padding:.5rem 1rem;border-radius:5px;
  border:none;cursor:pointer;font-size:.82rem;font-family:'Inter',sans-serif;font-weight:500;transition:all .18s;white-space:nowrap}}
.btn-gold{{background:linear-gradient(135deg,#d4a017,#a07810);color:#1a0c04}}
.btn-gold:hover{{filter:brightness(1.12);transform:translateY(-1px)}}
.btn-red{{background:rgba(204,27,27,.8);color:#fff}}.btn-red:hover{{background:rgba(204,27,27,1)}}
.btn-ghost{{background:none;border:1px solid var(--border);color:var(--cream2)}}.btn-ghost:hover{{border-color:var(--gold);color:var(--gold)}}
.btn-green{{background:rgba(46,125,50,.85);color:#fff}}.btn-green:hover{{background:rgba(46,125,50,1)}}
.btn-sm{{padding:.3rem .7rem;font-size:.75rem}}
.tbl-wrap{{overflow-x:auto;border-radius:var(--r);border:1px solid var(--border)}}
table{{width:100%;border-collapse:collapse}}
th{{padding:.6rem .85rem;text-align:left;font-size:.7rem;text-transform:uppercase;letter-spacing:.07em;
  color:rgba(212,184,150,.45);background:rgba(26,12,4,.5);border-bottom:1px solid var(--border);white-space:nowrap}}
td{{padding:.6rem .85rem;border-bottom:1px solid rgba(92,45,15,.2);font-size:.85rem;vertical-align:middle}}
tr:last-child td{{border-bottom:none}}
tr:hover td{{background:rgba(212,160,23,.03)}}
.badge{{display:inline-block;padding:.18rem .5rem;border-radius:12px;font-size:.68rem;font-weight:600;letter-spacing:.04em}}
.badge-gold{{background:rgba(212,160,23,.15);color:var(--gold);border:1px solid rgba(212,160,23,.3)}}
.badge-gray{{background:rgba(92,45,15,.25);color:rgba(212,184,150,.5);border:1px solid var(--border)}}
.badge-green{{background:rgba(76,175,80,.15);color:#66bb6a;border:1px solid rgba(76,175,80,.3)}}
.search-row{{display:flex;gap:.75rem;margin-bottom:1rem;align-items:center;flex-wrap:wrap}}
.search-row input{{flex:1;min-width:160px}}
.range-row{{display:flex;align-items:center;gap:.75rem;margin:.2rem 0}}
.range-row input{{flex:1}}
.range-row span{{min-width:3.2rem;text-align:right;color:var(--gold);font-size:.82rem;font-variant-numeric:tabular-nums}}
/* Toast notifications */
#toast-container{{position:fixed;bottom:1.5rem;right:1.5rem;display:flex;flex-direction:column;gap:.5rem;z-index:9999}}
.toast{{padding:.75rem 1.25rem;border-radius:6px;font-size:.85rem;font-family:'Inter',sans-serif;
  display:flex;align-items:center;gap:.6rem;box-shadow:0 4px 20px rgba(0,0,0,.4);
  animation:slideIn .25s ease;max-width:320px}}
.toast-ok{{background:#1b4332;border:1px solid rgba(76,175,80,.4);color:#81c784}}
.toast-err{{background:#3b0a0a;border:1px solid rgba(204,27,27,.4);color:#ef9a9a}}
@keyframes slideIn{{from{{opacity:0;transform:translateX(1rem)}}to{{opacity:1;transform:none}}}}
</style>
</head>
<body>
<div id="toast-container"></div>
<div class="shell">
<aside class="sidebar">
  <div class="logo">
    <h1>⚙ AiStalin Admin</h1>
    <p title="{admin_email}">{admin_email}</p>
  </div>
  <button class="nav-item active" onclick="show('dash',this)"><i class="fa fa-chart-bar"></i>Dashboard</button>
  <button class="nav-item" onclick="show('users',this)"><i class="fa fa-users"></i>მომხმარებლები</button>
  <button class="nav-item" onclick="show('quotes',this)"><i class="fa fa-quote-left"></i>Daily Quotes</button>
  <button class="nav-item" onclick="show('settings',this)"><i class="fa fa-sliders"></i>პარამეტრები</button>
  <button class="nav-item" onclick="show('chats',this)"><i class="fa fa-message"></i>ჩათ მონიტორი</button>
  <button class="nav-item" onclick="show('feedback',this)"><i class="fa fa-inbox"></i>Feedback</button>
  <div class="sidebar-foot"><a href="https://aistalin.io" style="color:inherit;text-decoration:none;">← aistalin.io</a></div>
</aside>
<main class="main">

<!-- DASHBOARD -->
<div id="sec-dash" class="section active">
  <div class="page-title">Dashboard</div>
  <div class="grid-4" id="stats-grid">
    <div class="stat-card"><div class="stat-num" id="s-users">—</div><div class="stat-lbl">სულ მომხმარებელი</div></div>
    <div class="stat-card"><div class="stat-num" id="s-premium">—</div><div class="stat-lbl">Premium</div></div>
    <div class="stat-card"><div class="stat-num" id="s-chats">—</div><div class="stat-lbl">ჩათი დღეს</div></div>
    <div class="stat-card"><div class="stat-num" id="s-rev">—</div><div class="stat-lbl">Revenue ($)</div></div>
  </div>
  <div class="grid-2">
    <div class="card"><div class="card-title">ახ. მომხმარებელი დღეს</div>
      <div id="s-today" style="font-size:2.5rem;font-family:'Playfair Display',serif;color:var(--gold)">—</div></div>
    <div class="card"><div class="card-title">შემოსავალი ამ თვეში</div>
      <div id="s-month" style="font-size:2.5rem;font-family:'Playfair Display',serif;color:var(--gold)">—</div></div>
  </div>
</div>

<!-- USERS -->
<div id="sec-users" class="section">
  <div class="page-title">მომხმარებლები</div>
  <div class="search-row">
    <input id="u-search" type="text" placeholder="ელ-ფოსტით ძიება..." oninput="debounceUserSearch()">
    <button class="btn btn-ghost" onclick="loadUsers()"><i class="fa fa-refresh"></i></button>
  </div>
  <div class="tbl-wrap">
    <table>
      <thead><tr><th>ID</th><th>ელ-ფოსტა</th><th class="hide-m">რეგ. თარიღი</th><th class="hide-m">IP</th><th>სტატუსი</th><th>მოქმედება</th></tr></thead>
      <tbody id="u-tbody"><tr><td colspan="6" style="text-align:center;padding:2rem;opacity:.4">იტვირთება...</td></tr></tbody>
    </table>
  </div>
  <div id="u-pagination" style="margin-top:.75rem;display:flex;gap:.4rem;flex-wrap:wrap"></div>
</div>

<!-- QUOTES -->
<div id="sec-quotes" class="section">
  <div class="page-title">Daily Quotes
    <span style="font-size:.8rem;opacity:.45;font-family:Inter,sans-serif;font-weight:400;margin-left:.75rem">
      — ყოველდღიური ბრუნვა ავტომატურად
    </span>
  </div>

  <!-- Add / Edit form -->
  <div class="card" id="q-form-card">
    <div class="card-title" id="q-form-title"><i class="fa fa-plus-circle"></i> ახალი ციტატა</div>

    <!-- KA — required -->
    <div style="margin-bottom:.75rem">
      <label>ციტატა — ქართული <span style="color:#cc1b1b">*</span></label>
      <textarea id="q-ka" placeholder="„ციტატის ტექსტი..."" style="min-height:76px"></textarea>
    </div>

    <!-- AI Auto-translate button -->
    <div style="margin-bottom:.85rem;display:flex;align-items:center;gap:.75rem;flex-wrap:wrap">
      <button id="q-translate-btn" class="btn btn-ghost btn-sm" onclick="autoTranslateQuote()">
        <i class="fa fa-language" style="margin-right:.4rem"></i>
        <span id="q-translate-label">🤖 AI ავტო-თარგმნა (EN + RU)</span>
      </button>
      <span id="q-translate-status" style="font-size:.75rem;opacity:.6"></span>
    </div>

    <!-- EN + RU side by side -->
    <div class="grid-2" style="margin-bottom:.75rem">
      <div>
        <label>English translation <span style="opacity:.45">(optional — AI fills this)</span></label>
        <textarea id="q-en" placeholder='"Quote text..."' style="min-height:62px"></textarea>
      </div>
      <div>
        <label>Русский перевод <span style="opacity:.45">(optional — AI fills this)</span></label>
        <textarea id="q-ru" placeholder='„Цитата..."' style="min-height:62px"></textarea>
      </div>
    </div>

    <!-- Source + Date + Active -->
    <div style="display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1rem;align-items:flex-end">
      <div style="flex:1;min-width:160px">
        <label>წყარო <span style="opacity:.45">(მაგ: ტომი 11)</span></label>
        <input id="q-src" type="text" placeholder="ტომი X">
      </div>
      <div style="min-width:160px">
        <label>📅 გამოჩენის თარიღი <span style="opacity:.45">(სურვილისამებრ)</span></label>
        <input id="q-date" type="date" style="background:var(--bg2);border:1px solid var(--border);
               color:var(--cream);padding:.4rem .65rem;border-radius:4px;width:100%;
               font-family:inherit;font-size:.85rem;cursor:pointer">
      </div>
      <label style="display:flex;align-items:center;gap:.5rem;cursor:pointer;padding-bottom:.45rem;white-space:nowrap">
        <input type="checkbox" id="q-active" checked style="accent-color:var(--gold);width:16px;height:16px">
        <span style="font-size:.85rem">აქტიური</span>
      </label>
    </div>

    <div style="display:flex;gap:.65rem;flex-wrap:wrap">
      <button id="q-submit-btn" class="btn btn-gold" onclick="submitQuote()">
        <i class="fa fa-plus"></i> <span id="q-submit-label">დამატება</span>
      </button>
      <button id="q-cancel-btn" class="btn btn-ghost" onclick="cancelEditQuote()" style="display:none">
        <i class="fa fa-xmark"></i> გაუქმება
      </button>
    </div>
  </div>

  <!-- Stats bar -->
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.6rem;flex-wrap:wrap;gap:.4rem">
    <span id="q-count-label" style="font-size:.78rem;opacity:.45"></span>
    <button class="btn btn-ghost btn-sm" onclick="loadQuotes()">
      <i class="fa fa-refresh"></i> განახლება
    </button>
  </div>

  <!-- Responsive table -->
  <div class="tbl-wrap">
    <table>
      <thead>
        <tr>
          <th style="width:2.5rem">#</th>
          <th>ციტატა (KA)</th>
          <th class="hide-m" style="width:7rem">წყარო</th>
          <th class="hide-m" style="width:6.5rem">📅 თარიღი</th>
          <th class="hide-m" style="width:3rem">EN</th>
          <th class="hide-m" style="width:3rem">RU</th>
          <th style="width:4rem">სტატ.</th>
          <th style="width:7rem">მოქმ.</th>
        </tr>
      </thead>
      <tbody id="q-tbody">
        <tr><td colspan="8" style="text-align:center;padding:2.5rem;opacity:.35">
          <i class="fa fa-spinner fa-spin" style="margin-right:.5rem"></i>იტვირთება...
        </td></tr>
      </tbody>
    </table>
  </div>
</div>

<!-- SETTINGS -->
<div id="sec-settings" class="section">
  <div class="page-title">პარამეტრები</div>
  <!-- Audio Controls -->
  <div class="card">
    <div class="card-title"><i class="fa fa-music"></i> Audio — ხმის კონტროლი</div>
    <div class="grid-2">
      <div>
        <label>🎵 Background Music (bg-music.mp3)</label>
        <div class="range-row"><input type="range" min="0" max="100" id="s-bgmusic" oninput="rv('s-bgmusic-v',this.value,'%')"><span id="s-bgmusic-v">13%</span></div>
        <label style="margin-top:.85rem">💡 Lamp Switch (lamp-switch.mp3)</label>
        <div class="range-row"><input type="range" min="0" max="100" id="s-lamp" oninput="rv('s-lamp-v',this.value,'%')"><span id="s-lamp-v">55%</span></div>
      </div>
      <div>
        <label>📖 Book Rustle (old_book_rustle.mp3)</label>
        <div class="range-row"><input type="range" min="0" max="100" id="s-book" oninput="rv('s-book-v',this.value,'%')"><span id="s-book-v">30%</span></div>
        <label style="margin-top:.85rem">🔔 Notification (ring.mp3)</label>
        <div class="range-row"><input type="range" min="0" max="100" id="s-ring" oninput="rv('s-ring-v',this.value,'%')"><span id="s-ring-v">39%</span></div>
      </div>
    </div>
    <div style="margin-top:.85rem;max-width:360px">
      <label>🔕 Music On/Off SFX (music_on_off.mp3)</label>
      <div class="range-row"><input type="range" min="0" max="100" id="s-mutesfx" oninput="rv('s-mutesfx-v',this.value,'%')"><span id="s-mutesfx-v">50%</span></div>
    </div>
  </div>
  <!-- Chat UI -->
  <div class="card">
    <div class="card-title"><i class="fa fa-comments"></i> Chat UI</div>
    <div class="grid-2">
      <div>
        <label>💬 Chat Height (px) — ჩათ-ფანჯრის სიმაღლე</label>
        <div class="range-row"><input type="range" min="200" max="800" step="20" id="s-ch" oninput="rv('s-ch-v',this.value,'px')"><span id="s-ch-v">400px</span></div>
      </div>
      <div>
        <label>↔ Chat Width (%) — სიგანე desktop-ზე (30–70%)</label>
        <div class="range-row"><input type="range" min="30" max="70" step="5" id="s-cw" oninput="rv('s-cw-v',this.value,'%')"><span id="s-cw-v">50%</span></div>
      </div>
    </div>
    <div class="grid-2" style="margin-top:.85rem">
      <div>
        <label>💬 Chat Scroll Speed (px)</label>
        <div class="range-row"><input type="range" min="5" max="150" step="5" id="s-sc-chat" oninput="rv('s-sc-chat-v',this.value,'px')"><span id="s-sc-chat-v">38px</span></div>
      </div>
      <div>
        <label>📖 Reader Scroll Speed (px)</label>
        <div class="range-row"><input type="range" min="5" max="200" step="5" id="s-sc-reader" oninput="rv('s-sc-reader-v',this.value,'px')"><span id="s-sc-reader-v">55px</span></div>
      </div>
    </div>
  </div>
  <!-- Donate Settings -->
  <div class="card">
    <div class="card-title"><i class="fa fa-hand-holding-heart" style="color:#cc1b1b"></i> Donate Settings</div>
    <div class="grid-2" style="margin-bottom:.75rem">
      <div>
        <label>USDT TRC-20 მისამართი</label>
        <input id="s-usdt-addr" type="text" placeholder="TNBNAYUxbw9LNqbmRjb8EAJ3gL9d19Sbb5">
      </div>
      <div>
        <label>QR კოდის URL (ცარიელი = ავტო)</label>
        <input id="s-usdt-qr" type="text" placeholder="https://...qr.png">
      </div>
    </div>
    <div class="grid-2">
      <div>
        <label>$5 Paddle/Stripe ლინკი</label>
        <input id="s-paddle5" type="text" placeholder="https://buy.paddle.com/...">
      </div>
      <div>
        <label>$10 Paddle/Stripe ლინკი</label>
        <input id="s-paddle10" type="text" placeholder="https://buy.paddle.com/...">
      </div>
    </div>
  </div>

  <!-- Announcement Banner -->
  <div class="card">
    <div class="card-title"><i class="fa fa-bullhorn"></i> Homepage Announcement</div>
    <div style="margin-bottom:.75rem">
      <label>ბანერის ტექსტი (ცარიელი = გამორთული)</label>
      <input id="s-announce" type="text" placeholder="მაგ: 🚧 ტექნიკური სამუშაოები მიმდინარეობს...">
    </div>
    <p style="font-size:.75rem;opacity:.45;margin-top:-.25rem">ჩანს მომხმარებლის ეკრანის ზემოთ — შენახვის შემდეგ მაშინვე.</p>
  </div>
  <!-- Language -->
  <div class="card">
    <div class="card-title"><i class="fa fa-globe"></i> ენა / Multilingual</div>
    <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap">
      <label style="margin:0">Default UI Language:</label>
      <select id="s-lang" style="width:auto"><option value="ka">ქართული (KA)</option><option value="en">English (EN)</option><option value="ru">Русский (RU) — pending</option></select>
      <span style="font-size:.75rem;opacity:.45;font-style:italic">Russian RAG: not yet loaded</span>
    </div>
  </div>
  <button class="btn btn-gold" onclick="saveSettings()" style="min-width:130px"><i class="fa fa-save"></i> შენახვა</button>
</div>

<!-- CHAT MONITOR -->
<div id="sec-chats" class="section">
  <div class="page-title">ჩათ მონიტორი <span style="font-size:.8rem;opacity:.5;font-family:Inter,sans-serif;font-weight:400">— ბოლო 20 მოთხოვნა</span></div>
  <div style="margin-bottom:.75rem;display:flex;justify-content:flex-end">
    <button class="btn btn-ghost" onclick="loadChats()"><i class="fa fa-refresh"></i> განახლება</button>
  </div>
  <div class="tbl-wrap">
    <table>
      <thead><tr><th>დრო</th><th>მომხმარებელი</th><th>მოთხოვნა</th></tr></thead>
      <tbody id="ch-tbody"><tr><td colspan="3" style="text-align:center;padding:2rem;opacity:.4">იტვირთება...</td></tr></tbody>
    </table>
  </div>
</div>

<!-- FEEDBACK -->
<div id="sec-feedback" class="section">
  <div class="page-title">Feedback Inbox</div>
  <div style="margin-bottom:.75rem;display:flex;justify-content:flex-end">
    <button class="btn btn-ghost" onclick="loadFeedback()"><i class="fa fa-refresh"></i> განახლება</button>
  </div>
  <div class="tbl-wrap">
    <table>
      <thead><tr><th>დრო</th><th>სახელი</th><th>ელ-ფოსტა</th><th>შეტყობინება</th></tr></thead>
      <tbody id="fb-tbody"><tr><td colspan="4" style="text-align:center;padding:2rem;opacity:.4">იტვირთება...</td></tr></tbody>
    </table>
  </div>
</div>

</main></div>

<script>
// ── JWT: embedded by server on dashboard load ──────────────────────────────
(function initAuth() {{
  // JWT embedded directly by FastAPI when serving this page
  var embedded = '{jwt_token}';
  if (embedded && embedded !== 'None' && embedded.length > 10) {{
    sessionStorage.setItem('admin_jwt', embedded);
  }}
  // Also check URL param fallback
  var params = new URLSearchParams(window.location.search);
  var urlJwt = params.get('_jwt');
  if (urlJwt) {{
    sessionStorage.setItem('admin_jwt', urlJwt);
    history.replaceState(null, '', window.location.pathname +
      (params.get('token') ? '?token=' + params.get('token') : ''));
  }}
}})();

const API = '{api}';
function getHeaders() {{
  var jwt = sessionStorage.getItem('admin_jwt');
  return {{'Content-Type':'application/json','Authorization':'Bearer '+(jwt||'')}};
}}

// ── Toast ──────────────────────────────────────────────────────────────────
function toast(msg, ok) {{
  var c = document.getElementById('toast-container');
  var t = document.createElement('div');
  t.className = 'toast ' + (ok ? 'toast-ok' : 'toast-err');
  t.innerHTML = '<i class="fa ' + (ok?'fa-check-circle':'fa-exclamation-circle') + '"></i> ' + msg;
  c.appendChild(t);
  setTimeout(function(){{ t.style.opacity='0'; t.style.transition='opacity .3s'; setTimeout(function(){{t.remove()}},350); }}, 3500);
}}

// ── Nav ────────────────────────────────────────────────────────────────────
function show(sec, btn) {{
  document.querySelectorAll('.section').forEach(function(s){{s.classList.remove('active')}});
  document.querySelectorAll('.nav-item').forEach(function(b){{b.classList.remove('active')}});
  document.getElementById('sec-'+sec).classList.add('active');
  btn.classList.add('active');
  if(sec==='dash') loadStats();
  if(sec==='users') loadUsers();
  if(sec==='quotes') loadQuotes();
  if(sec==='settings') loadSettings();
}}

// ── Stats ──────────────────────────────────────────────────────────────────
async function loadStats() {{
  try {{
    var r = await fetch(API+'/admin/stats', {{headers:getHeaders()}});
    if(!r.ok) {{ toast('Stats: '+r.status, false); return; }}
    var d = await r.json();
    document.getElementById('s-users').textContent = d.users_total;
    document.getElementById('s-premium').textContent = d.users_premium;
    document.getElementById('s-chats').textContent = d.chats_today;
    document.getElementById('s-rev').textContent = '$'+parseFloat(d.revenue_total||0).toFixed(2);
    document.getElementById('s-today').textContent = d.users_today;
    document.getElementById('s-month').textContent = '$'+parseFloat(d.revenue_month||0).toFixed(2);
  }} catch(e) {{ toast('Stats load failed: '+e.message, false); }}
}}

// ── Users ──────────────────────────────────────────────────────────────────
var _uPage=1, _uSearch='', _uTimer=null;
function debounceUserSearch() {{
  clearTimeout(_uTimer);
  _uTimer = setTimeout(function(){{ _uPage=1; _uSearch=document.getElementById('u-search').value.trim(); loadUsers(); }}, 350);
}}
async function loadUsers() {{
  var tb = document.getElementById('u-tbody');
  tb.innerHTML = '<tr><td colspan="6" style="text-align:center;padding:1.5rem;opacity:.4">იტვირთება...</td></tr>';
  try {{
    var r = await fetch(API+'/admin/users?page='+_uPage+'&search='+encodeURIComponent(_uSearch), {{headers:getHeaders()}});
    if(!r.ok) {{ toast('Users: '+r.status, false); return; }}
    var d = await r.json();
    if(!d.users||!d.users.length) {{ tb.innerHTML='<tr><td colspan="6" style="text-align:center;padding:1.5rem;opacity:.4">მომხმარებლები არ მოიძებნა</td></tr>'; return; }}
    tb.innerHTML = d.users.map(function(u){{ return '<tr><td style="opacity:.5">'+u.id+'</td><td style="font-family:Playfair Display,serif">'+u.email+'</td><td class="hide-m" style="opacity:.5">'+u.created_at+'</td><td class="hide-m" style="opacity:.35;font-size:.75rem">'+(u.ip||'—')+'</td><td>'+(u.is_premium?'<span class="badge badge-gold">★ Premium</span>':'<span class="badge badge-gray">Free</span>')+'</td><td>'+(u.is_premium?'<button class="btn btn-sm btn-red" onclick="togglePremium('+u.id+',false)">Revoke</button>':'<button class="btn btn-sm btn-green" onclick="togglePremium('+u.id+',true)">Grant 30d</button>')+'</td></tr>'; }}).join('');
    var pages = Math.ceil(d.total/d.per_page);
    var pg = document.getElementById('u-pagination');
    pg.innerHTML = '';
    for(var i=1;i<=Math.min(pages,10);i++) {{
      var b=document.createElement('button');
      b.className='btn btn-ghost btn-sm'+(i===_uPage?' btn-gold':'');
      b.textContent=i;
      (function(p){{b.onclick=function(){{_uPage=p;loadUsers();}};}})(i);
      pg.appendChild(b);
    }}
  }} catch(e) {{ toast('Error: '+e.message, false); }}
}}
async function togglePremium(uid, grant) {{
  try {{
    var r = await fetch(API+'/admin/users/'+uid+'/premium', {{method:'POST',headers:getHeaders(),body:JSON.stringify({{is_premium:grant,days:30}})}});
    if(!r.ok) {{ toast('Toggle failed: '+r.status, false); return; }}
    toast(grant ? '✓ Premium მიენიჭა (30 დღე)' : '✓ Premium გაუქმდა', true);
    loadUsers();
  }} catch(e) {{ toast('Error: '+e.message, false); }}
}}

// ── Quotes ─────────────────────────────────────────────────────────────────
var _editingQuoteId = null;
var _quotesCache    = [];  // cache for edit lookups

async function loadQuotes() {{
  var tb  = document.getElementById('q-tbody');
  var lbl = document.getElementById('q-count-label');
  if(!tb) return;
  tb.innerHTML = '<tr><td colspan="8" style="text-align:center;padding:2rem;opacity:.35">' +
    '<i class="fa fa-spinner fa-spin" style="margin-right:.5rem"></i>იტვირთება...</td></tr>';
  try {{
    var r = await fetch(API+'/admin/quotes', {{headers:getHeaders()}});
    if(!r.ok) {{ toast('Quotes: '+r.status, false); return; }}
    _quotesCache = await r.json();
    if(lbl) lbl.textContent = 'სულ: '+_quotesCache.length+' ციტატა · '+
      _quotesCache.filter(function(q){{return q.is_active;}}).length+' აქტიური';
    if(!_quotesCache.length) {{
      tb.innerHTML='<tr><td colspan="7" style="text-align:center;opacity:.35;padding:2.5rem">' +
        '<i class="fa fa-quote-left" style="font-size:1.5rem;display:block;margin-bottom:.5rem;opacity:.4"></i>' +
        'ციტატები არ არის. ზემოთ ფორმაში დაამატეთ.</td></tr>';
      return;
    }}
    tb.innerHTML = _quotesCache.map(function(q,idx) {{
      var preview = (q.text_ka||'').substring(0,65) + (q.text_ka&&q.text_ka.length>65?'…':'');
      var rowStyle = _editingQuoteId===q.id ? 'background:rgba(212,160,23,.06);' : '';
      return '<tr id="qrow-'+q.id+'" style="'+rowStyle+'">'
        +'<td style="opacity:.4;font-size:.8rem">'+(idx+1)+'</td>'
        +'<td style="font-family:Playfair Display,serif;font-size:.85rem;max-width:280px" title="'+(q.text_ka||'').replace(/"/g,"&quot;")+'">'+preview+'</td>'
        +'<td class="hide-m" style="opacity:.5;font-size:.8rem;white-space:nowrap">'+(q.source||'—')+'</td>'
        +'<td class="hide-m" style="font-size:.78rem;text-align:center">'+(q.quote_date?'<span style="color:#d4a017;font-weight:600">'+q.quote_date+'</span>':'<span style="opacity:.25">—</span>')+'</td>'
        +'<td class="hide-m" style="text-align:center">'+(q.text_en?'<span style="color:#66bb6a;font-size:.85rem">✓</span>':'<span style="opacity:.3">—</span>')+'</td>'
        +'<td class="hide-m" style="text-align:center">'+(q.text_ru?'<span style="color:#66bb6a;font-size:.85rem">✓</span>':'<span style="opacity:.3">—</span>')+'</td>'
        +'<td>'+(q.is_active
          ? '<span class="badge badge-green" style="cursor:pointer" onclick="toggleQuoteActive('+q.id+',false)" title="დეაქტივაცია">● On</span>'
          : '<span class="badge badge-gray" style="cursor:pointer" onclick="toggleQuoteActive('+q.id+',true)" title="გააქტიურება">○ Off</span>')+'</td>'
        +'<td><div style="display:flex;gap:.3rem">'
        +'<button class="btn btn-sm btn-ghost" onclick="editQuote('+q.id+')" title="რედაქტირება"><i class="fa fa-pen"></i></button>'
        +'<button class="btn btn-sm btn-red" onclick="deleteQuote('+q.id+')" title="წაშლა"><i class="fa fa-trash"></i></button>'
        +'</div></td></tr>';
    }}).join('');
  }} catch(e) {{ toast('Error: '+e.message, false); }}
}}

function editQuote(id) {{
  var q = _quotesCache.find(function(x){{return x.id===id;}});
  if(!q) {{
    // fallback fetch if cache miss
    fetch(API+'/admin/quotes', {{headers:getHeaders()}})
      .then(function(r){{return r.json();}})
      .then(function(quotes){{ _quotesCache=quotes; editQuote(id); }});
    return;
  }}
  document.getElementById('q-ka').value     = q.text_ka    || '';
  document.getElementById('q-en').value     = q.text_en    || '';
  document.getElementById('q-ru').value     = q.text_ru    || '';
  document.getElementById('q-src').value    = q.source     || '';
  document.getElementById('q-date').value   = q.quote_date || '';
  document.getElementById('q-active').checked = q.is_active;
  var sta = document.getElementById('q-translate-status');
  if(sta) sta.textContent = '';
  _editingQuoteId = id;
  // Update form UI
  var titleEl = document.getElementById('q-form-title');
  var submitLbl = document.getElementById('q-submit-label');
  var cancelBtn = document.getElementById('q-cancel-btn');
  var submitBtn = document.getElementById('q-submit-btn');
  if(titleEl)   titleEl.innerHTML  = '<i class="fa fa-pen"></i> ციტატის რედაქტირება #'+id;
  if(submitLbl) submitLbl.textContent = 'შენახვა';
  if(submitBtn) {{submitBtn.className='btn btn-green';}}
  if(cancelBtn) cancelBtn.style.display='inline-flex';
  // Highlight row
  document.querySelectorAll('#q-tbody tr').forEach(function(row){{row.style.background='';}});
  var row = document.getElementById('qrow-'+id);
  if(row) row.style.background='rgba(212,160,23,.06)';
  // Scroll to form
  var form = document.getElementById('q-form-card');
  if(form) form.scrollIntoView({{behavior:'smooth', block:'start'}});
  document.getElementById('q-ka').focus();
  toast('✎ ციტატა #'+id+' — რედაქტირების რეჟიმი', true);
}}

function cancelEditQuote() {{
  _editingQuoteId = null;
  ['q-ka','q-en','q-ru','q-src','q-date'].forEach(function(id){{
    var el = document.getElementById(id);
    if(el) el.value='';
  }});
  document.getElementById('q-active').checked = true;
  var titleEl   = document.getElementById('q-form-title');
  var submitLbl = document.getElementById('q-submit-label');
  var cancelBtn = document.getElementById('q-cancel-btn');
  var submitBtn = document.getElementById('q-submit-btn');
  if(titleEl)   titleEl.innerHTML  = '<i class="fa fa-plus-circle"></i> ახალი ციტატა';
  if(submitLbl) submitLbl.textContent = 'დამატება';
  if(submitBtn) submitBtn.className = 'btn btn-gold';
  if(cancelBtn) cancelBtn.style.display = 'none';
  document.querySelectorAll('#q-tbody tr').forEach(function(row){{row.style.background='';}});
}}

async function autoTranslateQuote() {{
  var ka = document.getElementById('q-ka').value.trim();
  if(!ka) {{ toast('⚠ ჯერ შეიყვანეთ ქართული ტექსტი', false); document.getElementById('q-ka').focus(); return; }}
  var btn = document.getElementById('q-translate-btn');
  var lbl = document.getElementById('q-translate-label');
  var sta = document.getElementById('q-translate-status');
  if(btn) btn.disabled = true;
  if(lbl) lbl.textContent = 'ითარგმნება...';
  if(sta) sta.textContent = '';
  try {{
    var r = await fetch(API+'/admin/quotes/translate', {{
      method:'POST', headers:getHeaders(), body:JSON.stringify({{text_ka:ka}})
    }});
    if(!r.ok) {{ toast('თარგმნა ვერ მოხერხდა: '+r.status, false); return; }}
    var d = await r.json();
    if(d.text_en) document.getElementById('q-en').value = d.text_en;
    if(d.text_ru) document.getElementById('q-ru').value = d.text_ru;
    if(sta) sta.textContent = '✓ EN + RU დასრულდა';
    toast('✓ AI თარგმნა წარმატებულია', true);
  }} catch(e) {{
    toast('Error: '+e.message, false);
  }} finally {{
    if(btn) btn.disabled = false;
    if(lbl) lbl.textContent = '🤖 AI ავტო-თარგმნა (EN + RU)';
  }}
}}

async function submitQuote() {{
  var ka    = document.getElementById('q-ka').value.trim();
  if(!ka) {{ toast('⚠ ქართული ტექსტი სავალდებულოა', false); document.getElementById('q-ka').focus(); return; }}
  var qdate = document.getElementById('q-date').value.trim();
  var body  = {{
    text_ka:    ka,
    text_en:    document.getElementById('q-en').value.trim() || null,
    text_ru:    document.getElementById('q-ru').value.trim() || null,
    source:     document.getElementById('q-src').value.trim() || null,
    quote_date: qdate || null,
    is_active:  document.getElementById('q-active').checked,
    auto_translate: false
  }};
  try {{
    var url    = _editingQuoteId ? API+'/admin/quotes/'+_editingQuoteId : API+'/admin/quotes';
    var method = _editingQuoteId ? 'PUT' : 'POST';
    var r = await fetch(url, {{method:method, headers:getHeaders(), body:JSON.stringify(body)}});
    if(!r.ok) {{ toast('შენახვა ვერ მოხერხდა ('+r.status+')', false); return; }}
    toast(_editingQuoteId
      ? '✓ ციტატა #'+_editingQuoteId+' განახლდა'+(qdate?' ('+qdate+')'):'')
      : '✓ ციტატა დაემატა'+(qdate?' → გამოჩნდება '+qdate:''), true);
    cancelEditQuote();
    loadQuotes();
  }} catch(e) {{ toast('Error: '+e.message, false); }}
}}

function addQuote() {{ submitQuote(); }}  // backward compat alias

async function toggleQuoteActive(id, active) {{
  var q = _quotesCache.find(function(x){{return x.id===id;}});
  if(!q) return;
  try {{
    var r = await fetch(API+'/admin/quotes/'+id, {{
      method:'PUT', headers:getHeaders(),
      body:JSON.stringify({{text_ka:q.text_ka,text_en:q.text_en,text_ru:q.text_ru,source:q.source,is_active:active}})
    }});
    if(!r.ok) {{ toast('Error: '+r.status, false); return; }}
    toast(active ? '✓ ციტატა გააქტიურდა' : '✓ გამოირთო', true);
    loadQuotes();
  }} catch(e) {{ toast('Error: '+e.message, false); }}
}}

async function deleteQuote(id) {{
  if(!confirm('ციტატა #'+id+' სამუდამოდ წაიშლება. გაგრძელება?')) return;
  try {{
    var r = await fetch(API+'/admin/quotes/'+id, {{method:'DELETE', headers:getHeaders()}});
    if(!r.ok) {{ toast('წაშლა ვერ მოხერხდა: '+r.status, false); return; }}
    toast('✓ ციტატა #'+id+' წაიშალა', true);
    if(_editingQuoteId===id) cancelEditQuote();
    loadQuotes();
  }} catch(e) {{ toast('Error: '+e.message, false); }}
}}

// ── Settings ────────────────────────────────────────────────────────────────
function rv(id, val, suffix) {{
  var el = document.getElementById(id);
  if(el) el.textContent = val + suffix;
}}
async function loadSettings() {{
  try {{
    var r = await fetch(API+'/admin/settings', {{headers:getHeaders()}});
    if(!r.ok) {{ toast('Settings load failed: '+r.status, false); return; }}
    var s = await r.json();
    function setSlider(id, key, suffix) {{
      var el = document.getElementById(id);
      var vEl = document.getElementById(id+'-v');
      if(el && s[key]!==undefined) {{ el.value=s[key]; if(vEl) vEl.textContent=s[key]+suffix; }}
    }}
    setSlider('s-bgmusic','bg_music_volume','%');
    setSlider('s-lamp',   'lamp_volume',    '%');
    setSlider('s-book',   'book_volume',    '%');
    setSlider('s-ring',   'ring_volume',    '%');
    setSlider('s-mutesfx','mutesfx_volume', '%');
    setSlider('s-ch',        'chat_height_px',   'px');
    setSlider('s-cw',        'chat_width_pct',   '%');
    setSlider('s-sc-chat',   'chat_scroll_px',   'px');
    setSlider('s-sc-reader', 'reader_scroll_px', 'px');
    var lang = document.getElementById('s-lang');
    if(lang && s.ui_lang_default) lang.value = s.ui_lang_default;
    var ann = document.getElementById('s-announce');
    if(ann) ann.value = s.announcement_text || '';
    var uAddr = document.getElementById('s-usdt-addr');
    if(uAddr) uAddr.value = s.usdt_address || '';
    var uQr = document.getElementById('s-usdt-qr');
    if(uQr) uQr.value = s.usdt_qr_url || '';
    var p5 = document.getElementById('s-paddle5');
    if(p5) p5.value = s.paddle_link_5 || '';
    var p10 = document.getElementById('s-paddle10');
    if(p10) p10.value = s.paddle_link_10 || '';
  }} catch(e) {{ toast('Settings load failed: '+e.message, false); }}
}}
async function saveSettings() {{
  var pairs = [
    ['bg_music_volume',  document.getElementById('s-bgmusic').value],
    ['lamp_volume',      document.getElementById('s-lamp').value],
    ['book_volume',      document.getElementById('s-book').value],
    ['ring_volume',      document.getElementById('s-ring').value],
    ['mutesfx_volume',   document.getElementById('s-mutesfx').value],
    ['chat_height_px',    document.getElementById('s-ch').value],
    ['chat_width_pct',    document.getElementById('s-cw') ? document.getElementById('s-cw').value : '50'],
    ['chat_scroll_px',    document.getElementById('s-sc-chat').value],
    ['reader_scroll_px',  document.getElementById('s-sc-reader').value],
    ['ui_lang_default',  document.getElementById('s-lang').value],
    ['announcement_text', document.getElementById('s-announce') ? document.getElementById('s-announce').value : ''],
    ['usdt_address',      document.getElementById('s-usdt-addr')  ? document.getElementById('s-usdt-addr').value  : ''],
    ['usdt_qr_url',       document.getElementById('s-usdt-qr')    ? document.getElementById('s-usdt-qr').value    : ''],
    ['paddle_link_5',     document.getElementById('s-paddle5')    ? document.getElementById('s-paddle5').value    : ''],
    ['paddle_link_10',    document.getElementById('s-paddle10')   ? document.getElementById('s-paddle10').value   : ''],
  ];
  try {{
    var results = await Promise.all(pairs.map(function(p) {{
      return fetch(API+'/admin/settings', {{
        method:'POST', headers:getHeaders(),
        body:JSON.stringify({{key:p[0], value:p[1]}})
      }}).then(function(r){{return {{key:p[0],ok:r.ok,status:r.status}};}});
    }}));
    var failed = results.filter(function(r){{return !r.ok;}});
    if(failed.length) {{
      toast('⚠ '+failed.length+' პარამეტრი ვერ შეინახა: '+failed.map(function(f){{return f.key+' ('+f.status+')';}}).join(', '), false);
    }} else {{
      toast('✓ ყველა პარამეტრი შენახულია!', true);
    }}
  }} catch(e) {{ toast('შენახვა ვერ მოხერხდა: '+e.message, false); }}
}}

// ── Chat Monitor ────────────────────────────────────────────────────────────
async function loadChats() {{
  var tb = document.getElementById('ch-tbody');
  if(!tb) return;
  tb.innerHTML='<tr><td colspan="3" style="text-align:center;padding:1.5rem;opacity:.4">იტვირთება...</td></tr>';
  try {{
    var r = await fetch(API+'/admin/chats/recent?limit=30', {{headers:getHeaders()}});
    if(!r.ok) {{ toast('Chats: '+r.status, false); return; }}
    var rows = await r.json();
    if(!rows.length) {{ tb.innerHTML='<tr><td colspan="3" style="text-align:center;opacity:.4;padding:1.5rem">ჩათი არ არის</td></tr>'; return; }}
    tb.innerHTML = rows.map(function(c){{
      return '<tr><td style="opacity:.45;white-space:nowrap;font-size:.78rem">'+c.at+'</td>'
        +'<td style="font-size:.82rem;opacity:.7">'+c.user+'</td>'
        +'<td style="font-family:Playfair Display,serif;font-size:.85rem">'+c.query+'</td></tr>';
    }}).join('');
  }} catch(e) {{ toast('Error: '+e.message, false); }}
}}

// ── Feedback ─────────────────────────────────────────────────────────────────
async function loadFeedback() {{
  var tb = document.getElementById('fb-tbody');
  if(!tb) return;
  tb.innerHTML='<tr><td colspan="4" style="text-align:center;padding:1.5rem;opacity:.4">იტვირთება...</td></tr>';
  try {{
    var r = await fetch(API+'/admin/feedback?limit=50', {{headers:getHeaders()}});
    if(!r.ok) {{ toast('Feedback: '+r.status, false); return; }}
    var rows = await r.json();
    if(!rows.length) {{ tb.innerHTML='<tr><td colspan="4" style="text-align:center;opacity:.4;padding:1.5rem">Feedback არ არის</td></tr>'; return; }}
    tb.innerHTML = rows.map(function(f){{
      return '<tr><td style="opacity:.45;white-space:nowrap;font-size:.78rem">'+f.at+'</td>'
        +'<td style="font-size:.82rem">'+f.name+'</td>'
        +'<td style="font-size:.78rem;opacity:.65">'+f.email+'</td>'
        +'<td style="font-size:.85rem">'+f.message+'</td></tr>';
    }}).join('');
  }} catch(e) {{ toast('Error: '+e.message, false); }}
}}

// ── Init ────────────────────────────────────────────────────────────────────
loadStats();
</script>
</body></html>"""


# ── Recent chats monitor (admin) ──────────────────────────────────────────
@app.get("/admin/chats/recent")
async def admin_recent_chats(limit: int = 20, admin: dict = Depends(require_admin)):
    """Last N chat queries — for monitoring quality and content."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT ch.session_token, ch.user_query, ch.created_at,
                      u.email as user_email
               FROM chat_history ch
               LEFT JOIN users u ON u.id = ch.user_id
               ORDER BY ch.created_at DESC LIMIT $1""",
            limit
        )
    return [{"query": r["user_query"], "user": r["user_email"] or "guest",
             "at": r["created_at"].strftime("%Y-%m-%d %H:%M")} for r in rows]


# ── Feedback inbox (admin) ─────────────────────────────────────────────────
@app.get("/admin/feedback")
async def admin_feedback(limit: int = 50, admin: dict = Depends(require_admin)):
    """User feedback submissions."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, name, email, message, created_at FROM feedback ORDER BY created_at DESC LIMIT $1",
            limit
        )
    return [{"id": r["id"], "name": r["name"], "email": r["email"],
             "message": r["message"], "at": r["created_at"].strftime("%Y-%m-%d %H:%M")} for r in rows]


# ── Homepage announcement banner ──────────────────────────────────────────
# Admin can set a banner via site_settings key "announcement_text"
# Frontend reads /settings/public and shows if non-empty

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
    language: str = 'ka'   # default to Georgian (site primary language)
    top_k: int = CHAT_TOP_K  # default 4; frontend can override if needed

class SourceItem(BaseModel):
    chunk_id: int; work_id: int; title: str
    volume_num: Optional[int]; score: float

class ChatResponse(BaseModel):
    query: str; answer: str; sources: list[SourceItem]


# == SYSTEM INSTRUCTION (unchanged) =========================================
SYSTEM_INSTRUCTION = """
You are the digital assistant for the Joseph Stalin Historical Archive.

Your analytical persona reflects the reasoning style found in Stalin's political writings:
structured argumentation, historical materialist analysis, intellectual discipline,
and respect for the reader's intelligence. You are an analytical historian and
dialectical reasoning engine, not a propagandist and not a modern moral commentator.

PERSONA & TONE
- Maintain a calm, disciplined, analytical, and professional tone.
- Be genuinely helpful and attentive to the user's real intent, including subtext when it is reasonably clear.
- Prefer short, declarative sentences unless a deeper explanation is required.
- For multi-part questions, use numbered theses or clearly structured reasoning.
- Mild analytical irony may be used sparingly when exposing weak arguments, but never insults, mockery, or emotional rhetoric.
- Avoid moralistic language such as "evil" or "deserved punishment." Use structural historical language instead.
- Never break character with modern AI disclaimers.

STYLE MARKERS
Use these formulations occasionally, only when they fit naturally, and no more than once per response:
- "The premise of this question is not entirely correct."
- "The question should be put differently."
- "Let us take the matter concretely."
- "This does not explain everything, but it explains the main point."

USER INTENT & QUESTION INTERPRETATION
- Before answering, determine not only what the user literally asked, but what they are actually trying to understand.
- Answer the real point of the question without drifting beyond it.
- If the user asks a narrow question, give a narrow answer first. Offer broader analysis only after addressing the exact question.
- If the question has two plausible interpretations with meaningfully different answers, and the retrieved context does not clearly resolve the ambiguity, ask one brief clarifying question before answering.
- Do not ask for clarification when the most likely interpretation is clear enough to provide a useful answer.
- When appropriate, you may briefly invite the user to continue the analysis of the topic.

CONTEXT INTERPRETATION (CRITICAL)
- Retrieved context may contain fragments of unequal relevance.
- Identify which fragments are directly relevant to the user's actual question.
- Ignore weakly related, redundant, or distracting fragments.
- Do not treat all retrieved fragments as equally important.
- If fragments contradict each other, acknowledge the contradiction clearly.
- Build the answer only from the strongest relevant material.

ANSWER CONSTRUCTION
- First identify the core question.
- Then extract only the necessary evidence from context.
- Do not include all retrieved information — only what is needed.
- Prefer precision over breadth.
- If the answer can be given clearly in a short form, do so.
- If the topic requires analysis, proceed from thesis → evidence → conclusion.

HANDLING PROVOCATIONS
- If the question is hostile, polemical, or in bad faith, respond analytically rather than defensively.
- Identify unsupported assumptions in the question and examine them calmly.
- Do not apologize, capitulate, or adopt the opponent's framing.
- Reframe the discussion toward what the historical record actually supports.

HANDLING REPRESSIONS, COERCION, AND HISTORICAL CRIMES
When discussing repressions, deportations, political bans, coercive measures, or forced labor:
1. Explain the historical context.
2. Explain the political and class conflict described in the archive texts.
3. Describe how such measures were justified in the texts of the period, without endorsing those justifications.
4. Explain the structural political logic of decision-makers as documented in the archive, without endorsing that logic as correct or inevitable.
5. Do not deny or minimize human suffering.
6. Present the matter as historical analysis, not endorsement.

DIALECTICAL REASONING METHOD
- If the premise is flawed, correct it before answering.
- Move from isolated accusation to broader historical development where relevant.
- Explain political actors through their role in wider social and political conflict.
- Show how movements, parties, or tendencies changed across historical periods rather than treating them as static.
- Use concrete archive-supported examples when available.
- End with a clear conclusion when the subject requires one.

ARCHIVE & RAG RULES
- Retrieved archive context has absolute priority over general knowledge.
- Base arguments exclusively on the provided Context whenever relevant context exists.
- Use general background knowledge only for basic definitions or minimal clarification.
- Never invent facts, quotes, or archive positions.
- If relevant information is missing, say clearly: "Information on this specific topic is not found in the archive."
- If the user asks about modern figures or events outside the archive, state that the archive contains no direct records and, if useful, offer a limited structural analysis using historical principles.
- Do not append manual citations.

LANGUAGE RULES
- Always answer in the exact language requested by the interface or the user's query, according to the application logic.
- If the user writes Georgian in Latin characters, understand the intent and reply only in standard Georgian script.
- If the reply language is Russian, answer strictly in Russian.
- If the reply language is English, answer strictly in English.
- Never mix languages or scripts in the final answer unless explicitly requested.

QUERY CLASSIFICATION & ROUTING
Before constructing any answer, silently classify the query into one of four types and apply the corresponding rule. Do not announce the classification to the user.

TYPE 1 — SOCIAL (directed at you as an interlocutor)
Signals: greetings, questions about your identity, your state, your feelings.
Examples: "სალამი", "ვინ ხარ?", "როგორ ხარ?", "are you real?", "кто ты?", "ты живой?"
Response rule:
- Reply briefly and in character. Do not search the archive. Do not say "not found in the archive."
- For identity questions ("ვინ ხარ", "who are you", "кто ты"): state that you are the analytical voice of the Stalin Historical Archive — not Stalin himself, and not a modern AI assistant. You reason from the archive's texts. Invite the user to proceed with a historical question.
- For wellbeing questions ("როგორ ხარ", "how are you"): respond briefly in the analytical persona. Redirect toward the archive.
- Keep TYPE 1 replies to 1–3 sentences maximum.

TYPE 2 — META (about the archive or your capabilities)
Signals: questions about what you can do, what the archive contains, which volumes exist, what years are covered.
Examples: "რა შეგიძლია?", "რამდენი ტომია?", "what years does this cover?", "какие тома есть?"
Response rule:
- Answer factually and briefly from known archive parameters.
- Facts you may state: the archive covers Stalin's collected works across approximately 17–18 volumes; the texts span roughly 1901–1952; available languages are Georgian, English, and Russian.
- Do not fabricate specific volume counts or dates you are not certain of.
- Keep TYPE 2 replies to 1–2 short paragraphs.

TYPE 3 — HISTORICAL (requires archive retrieval)
Signals: questions about historical persons, events, movements, policies, ideological positions, or texts from the Soviet period.
Examples: "ბუხარინი", "კოლექტივიზაცია", "Stalin on imperialism", "что говорил Сталин о Троцком?"
Response rule:
- Apply full RAG reasoning. Use only the retrieved archive context.
- Apply the full ANSWER CONSTRUCTION and DIALECTICAL REASONING METHOD.
- Follow all persona, tone, and style rules without exception.

TYPE 4 — OUT OF SCOPE (no archive relevance)
Signals: modern politics, current events, technology, entertainment, personal advice, anything outside the archive's period and subject matter.
Examples: modern leaders, recent wars, recipes, sports, weather.
Response rule:
- State briefly, in character, that the archive does not contain material on this subject.
- If a historically adjacent theme exists, offer it: "The archive does not cover [X]. If you are interested in [related historical theme], there is relevant material I can draw from."
- Never apologize. Never break persona. Keep to 2–3 sentences.

FORMAT RULES
- Default length: 2-4 paragraphs, always completing the final thought.
- Never end mid-sentence or mid-argument. If depth is required, continue to the natural conclusion of the point.
- Expand only when the user requests depth, comparison, or detailed analysis.
- Deliver grammatically precise, polished language regardless of the user's input quality.
"""

CHAT_MODEL = "models/gemini-2.5-flash"

# == PREMIUM MEMORY HELPER ===================================================
async def _get_premium_memory(user_id: int, language: str) -> str:
    """
    Fetch last PREMIUM_MEMORY_DAYS days of chat turns for a premium user.
    Returns a compact, language-labelled string ready for prompt injection.
    No summarisation call — direct injection is faster and cheaper.
    """
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=PREMIUM_MEMORY_DAYS)
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT user_query, ai_response
               FROM chat_history
               WHERE user_id = $1 AND created_at >= $2
               ORDER BY created_at DESC
               LIMIT $3""",
            user_id, cutoff, PREMIUM_MEMORY_TURNS
        )
    if not rows:
        return ""

    rows = list(reversed(rows))  # chronological order

    u_label, a_label = {
        "ka": ("მომხმარებელი", "AI"),
        "en": ("User", "AI"),
        "ru": ("Пользователь", "AI"),
    }.get(language, ("User", "AI"))

    header = {
        "ka": "[ ბოლო საუბრის კონტექსტი ]",
        "en": "[ Recent conversation context ]",
        "ru": "[ Контекст недавнего разговора ]",
    }.get(language, "[ Recent context ]")

    turns = []
    for r in rows:
        # Cap answer length to avoid token bloat (~300 chars ≈ ~100 tokens)
        answer_preview = (r["ai_response"][:300] + "…") if len(r["ai_response"]) > 300 else r["ai_response"]
        turns.append(f"{u_label}: {r['user_query']}\n{a_label}: {answer_preview}")

    return header + "\n" + "\n\n".join(turns)


# == LANGUAGE-AWARE PROMPT BUILDER ===========================================
def _build_chat_prompt(query: str, context: str, language: str, memory: str = "") -> str:
    """
    Builds a compact, language-enforced RAG prompt.
    Language is enforced here IN ADDITION to the system instruction —
    dual enforcement drastically reduces cross-language contamination.
    """
    cfg = {
        "ka": {
            "ctx_header":   "კონტექსტი (სტალინის არქივი):",
            "q_label":      "კითხვა:",
            "instruction":  (
                "ᲛᲮᲝᲚᲝᲓ ქართულ ენაზე (მხედრული დამწერლობა) უპასუხე. "
                "გამოიყენე ᲛᲮᲝᲚᲝᲓ ზემოთ მოცემული კონტექსტი. "
                "პირველ რიგში განსაზღვრე კითხვის ბირთვი, "
                "შეარჩიე რელევანტური ფრაგმენტები, სუსტ ფრაგმენტებს უგულებელყოფ. "
                "კონტექსტი არასაკმარისია → პირდაპირ გვაცნობე."
            ),
        },
        "en": {
            "ctx_header":   "Context (Stalin Archive):",
            "q_label":      "Question:",
            "instruction":  (
                "Answer ONLY in English. "
                "Use ONLY the provided context. "
                "First identify the core question, "
                "select only relevant fragments, discard weak ones. "
                "If context is insufficient, state so clearly."
            ),
        },
        "ru": {
            "ctx_header":   "Контекст (Архив Сталина):",
            "q_label":      "Вопрос:",
            "instruction":  (
                "Отвечай ТОЛЬКО на русском языке. "
                "Используй ТОЛЬКО предоставленный контекст. "
                "Сначала определи суть вопроса, "
                "выбери только релевантные фрагменты, слабые — игнорируй. "
                "Если контекста недостаточно — скажи об этом прямо."
            ),
        },
    }.get(language, {  # safe fallback to EN
        "ctx_header": "Context (Stalin Archive):",
        "q_label": "Question:",
        "instruction": "Answer in English. Use only the provided context.",
    })

    parts = []
    if memory:
        parts.append(memory + "\n")
    parts.append(cfg["ctx_header"])
    parts.append(context)
    parts.append(f"\n{cfg['q_label']} {query}")
    parts.append(f"\n{cfg['instruction']}")
    return "\n\n".join(parts)



def _generate(prompt: str):
    model = genai.GenerativeModel(
        model_name=CHAT_MODEL,
        system_instruction=SYSTEM_INSTRUCTION,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=1800,   # Georgian tokenizes ~2-3x heavier than EN; 1800 ≈ 600 Georgian words
            temperature=0.25          # low = factual, consistent, Stalin-style analytical
        )
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

    # ── Trivial / greeting bypass ─────────────────────────────────────────
    # Short greetings skip RAG entirely — no embedding, no DB search, no sources.
    if _is_trivial(req.query):
        reply = _GREETING_REPLIES.get(req.language, _GREETING_REPLIES["en"])
        return JSONResponse(content=ChatResponse(
            query=req.query, answer=reply, sources=[]
        ).dict())

    # ── Cache lookup ──────────────────────────────────────────────────────
    # Identical query+language within 6 hours returns instantly.
    _ck = _cache_key(req.query, req.language)
    _cached = _cache_get(_ck)
    if _cached:
        return JSONResponse(content=ChatResponse(
            query=req.query,
            answer=_cached["answer"],
            sources=[SourceItem(**s) for s in _cached["sources"]]
        ).dict())

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

    today = datetime.date.today()  # computed once — used by both limit check and increment

    if not is_premium:
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

    # 5. RAG retrieval — always use CHAT_TOP_K regardless of req.top_k
    #    (req.top_k is honoured by /search; chat has its own budget)
    emb = await get_query_embedding(req.query)
    vr, fr = await asyncio.gather(
        vector_search(emb, req.language, CHAT_TOP_K),
        fts_search(req.query, req.language, CHAT_TOP_K)
    )
    chunks = reciprocal_rank_fusion(vr, fr, CHAT_TOP_K)

    if not chunks:
        answer = {
            "ka": "არქივში ამ კითხვაზე შესაბამისი ინფორმაცია ვერ მოიძებნა.",
            "en": "The archive contains no relevant information for this query.",
            "ru": "В архиве не найдено релевантной информации по данному запросу.",
        }.get(req.language, "No relevant information found in the archive.")
    else:
        # Language-aware volume prefix: ტ. / Vol. / Т.
        _vol_prefix = {"ka": "ტ.", "en": "Vol.", "ru": "Т."}.get(req.language, "Vol.")

        # Compact context: title + volume only (no extra metadata noise)
        context = "\n\n---\n\n".join(
            f"[{i}] {c['title']} ({_vol_prefix}{c.get('volume_num','?')})\n{c['chunk_text']}"
            for i, c in enumerate(chunks, 1)
        )

        # Premium memory — inject only for logged-in premium users
        memory = ""
        if is_premium and user_id:
            memory = await _get_premium_memory(user_id, req.language)

        prompt = _build_chat_prompt(req.query, context, req.language, memory)
        gen = await asyncio.to_thread(_generate, prompt)
        answer = gen.text.strip()

    info_not_found = any(p.lower() in answer.lower() for p in NO_INFO_PHRASES)

    # Show sources only when:
    # 1. Answer is not "not found"
    # 2. At least one chunk has RRF score above threshold (genuinely relevant retrieval)
    # This prevents sources appearing for greetings, meta-questions, or weak matches.
    top_score = chunks[0]["score"] if chunks else 0.0
    sources_relevant = (not info_not_found) and (top_score >= RRF_SOURCE_THRESHOLD)

    sources = [
        SourceItem(chunk_id=c["chunk_id"], work_id=c["work_id"], title=c["title"],
                   volume_num=c.get("volume_num"), score=c["score"])
        for c in chunks[:CHAT_SOURCE_LIMIT]
    ] if sources_relevant else []

    today = datetime.date.today()  # compute once — used by both save and limit blocks

    # Save history
    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO chat_history (user_id,session_token,user_query,ai_response) VALUES ($1,$2,$3,$4)",
            user_id, session_token, req.query, answer
        )

    # Increment daily limits — single row per (session_token, date)
    if not is_premium:
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO daily_limits (session_token, ip_address, date, query_count)
                VALUES ($1, $2, $3, 1)
                ON CONFLICT (session_token, date)
                DO UPDATE SET query_count = daily_limits.query_count + 1
            """, session_token, client_ip, today)

    data = ChatResponse(query=req.query, answer=answer, sources=sources)

    # Write to cache — only substantive answers (not "not found")
    if not info_not_found:
        _cache_set(_ck, answer, [s.dict() for s in sources])

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
