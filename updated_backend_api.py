import sqlite3
import re
import logging
from pathlib import Path
import ollama
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="NL to SQL Chatbot API")

# Enable CORS for Streamlit front‑end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    db_path: str
    user_question: str
    db_schema: str = "" # Optional

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

OLLAMA_MODEL   = "llama3.1"
MAX_RETRIES    = 3
MAX_ROWS       = 500          # cap result rows to avoid flooding UI
DB_TIMEOUT     = 10.0         # seconds before query times out

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# BANNED SQL KEYWORDS  (read-only enforcement)
# ─────────────────────────────────────────────

WRITE_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
    "TRUNCATE", "REPLACE", "ATTACH", "DETACH", "PRAGMA",
    "BEGIN", "COMMIT", "ROLLBACK", "VACUUM", "REINDEX",
    "GRANT", "REVOKE", "SAVEPOINT", "RELEASE",
]

DANGEROUS_PATTERNS = [
    r";\s*(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)",   # SQL injection via semicolon
    r"--\s*",                                          # inline SQL comment tricks
    r"/\*.*?\*/",                                      # block comments
    r"UNION\s+ALL\s+SELECT.*FROM\s+sqlite_",           # schema dump via UNION
]

# Natural language phrases that indicate a write / mutation intent.
WRITE_REQUEST_PATTERNS = [
    r"\b(delete|remove|erase|wipe|purge|drop|destroy|clear)\b",
    r"\b(insert|add|create|put|push|append|register|enter)\b",
    r"\b(update|edit|change|modify|set|rename|replace|patch|fix|correct)\b",
    r"\b(truncate|reset|flush|bulk.?delete|bulk.?update)\b",
    r"\b(make\s+.{0,30}\s+(active|inactive|status|available|unavailable))\b",
    r"\b(mark\s+.{0,30}\s+as)\b",
    r"\b(assign|transfer|move|migrate|merge|duplicate|copy\s+to)\b",
    r"\b(increment|decrement|increase|decrease|raise|lower)\s+.{0,20}\s+(by|to)\b",
]

# ─────────────────────────────────────────────
# SCHEMA EXTRACTION
# ─────────────────────────────────────────────

def get_schema(db_path: str) -> str:
    """
    Extract full schema from SQLite DB using read-only URI.
    Returns a clean human-readable schema string for the LLM prompt.
    """
    uri = f"file:{db_path}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=DB_TIMEOUT)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row for row in cursor.fetchall()]

        if not tables:
            return "No tables found in database."

        schema_parts = []
        for table in tables:
            cursor.execute(f"PRAGMA table_info('{table}')")
            columns = cursor.fetchall()
            col_lines = []
            for col in columns:
                col_id, name, dtype, notnull, default, pk = col
                parts = [f"  {name} {dtype}"]
                if pk:
                    parts.append("PRIMARY KEY")
                if notnull:
                    parts.append("NOT NULL")
                if default is not None:
                    parts.append(f"DEFAULT {default}")
                col_lines.append(" ".join(parts))

            try:
                cursor.execute(f"SELECT COUNT(*) FROM '{table}'")
                row_count = cursor.fetchone()
                count_hint = f"  -- {row_count} rows"
            except Exception:
                count_hint = ""

            schema_parts.append(
                f"Table: {table}{count_hint}\n"
                f"Columns:\n" + "\n".join(col_lines)
            )
        conn.close()
        return "\n\n".join(schema_parts)
    except sqlite3.OperationalError as e:
        log.error(f"Schema extraction failed: {e}")
        raise RuntimeError(f"Cannot read database: {e}")

# ─────────────────────────────────────────────
# SQL SAFETY VALIDATION
# ─────────────────────────────────────────────

def validate_sql_safety(sql: str) -> tuple[bool, str]:
    """
    Multi-layer SQL safety check. Returns (is_safe, reason).
    Blocks anything that isn't a pure SELECT.
    """
    sql_upper = sql.upper().strip()

    # Layer 1: Must start with SELECT
    if not re.match(r"^\s*SELECT\b", sql_upper):
        return False, f"Query must start with SELECT. Got: '{sql[:40]}...'"

    # Layer 2: Banned write keywords
    for keyword in WRITE_KEYWORDS:
        if re.search(rf"\b{keyword}\b", sql_upper):
            return False, f"Forbidden keyword detected: {keyword}"

    # Layer 3: Dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, sql, re.IGNORECASE | re.DOTALL):
            return False, f"Dangerous SQL pattern detected: {pattern}"

    # Layer 4: Multiple statements (semicolon separation)
    stripped = sql.rstrip().rstrip(";")
    if ";" in stripped:
        return False, "Multiple SQL statements are not allowed."

    return True, "OK"

# ─────────────────────────────────────────────
# SQL EXECUTION (read-only URI)
# ─────────────────────────────────────────────

def execute_sql(db_path: str, sql: str) -> tuple[list[str], list[tuple]]:
    """
    Execute a SELECT query using strict read-only SQLite URI.
    Returns (column_names, rows).
    """
    sql_clean = sql.strip().rstrip(";")
    is_safe, reason = validate_sql_safety(sql_clean)
    if not is_safe:
        raise ValueError(f"SQL safety check failed: {reason}")

    uri = f"file:{db_path}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=DB_TIMEOUT)
        conn.set_authorizer(_read_only_authorizer)   # extra DB-level authorizer
        cursor = conn.cursor()
        cursor.execute(sql_clean)
        rows = cursor.fetchmany(MAX_ROWS)
        cols = [description for description in cursor.description]
        conn.close()
        return cols, rows
    except sqlite3.OperationalError as e:
        raise ValueError(f"SQL execution error: {e}")

def _read_only_authorizer(action_code, arg1, arg2, db_name, trigger):
    """
    SQLite authorizer callback — third line of defense.
    Allows only SELECT and READ operations.
    """
    ALLOWED = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,   # allow built-in functions like COUNT, SUM
    }
    if action_code in ALLOWED:
        return sqlite3.SQLITE_OK
    log.warning(f"Authorizer blocked action: {action_code} on {arg1}.{arg2}")
    return sqlite3.SQLITE_DENY

# ─────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────

FEW_SHOT_EXAMPLES = """
Examples:
Q: How many users are there?
A: SELECT COUNT(*) AS total_users FROM users;

Q: Show me the top 5 products by price
A: SELECT name, price FROM products ORDER BY price DESC LIMIT 5;

Q: What are the orders placed in the last 7 days?
A: SELECT * FROM orders WHERE order_date >= DATE('now', '-7 days');

Q: List customers who have never placed an order
A: SELECT c.* FROM customers c LEFT JOIN orders o ON c.id = o.customer_id WHERE o.id IS NULL;
"""

def build_system_prompt(schema: str) -> str:
    return f"""You are an expert SQLite SQL assistant. Your ONLY job is to convert natural language questions into valid SQLite SELECT queries.

DATABASE SCHEMA:
{schema}

{FEW_SHOT_EXAMPLES}

STRICT RULES — follow every rule, no exceptions:
1. Return ONLY the raw SQL query — no markdown, no backticks, no explanation, no preamble
2. ONLY write SELECT statements — never INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, PRAGMA, TRUNCATE, REPLACE, ATTACH, DETACH, SAVEPOINT, RELEASE,or any write operation
3. Use exact table and column names from the schema above
4. Always end the query with a semicolon
5. If the question is ambiguous, ask the user for more information
6. If the question cannot be answered with the available schema, respond with exactly: CANNOT_ANSWER
7. Never use subqueries that modify data
8. Never use ATTACH, DETACH, or access sqlite_master directly
9. THE final answer should be in proper english with correct meaning 
10. IF the user asks what is in the database, what data is available, or what tables exist, DO NOT write SQL. Read the table names from the schema and respond EXACTLY in this format:
META: The database contains company data. Available tables are: [insert table names]. Please ask detailed questions."""

def build_retry_prompt(original_question: str, failed_sql: str, error: str) -> str:
    return f"""The previous SQL query failed. Fix it.

Original question: {original_question}
Failed SQL: {failed_sql}
Error: {error}

Return ONLY the corrected SQL query. No explanation."""

# ─────────────────────────────────────────────
# LLM INTERACTION
# ─────────────────────────────────────────────

def call_ollama(system_prompt: str, user_message: str) -> str:
    """Call Ollama llama3.1 and return the raw text response."""
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            options={
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 512,
            }
        )
        return response["message"]["content"].strip()
    except Exception as e:
        log.error(f"Ollama error: {e}")
        raise RuntimeError(
            f"Cannot reach Ollama. Make sure it's running: `ollama serve`\nError: {e}"
        )

def clean_llm_sql_output(raw: str) -> str:
    """Strip any markdown or wrapper the LLM accidentally adds."""
    cleaned = re.sub(r"