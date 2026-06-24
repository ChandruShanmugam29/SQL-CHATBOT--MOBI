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
    chat_history: list = [] # Optional chat history for context

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
        # Extract the table name string from each tuple
        tables = [row[0] for row in cursor.fetchall()]

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
                row_count = cursor.fetchone()[0]
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
META: The database contains company data. Available tables are: [insert table names]. Please ask detailed questions.
11. IF the user asks a general question or greeting (like 'hi', 'who are you', etc) that does not need a database query, respond EXACTLY in this format:
META: Hello! I am your database assistant. Please ask me questions about the data."""

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
    # Fixed the regex to avoid using actual triple backticks so it doesn't break the formatting!
    cleaned = re.sub(r"`{3}(?:sql)?\s*([\s\S]*?)`{3}", r"\1", raw, flags=re.IGNORECASE)
    cleaned = cleaned.strip("` \n")
    return cleaned

def generate_natural_answer(question: str, sql: str, columns: list, rows: list) -> str:
    """
    Generate a natural language response in proper English using Ollama.
    """
    if not rows:
        return "No results found matching your query."
    
    # Format a clean representation of the rows
    data_summary = f"Columns: {', '.join(columns)}\nRows:\n"
    for row in rows[:100]:
        data_summary += f"- {row}\n"
    if len(rows) > 100:
        data_summary += f"... (and {len(rows) - 100} more rows)\n"
        
    system_prompt = (
        "You are an expert SQLite assistant. Your job is to answer the user's question "
        "by formatting the database results into a clear, proper English response with correct meaning.\n"
        "RULES:\n"
        "1. Do NOT explain the SQL query, write any code, or show any raw JSON.\n"
        "2. If the results contain multiple records, format them nicely using a numbered list, "
        "bullet points, or a markdown table so it looks highly readable in the chat bubble.\n"
        "3. Make sure to display the full list of names/records found in the database results. "
        "Do NOT truncate the list unless there are more than 100 rows.\n"
        "4. Use single newlines for lists (do NOT leave blank lines or empty lines between items) to keep the text compact and professional."
    )
    
    user_message = f"""User Question: {question}
Executed SQL: {sql}
Database Results:
{data_summary}

Please provide the direct English answer below:"""
    
    try:
        answer = call_ollama(system_prompt, user_message)
        return answer
    except Exception as e:
        log.error(f"Error generating natural answer: {e}")
        return f"Executed query returned {len(rows)} rows."

# ─────────────────────────────────────────────
# ROUTER, REPHRASER & GENERAL CHAT LOGIC
# ─────────────────────────────────────────────

def rephrase_question(user_question: str, chat_history: list) -> str:
    """Uses LLM to rephrase a follow-up question based on conversation history."""
    if not chat_history:
        return user_question
        
    history_text = "\n".join([f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}" for msg in chat_history])
    prompt = f"""You are a question rephraser.
Given the following conversation history and a follow-up question, rephrase the follow-up question to be a complete, standalone question.
If the follow-up question is already standalone and clear, return it exactly as is.
DO NOT answer the question. JUST rephrase it.

History:
{history_text}

Follow-up Question: {user_question}

Standalone Question:"""
    
    try:
        response = call_ollama(prompt, "")
        return response.strip()
    except Exception as e:
        log.error(f"Rephrase error: {e}")
        return user_question

def classify_intent(user_question: str) -> str:
    """Step 1 Router: Decide if the user wants general chat or database data."""
    router_prompt = """You are an intent classifier.
    Classify the user's input into TWO categories:
    1. 'GENERAL' - Greetings (hi, hello), general chat, or asking what you do.
    2. 'SQL' - Asking for specific data, counts, company details, or what tables exist.
    Reply with EXACTLY ONE WORD: either GENERAL or SQL."""
    try:
        response = call_ollama(router_prompt, user_question)
        if "SQL" in response.upper():
            return "SQL"
        return "GENERAL"
    except Exception:
        return "SQL" # Default fallback

def handle_general_chat(user_question: str) -> str:
    """General conversation handler without database access."""
    chat_prompt = """You are a helpful Database Assistant. 
    Keep your answer polite, short, and friendly (1-2 sentences max). 
    Remind them they can ask you to fetch data from the database."""
    return call_ollama(chat_prompt, user_question)

# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/status")
async def status_endpoint():
    """Health‑check endpoint."""
    return {"ok": True, "message": "service running"}

@app.get("/schema")
async def schema_endpoint(db_path: str):
    """Return the SQLite schema for the requested database file."""
    try:
        return {"schema": get_schema(db_path)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """Process a natural‑language question, generate SQL via Ollama, execute it, and return results."""
    log.info(f"Received question: {request.user_question}")
    
    # 0. REPHRASE QUESTION USING HISTORY
    actual_question = request.user_question
    if request.chat_history:
        actual_question = rephrase_question(request.user_question, request.chat_history)
        log.info(f"Rephrased question: {actual_question}")
    
    # 1. INTENT ROUTING
    intent = classify_intent(actual_question)
    log.info(f"Intent Detected: {intent}")
    
    # 2. GENERAL CHAT BRANCH
    if intent == "GENERAL":
        chat_response = handle_general_chat(actual_question)
        return {
            "sql": None, 
            "columns": [], 
            "rows": [],
            "natural_answer": chat_response, 
            "error": None, 
            "attempts": 1
        }

    # 3. SQL BRANCH
    schema_str = get_schema(request.db_path)
    system_prompt = build_system_prompt(schema_str)

    # Get raw SQL from LLM
    raw_sql = call_ollama(system_prompt, actual_question)
    
    # Check for Privacy/Outline response (META tag)
    if raw_sql.startswith("META:"):
        safe_answer = raw_sql.replace("META:", "").strip()
        return {
            "sql": None, 
            "columns": [], 
            "rows": [],
            "natural_answer": safe_answer, 
            "error": None, 
            "attempts": 1
        }

    sql = clean_llm_sql_output(raw_sql)

    # Check if LLM determined it cannot answer
    if "CANNOT_ANSWER" in sql.upper():
        return {
            "sql": None,
            "columns": [],
            "rows": [],
            "natural_answer": "I cannot answer this question based on the database schema provided.",
            "error": "I cannot answer this question based on the database schema provided.",
            "attempts": 1,
        }

    # Validate safety
    safe, reason = validate_sql_safety(sql)
    if not safe:
        raise HTTPException(status_code=400, detail=f"SQL safety check failed: {reason}")

    # Execute query
    try:
        cols, rows = execute_sql(request.db_path, sql)
        # cursor.description returns tuples like (name, type_code, ...); extract just the name
        column_names = [c[0] for c in cols] if cols else []
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"SQL execution error: {e}")

    # Generate natural language response using query results
    natural_answer = generate_natural_answer(request.user_question, sql, column_names, rows)

    return {
        "sql": sql,
        "columns": column_names,
        "rows": rows,
        "natural_answer": natural_answer,
        "error": None,
        "attempts": 1,
    }