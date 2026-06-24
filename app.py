import streamlit as st
import pandas as pd
import time
from pathlib import Path

import requests

BACKEND_URL = "http://localhost:8000"
OLLAMA_MODEL = "llama3.1"
MAX_ROWS = 500

def check_ollama_status() -> tuple[bool, str]:
    try:
        response = requests.get(f"{BACKEND_URL}/status")
        if response.status_code == 200:
            data = response.json()
            return data["ok"], data["message"]
        return False, f"Backend returned status code {response.status_code}"
    except Exception as e:
        return False, f"Cannot connect to backend: {e}"

def get_schema(db_path: str) -> str:
    try:
        response = requests.get(f"{BACKEND_URL}/schema", params={"db_path": db_path})
        if response.status_code == 200:
            return response.json()["schema"]
        raise RuntimeError(response.json().get("detail", "Failed to retrieve schema"))
    except Exception as e:
        raise RuntimeError(f"Error fetching schema from backend: {e}")

def process_query(db_path: str, user_question: str, schema: str, chat_history: list = None) -> dict:
    try:
        payload = {
            "db_path": db_path,
            "user_question": user_question,
            "db_schema": schema,
            "chat_history": chat_history or []
        }
        response = requests.post(f"{BACKEND_URL}/query", json=payload)
        if response.status_code == 200:
            return response.json()
        return {
            "sql": None, "columns": [], "rows": [], "natural_answer": None,
            "error": response.json().get("detail", f"Backend error {response.status_code}"),
            "attempts": 0
        }
    except Exception as e:
        return {
            "sql": None, "columns": [], "rows": [], "natural_answer": None,
            "error": f"Failed to connect to backend: {e}",
            "attempts": 0
        }

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="NL → SQL Chatbot",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Apply modern font globally and dark theme background */
html, body, [class*="css"], .stApp {
    font-family: 'Outfit', sans-serif !important;
    background-color: #090d16 !important;
}

/* Sidebar premium styling */
[data-testid="stSidebar"] {
    border-right: 1px solid #1e293b;
    background-color: #0b0f19 !important;
}

/* Smooth Scrolling and Custom Scrollbar */
html {
    scroll-behavior: smooth;
}
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.3);
}

/* Chat bubble entrance animation */
@keyframes bubbleEntrance {
    from {
        opacity: 0;
        transform: translateY(12px) scale(0.98);
    }
    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

/* Chat bubbles styling */
.user-bubble {
    background: linear-gradient(135deg, #3a86ff 0%, #8338ec 100%);
    border-radius: 20px 20px 4px 20px;
    padding: 14px 20px;
    margin: 12px 0 12px auto;
    max-width: 80%;
    color: #ffffff !important;
    font-size: 14px;
    font-weight: 400;
    line-height: 1.2;
    box-shadow: 0 4px 15px rgba(58, 134, 255, 0.15);
    white-space: pre-wrap;
    animation: bubbleEntrance 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) both;
    transition: all 0.3s ease;
}
.user-bubble:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(58, 134, 255, 0.25);
}

.bot-bubble {
    background: #ffffff;
    color: #1e293b;
    border: 1px solid #e2e8f0;
    border-radius: 20px 20px 20px 4px;
    padding: 16px 22px;
    margin: 12px 0;
    max-width: 80%;
    font-size: 12px;
    font-weight: 400;
    line-height: 1.2;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.03);
    white-space: pre-wrap;
    animation: bubbleEntrance 0.35s cubic-bezier(0.34, 1.56, 0.64, 1) both;
    transition: all 0.3s ease;
}
.bot-bubble:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 16px rgba(15, 23, 42, 0.06);
    border-color: #cbd5e1;
}

/* SQL code block inside chat */
.sql-block {
    background: #1e1e2e;
    border-radius: 8px;
    padding: 12px 16px;
    font-family: 'Courier New', monospace;
    font-size: 13px;
    color: #cdd6f4;
    margin: 8px 0;
    border-left: 4px solid #89b4fa;
}

/* Status pills */
.status-ok { 
    background: #10b98115; 
    color: #34d399; 
    border-radius: 20px; 
    padding: 4px 12px; 
    font-size: 12px; 
    font-weight: 500;
    border: 1px solid #10b98130;
}
.status-err { 
    background: #ef444415; 
    color: #f87171; 
    border-radius: 20px; 
    padding: 4px 12px; 
    font-size: 12px; 
    font-weight: 500;
    border: 1px solid #ef444430;
}

/* Hide Streamlit footer */
footer { visibility: hidden; }

/* Customizing Streamlit Button */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    padding: 8px 20px !important;
    border: 1px solid #1e293b !important;
    background: #111827 !important;
    color: #f8fafc !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    border-color: #334155 !important;
    background: #1e293b !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* Primary buttons style */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3a86ff 0%, #8338ec 100%) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(58, 134, 255, 0.2) !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #4b93ff 0%, #904cff 100%) !important;
    box-shadow: 0 6px 18px rgba(58, 134, 255, 0.35) !important;
}

/* Customizing input field border */
div[data-baseweb="input"] {
    border-radius: 10px !important;
    border: 1px solid #1e293b !important;
    background-color: #111827 !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
div[data-baseweb="input"]:focus-within {
    border-color: #3a86ff !important;
    box-shadow: 0 0 0 3px rgba(58, 134, 255, 0.25) !important;
}

/* Beautiful custom spinner styling */
div[data-testid="stSpinner"] {
    background: #111827 !important;
    color: #f8fafc !important;
    border: 1px solid #1e293b !important;
    border-radius: 14px !important;
    padding: 16px 24px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2) !important;
    max-width: 360px;
    margin: 12px 0 12px 20px;
    animation: spinnerPulse 2s infinite ease-in-out;
}
@keyframes spinnerPulse {
    0%, 100% { transform: scale(1); box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2); }
    50% { transform: scale(0.99); box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15); }
}

/* Hide Streamlit dataframe toolbar (download, search, fullscreen) */
[data-testid="stElementToolbar"] {
    display: none !important;
}

/* Typing indicator animation */
.typing-indicator {
    display: inline-flex;
    align-items: center;
    column-gap: 4px;
    padding: 6px 4px;
}
.typing-dot {
    width: 6px;
    height: 6px;
    background-color: #94a3b8;
    border-radius: 50%;
    animation: typingBounce 1.4s infinite ease-in-out both;
}
.typing-dot:nth-child(1) { animation-delay: -0.32s; }
.typing-dot:nth-child(2) { animation-delay: -0.16s; }

@keyframes typingBounce {
    0%, 80%, 100% { transform: scale(0); }
    40% { transform: scale(1); }
}
</style>
""", unsafe_allow_html=True)



# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────

def init_state():
    if "chat_history" in st.session_state:
        return  # Already initialized, skip costly HTTP requests
        
    default_db = "company.db" if Path("company.db").exists() else ""
    default_schema = ""
    default_loaded = False
    if default_db:
        try:
            default_schema = get_schema(default_db)
            default_loaded = True
        except Exception:
            pass

    # Auto check Ollama status
    try:
        ollama_ok, ollama_msg = check_ollama_status()
    except Exception as e:
        ollama_ok, ollama_msg = False, str(e)

    st.session_state["chat_history"] = []
    st.session_state["db_path"] = default_db
    st.session_state["schema"] = default_schema
    st.session_state["ollama_ok"] = ollama_ok
    st.session_state["ollama_msg"] = ollama_msg
    st.session_state["schema_loaded"] = default_loaded
    st.session_state["pending_question"] = None

init_state()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("##  NL → SQL Chatbot")
    st.caption(f"Powered by Ollama `{OLLAMA_MODEL}` + SQLite")
    st.divider()

    # ── Ollama status ──
    st.markdown("####  Ollama Status")
    if st.button("Check Connection", use_container_width=True):
        with st.spinner("Checking..."):
            ok, msg = check_ollama_status()
            st.session_state.ollama_ok  = ok
            st.session_state.ollama_msg = msg

    if st.session_state.ollama_msg:
        if st.session_state.ollama_ok:
            st.success(st.session_state.ollama_msg)
        else:
            st.error(st.session_state.ollama_msg)
            st.code(f"# In terminal:\nollama serve\nollama pull {OLLAMA_MODEL}")

    st.divider()

    # ── Database selection ──
    st.markdown("####  Database")

    db_input = st.text_input(
        "SQLite file path",
        value=st.session_state.db_path,
        placeholder="e.g. ./sample.db or /absolute/path/db.sqlite",
    )

    load_clicked = st.button("Load DB", use_container_width=True, type="primary")

    if load_clicked:
        db_path = db_input.strip()
        if not db_path:
            st.error("Enter a DB path.")
        elif not Path(db_path).exists():
            st.error(f"File not found:\n`{db_path}`")
        else:
            try:
                with st.spinner("Reading schema..."):
                    schema = get_schema(db_path)
                st.session_state.db_path       = db_path
                st.session_state.schema        = schema
                st.session_state.schema_loaded = True
                st.session_state.chat_history  = []
                st.success("Database loaded ")
                st.rerun()
            except RuntimeError as e:
                st.error(str(e))

    if Path("company.db").exists() and st.session_state.db_path != "company.db":
        if st.button("📂 Load existing company.db", use_container_width=True):
            try:
                with st.spinner("Loading database..."):
                    schema = get_schema("company.db")
                    st.session_state.db_path       = "company.db"
                    st.session_state.schema        = schema
                    st.session_state.schema_loaded = True
                    st.session_state.chat_history  = []
                st.success("company.db loaded!")
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load company.db: {e}")

    # ── Schema viewer ──
    if st.session_state.schema_loaded:
        st.divider()
        st.markdown("####  Schema")
        with st.expander("View full schema", expanded=False):
            st.code(st.session_state.schema, language="sql")

        st.divider()
        st.markdown("####  Settings")
        st.caption(f"Max rows returned: **{MAX_ROWS}**")
        st.caption(" Read-only mode: **ENFORCED**")

        st.divider()
        if st.button(" Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # ── Read-only warning ──
    st.divider()
    st.markdown("""
    <div style='background:#fff3cd;border-radius:8px;padding:10px;font-size:12px;color:#664d03'>
      <b>Read-Only Enforced</b><br>
    This chatbot can only <b>SELECT</b> data.<br>
    All write operations are permanently blocked at three layers:
    <ol style='margin:6px 0 0 14px;padding:0'>
    <li>SQL keyword filter</li>
    <li>SQLite read-only URI</li>
    <li>DB authorizer callback</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────

st.markdown("##  Chat with your Database")

if not st.session_state.schema_loaded:
    # ── Welcome screen ──
    st.markdown("""
    <div style='text-align:center;padding:60px 20px'>
        <h2 style='color:#888'> Load a database to start chatting</h2>
        <p style='color:#aaa'>Use the sidebar to connect your SQLite DB or create a sample one.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### How it works:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1️ Load DB**\nConnect your SQLite database from the sidebar.")
    with col2:
        st.markdown("**2️ Ask Anything**\nType questions in plain English.")
    with col3:
        st.markdown("**3️ Get Results**\nSee the generated SQL + live data instantly.")

else:
    # ── Chat history ──
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="user-bubble"> {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                # Bot message
                st.markdown(
                    f'<div class="bot-bubble"> {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
                # (Table removed, answer is now fully inside the bot-bubble)

        # Pending question processing has been moved to the bottom to avoid an extra rerun.

    st.divider()

    # ── Chat input ──
    with st.form(key="chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_input = st.text_input(
                "Ask a question about your data...",
                placeholder="TYPING...",
                label_visibility="collapsed",
                key="chat_input_field",
            )
        with col_btn:
            submitted = st.form_submit_button("Ask", use_container_width=True, type="primary")

    # ── Process query ──
    if submitted and user_input.strip():
        question = user_input.strip()

        # Add user message to chat history immediately
        st.session_state.chat_history.append({
            "role":    "user",
            "content": question,
        })
        
        # Display the user message dynamically in the chat container BEFORE calling backend
        with chat_container:
            st.markdown(
                f'<div class="user-bubble"> {question}</div>',
                unsafe_allow_html=True,
            )
            
            # Show animated typing indicator inside chat container
            typing_placeholder = st.empty()
            with typing_placeholder:
                st.markdown(
                    '<div class="bot-bubble"><div class="typing-indicator">'
                    '<span class="typing-dot"></span>'
                    '<span class="typing-dot"></span>'
                    '<span class="typing-dot"></span>'
                    '</div></div>',
                    unsafe_allow_html=True
                )
            
        # Extract history excluding the current question which was just appended
        history_to_send = []
        for msg in st.session_state.chat_history[:-1]:
            history_to_send.append({"role": msg["role"], "content": msg["content"]})
            
        # Blocking call to backend
        result = process_query(
            db_path=st.session_state.db_path,
            user_question=question,
            schema=st.session_state.schema,
            chat_history=history_to_send
        )

        # ── Build bot response ──────────────────────────────────────
        # Case 1: Hard error from backend
        if result.get("error"):
            content = result["error"]
            bot_msg = {
                "role":    "bot",
                "content": f"⚠️ {content}",
                "id":      len(st.session_state.chat_history),
            }

        # Case 2: General chat / META / CANNOT_ANSWER — no SQL executed
        elif not result.get("sql"):
            content = result.get("natural_answer") or "I'm not sure how to answer that."
            bot_msg = {
                "role":    "bot",
                "content": content,
                "id":      len(st.session_state.chat_history),
            }

        # Case 3: SQL was executed — show natural language answer
        else:
            content = result.get("natural_answer") or "Query executed successfully."
            bot_msg = {
                "role":    "bot",
                "content": content,
                "id":      len(st.session_state.chat_history),
            }

        st.session_state.chat_history.append(bot_msg)
        st.rerun()
