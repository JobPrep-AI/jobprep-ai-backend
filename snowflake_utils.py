"""
snowflake_utils.py
------------------
Central utility module for JobPrep AI.
- Single Snowflake connection with auto-reconnect
- Single llm() and get_query_embedding() used by all modules
- Reads credentials from st.secrets (or falls back to env vars)
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import snowflake.connector
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# CREDENTIALS
# Priority: st.secrets → environment variables → hard defaults
# ---------------------------------------------------------------
def _get_secret(key: str, default: str = "") -> str:
    """Read a value from st.secrets if Streamlit is running, else env vars."""
    try:
        import streamlit as st
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)


def _load_private_key(path: str):
    """Load RSA private key from a .p8 file. Path is resolved relative to this file."""
    key_path = Path(path)
    if not key_path.is_absolute():
        # Anchor to the directory of this file, not cwd
        key_path = Path(__file__).parent / key_path

    with open(key_path, "rb") as f:
        private_key = serialization.load_pem_private_key(
            f.read(), password=None, backend=default_backend()
        )

    return private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


# ---------------------------------------------------------------
# CONNECTION POOL (single connection, auto-reconnect)
# ---------------------------------------------------------------
_connection: snowflake.connector.SnowflakeConnection | None = None


def _build_connection() -> snowflake.connector.SnowflakeConnection:
    pkb = _load_private_key(_get_secret("SNOWFLAKE_PRIVATE_KEY_PATH", "rsa_key.p8"))
    conn = snowflake.connector.connect(
        user=_get_secret("SNOWFLAKE_USER", "BLUEJAY"),
        account=_get_secret("SNOWFLAKE_ACCOUNT", "pgb87192"),
        private_key=pkb,
        warehouse=_get_secret("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        database=_get_secret("SNOWFLAKE_DATABASE", "JOBPREP_DB"),
        schema=_get_secret("SNOWFLAKE_SCHEMA", "MARTS"),
    )
    logger.info("Snowflake connection established.")
    return conn


def get_connection() -> snowflake.connector.SnowflakeConnection:
    """
    Return the shared Snowflake connection.
    Automatically reconnects if the connection has been closed or timed out.
    """
    global _connection
    try:
        if _connection is None or _connection.is_closed():
            _connection = _build_connection()
        else:
            # Lightweight ping to catch silent timeouts
            _connection.cursor().execute("SELECT 1")
    except Exception as e:
        logger.warning(f"Snowflake connection lost ({e}), reconnecting...")
        try:
            _connection = _build_connection()
        except Exception as reconnect_err:
            logger.error(f"Reconnect failed: {reconnect_err}")
            raise
    return _connection


# Backwards-compatible alias used by legacy modules
def get_snowflake_connection():
    return get_connection()


# ---------------------------------------------------------------
# QUERY HELPERS
# ---------------------------------------------------------------
def fetch_df(query: str, params=None) -> pd.DataFrame:
    """Execute a query and return results as a DataFrame."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(query, params) if params else cur.execute(query)
        df = cur.fetch_pandas_all()
        df.columns = [str(c).lower() for c in df.columns]
        return df
    except Exception as e:
        logger.error(f"fetch_df failed: {e}\nQuery: {query[:200]}")
        raise
    finally:
        cur.close()


def execute(query: str, params=None):
    """Execute a non-SELECT statement."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(query, params) if params else cur.execute(query)
        conn.commit()
    except Exception as e:
        logger.error(f"execute failed: {e}\nQuery: {query[:200]}")
        raise
    finally:
        cur.close()


def executemany(query: str, rows: list):
    """Batch insert rows."""
    if not rows:
        return
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.executemany(query, rows)
        conn.commit()
    except Exception as e:
        logger.error(f"executemany failed: {e}\nQuery: {query[:200]}")
        raise
    finally:
        cur.close()


# ---------------------------------------------------------------
# LLM (single canonical implementation)
# ---------------------------------------------------------------
import re
import json


def clean_llm_output(text) -> str:
    """Normalize raw LLM response to a clean string."""
    if isinstance(text, list):
        text = "\n".join(map(str, text))
    elif isinstance(text, dict):
        text = json.dumps(text, ensure_ascii=False)
    elif text is None:
        return ""

    text = str(text).strip()

    # Unwrap outer JSON string quoting (Snowflake Cortex quirk)
    if text.startswith('"') and text.endswith('"'):
        try:
            unquoted = json.loads(text)
            if isinstance(unquoted, str):
                text = unquoted.strip()
        except Exception:
            pass

    # Strip <think>...</think> tags (DeepSeek-style reasoning traces)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    return text.strip()


def llm(prompt: str, model: str = "llama3.3-70b") -> str:
    """
    Call Snowflake Cortex LLM and return a clean string response.
    Uses $$ quoting to safely embed arbitrary prompt text.
    """
    # Escape $$ inside the prompt so it doesn't break the Snowflake SQL literal
    safe_prompt = prompt.replace("$$", "$ $")

    query = f"""
    SELECT SNOWFLAKE.CORTEX.AI_COMPLETE(
        '{model}',
        $$ {safe_prompt} $$
    ) AS RESPONSE
    """
    try:
        df = fetch_df(query)
        raw = df.iloc[0]["response"]
        return clean_llm_output(raw)
    except Exception as e:
        logger.error(f"llm() call failed (model={model}): {e}")
        return ""


# ---------------------------------------------------------------
# EMBEDDINGS
# ---------------------------------------------------------------
def get_embedding(text: str) -> np.ndarray:
    """Return a 768-dim embedding vector for the given text."""
    query = """
    SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768(
        'snowflake-arctic-embed-m-v1.5',
        %s
    ) AS EMB
    """
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(query, (text,))
        result = cur.fetchone()[0]
        cur.close()
        return np.array(result, dtype=float)
    except Exception as e:
        logger.error(f"get_embedding() failed: {e}")
        raise