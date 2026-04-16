"""
auth.py
-------
User registration, login, and session management.
Password hashing using hashlib — no external auth library needed.
All user data stored in Snowflake.
"""

import hashlib
import secrets
import uuid
import logging
from snowflake_utils import fetch_df, execute

logger = logging.getLogger(__name__)


def _hash_password(password: str, salt: str) -> str:
    """Hash password with salt using SHA-256."""
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()


def _generate_salt() -> str:
    """Generate a random salt."""
    return secrets.token_hex(16)


def _generate_user_id() -> str:
    """Generate a unique user ID."""
    return str(uuid.uuid4())[:16]


# -------------------------------
# REGISTRATION
# -------------------------------
def register_user(name: str, email: str, password: str,
                  target_role: str = "", target_company: str = "") -> dict:
    """
    Register a new user.
    Returns dict with success, user_id, and message.
    """
    email = email.strip().lower()

    # Check if email already exists
    try:
        existing = fetch_df("""
            SELECT user_id FROM JOBPREP_DB.MARTS.USER_PROFILES
            WHERE email = %s
        """, (email,))
        if not existing.empty:
            return {"success": False, "message": "Email already registered. Please log in."}
    except Exception as e:
        logger.error(f"Register check failed: {e}")
        return {"success": False, "message": "Database error during registration."}

    # Hash password
    salt = _generate_salt()
    password_hash = f"{salt}:{_hash_password(password, salt)}"
    user_id = _generate_user_id()

    # Insert user
    try:
        execute("""
            INSERT INTO JOBPREP_DB.MARTS.USER_PROFILES
            (user_id, name, email, password_hash, target_role, target_company)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (user_id, name.strip(), email, password_hash,
              target_role.strip(), target_company.strip()))

        logger.info(f"New user registered: {email}")
        return {"success": True, "user_id": user_id, "message": "Registration successful!"}

    except Exception as e:
        logger.error(f"Register insert failed: {e}")
        return {"success": False, "message": "Registration failed. Please try again."}


# -------------------------------
# LOGIN
# -------------------------------
def login_user(email: str, password: str) -> dict:
    """
    Authenticate a user.
    Returns dict with success, user data, and message.
    """
    email = email.strip().lower()

    try:
        df = fetch_df("""
            SELECT user_id, name, email, password_hash,
                   target_role, target_company, created_at
            FROM JOBPREP_DB.MARTS.USER_PROFILES
            WHERE email = %s
        """, (email,))

        if df.empty:
            return {"success": False, "message": "Email not found. Please register first."}

        row = df.iloc[0]
        stored_hash = row["password_hash"]

        # Verify password
        salt, hashed = stored_hash.split(":", 1)
        if _hash_password(password, salt) != hashed:
            return {"success": False, "message": "Incorrect password. Please try again."}

        logger.info(f"User logged in: {email}")
        return {
            "success": True,
            "message": "Login successful!",
            "user": {
                "user_id":        row["user_id"],
                "name":           row["name"],
                "email":          row["email"],
                "target_role":    row["target_role"],
                "target_company": row["target_company"],
                "created_at":     str(row["created_at"])[:10],
            }
        }

    except Exception as e:
        logger.error(f"Login failed: {e}")
        return {"success": False, "message": "Login error. Please try again."}


# -------------------------------
# UPDATE PROFILE
# -------------------------------
def update_profile(user_id: str, target_role: str, target_company: str) -> dict:
    """Update user's target role and company."""
    try:
        execute("""
            UPDATE JOBPREP_DB.MARTS.USER_PROFILES
            SET target_role = %s, target_company = %s
            WHERE user_id = %s
        """, (target_role.strip(), target_company.strip(), user_id))
        return {"success": True, "message": "Profile updated!"}
    except Exception as e:
        logger.error(f"Profile update failed: {e}")
        return {"success": False, "message": "Update failed."}


# -------------------------------
# GET PROFILE
# -------------------------------
def get_profile(user_id: str) -> dict:
    """Fetch a user's profile by user_id."""
    try:
        df = fetch_df("""
            SELECT user_id, name, email, target_role, target_company, created_at
            FROM JOBPREP_DB.MARTS.USER_PROFILES
            WHERE user_id = %s
        """, (user_id,))

        if df.empty:
            return {}

        row = df.iloc[0]
        return {
            "user_id":        row["user_id"],
            "name":           row["name"],
            "email":          row["email"],
            "target_role":    row["target_role"],
            "target_company": row["target_company"],
            "created_at":     str(row["created_at"])[:10],
        }
    except Exception as e:
        logger.error(f"Get profile failed: {e}")
        return {}