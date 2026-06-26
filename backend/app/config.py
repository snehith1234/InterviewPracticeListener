import os
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
SERVER_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
ALLOW_CLIENT_API_KEY = os.getenv("ALLOW_CLIENT_API_KEY", "true").lower() == "true"
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
