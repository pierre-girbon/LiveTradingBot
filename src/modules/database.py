import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

# Single Base declarative class for all models
Base = declarative_base()

# Database URL from environment or default
DATABASE_URL = os.environ.get("DB_URL", "sqlite:///tradebot.db")

# SQLAlchemy engine (reused by all modules)
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Session factory for scoped sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db_session():
    """Utility to get new DB session."""
    return SessionLocal()
