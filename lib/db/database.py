from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from ..config import AppConfig

# Use the connection string from AppConfig
SQLALCHEMY_DATABASE_URL = AppConfig.DATABASE_URL

assert SQLALCHEMY_DATABASE_URL is not None, "DATABASE_URL is not set in AppConfig."
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
