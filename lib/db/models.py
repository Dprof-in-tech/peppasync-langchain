from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    otp_code = Column(String(10), nullable=True)              # Stores the OTP code (optional, max 10 chars)
    otp_expires_at = Column(DateTime, nullable=True)          # When the OTP expires (optional)
    is_verified = Column(Integer, default=0)
