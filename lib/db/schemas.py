from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class UserCreate(BaseModel):
    username: str
    email: str

class UserOut(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime
    otp_code: Optional[str] = None           # Include in response only if you want (optional)
    otp_expires_at: Optional[datetime] = None  # Include in response only if you want (optional)
    is_verified: Optional[int] = 0
