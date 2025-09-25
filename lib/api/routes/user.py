from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from db.schemas import UserCreate # Import UserCreate
from auth.signup import create_user_with_otp, verify_user_otp, check_user_can_login # Updated import
from db.database import get_db

user_router = APIRouter()

# Pydantic schema for OTP verification request
class OTPVerification(BaseModel):
    email: str
    otp: str

# Pydantic schema for Login request
class UserLogin(BaseModel):
    email: str
    password: str # Assuming password will be added later

@user_router.post("/signup", summary="Register a new user and send OTP")
def signup_user(user_in: UserCreate, db: Session = Depends(get_db)):
    user, message = create_user_with_otp(db, user_in)
    if user is None:
        raise HTTPException(status_code=400, detail=message)
    return {"message": message, "user_email": user.email}

@user_router.post("/verify-otp", summary="Verify OTP for user registration")
def verify_otp_endpoint(otp_data: OTPVerification, db: Session = Depends(get_db)):
    is_valid, message = verify_user_otp(db, otp_data.email, otp_data.otp)
    if not is_valid:
        raise HTTPException(status_code=400, detail=message)
    return {"message": message}

@user_router.post("/login", summary="Login user after email verification")
def login_user(user_login: UserLogin, db: Session = Depends(get_db)):
    # For now, only checks email verification. Password check would be added here.
    if check_user_can_login(db, user_login.email):
        # In a real app, you'd verify password here and return a token (e.g., JWT)
        return {"message": "Login successful!"}
    else:
        raise HTTPException(status_code=403, detail="Email not verified or user not found. Please verify your email first.")
