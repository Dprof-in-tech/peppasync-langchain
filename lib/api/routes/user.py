from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import JSONResponse
import json
import logging

from ...db.schemas import UserCreate 
from ...auth.signup import create_user_with_otp, verify_user_otp, check_user_can_login
from ...db.database import get_db
from ...db.models import User
from lib.agent import UnifiedBusinessAgent
from lib.config import DatabaseManager, LLMManager
from langchain.schema import HumanMessage, SystemMessage

user_router = APIRouter()
# Set logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic schema for OTP verification request
class OTPVerification(BaseModel):
    email: str
    otp: str

# Pydantic schema for Login request
class UserLogin(BaseModel):
    email: str
    password: str # Assuming password will be added later

# Pydantic schema for Google sign-in user
class GoogleUserIn(BaseModel):
    email: str
    name: str
    google_id: str
    image: Optional[str] = None

# Pydantic schema for Welcome Insights request
class WelcomeInsightsRequest(BaseModel):
    email: str
    session_id: Optional[str] = None

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

@user_router.post("/google-signin", summary="Register or update user from Google sign-in")
def google_signin(user_in: GoogleUserIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == user_in.email).first()
    if user:
        # Update user info if needed
        user.username = user_in.name
        db.commit()
        db.refresh(user)
    else:
        user = User(username=user_in.name, email=user_in.email)
        db.add(user)
        db.commit()
        db.refresh(user)
    return {"id": user.id, "email": user.email, "username": user.username}

@user_router.get("/welcome-insights", summary="Get onboarding insights for new users")
async def welcome_insights(email: str, session_id: Optional[str] = None):
    """
    Returns structured onboarding insights for the welcome screen, using LLM for summary.
    """
    
    try:
        # Fetch business data for the user/session
        sales_data = await DatabaseManager.get_data(session_id=session_id, query_type="sales_data")
        inventory_data = await DatabaseManager.get_data(session_id=session_id, query_type="inventory_data")
        # Compose a prompt for the LLM to analyze business data for this user/session
        prompt = (
            f"""
            You are a professional business assistant. Your job is to analyze the business data {sales_data} and {inventory_data} for user with email {email} (session: {session_id}) and return 3 things that are going wrong with this business from their business data and some recommendations on how to fix them.

            This is what a sample response looks like:
            {{
              "problems":[
              "product 19 has 5 stock left, you WILL run out of stock tomorrow",
              "Product A has 0 sales this month, you are losing money",
              "Product B has high return rate of 15%, customers are unhappy"
              ]
              "recommendations": [
                "Increase marketing for Product C",
                "Consider discounts for Product A to boost sales"
              ]
            }}

            Analyze the data and return a JSON object with:
            - problems: list of top 3 issues identified
            - recommendations: list of actionable suggestions
            Respond ONLY with a valid JSON object.

            IMPORTANT:
            YOU MUST SOUND AS GRAVE AS POSSIBLE WHEN DESCRIBING THE PROBLEMS.
            """
        )

        messages = [
            SystemMessage(content="You are a business intelligence analyst. Respond ONLY with valid JSON. Do NOT use markdown code blocks or any formatting - just pure JSON."),
            HumanMessage(content=prompt)
        ]

        llm = LLMManager.get_chat_llm()

        llm_response = llm.invoke(messages)
        # Try to parse the LLM response as JSON
        try:
            result = json.loads(llm_response.content.strip())
            data = result if isinstance(result, dict) else json.loads(result)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": "LLM did not return valid JSON", "llm_response": str(llm_response)})
        return data
    except Exception as e:
        logger.error(f"Error in welcome insights: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
