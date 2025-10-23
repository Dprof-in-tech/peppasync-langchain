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
from lib.config import DatabaseManager

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
            You are an onboarding assistant. Analyze the business data {sales_data} and {inventory_data} for user with email {email} (session: {session_id}) and return a JSON object with:
            - low_stock_products: list of product names with low stock
            - underperforming_products: list of product names with declining sales
            - recommendations: list of actionable suggestions
            Respond ONLY with a valid JSON object.
            """
        )
        # Call the correct method on UnifiedBusinessAgent
        business_agent = UnifiedBusinessAgent()
        llm_response = await business_agent.analyze_direct_query(
            query=prompt,
            business_data={
                "sales_data": sales_data,
                "inventory_data": inventory_data
            },
            conversation_history=None
        )
        # Try to parse the LLM response as JSON
        try:
            data = llm_response if isinstance(llm_response, dict) else json.loads(llm_response)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": "LLM did not return valid JSON", "llm_response": str(llm_response)})
        return data
    except Exception as e:
        logger.error(f"Error in welcome insights: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
