from sqlalchemy.orm import Session
from ..db.models import User
from ..db.schemas import UserCreate
from ..pd import generate_otp, otp_expiry, verify_otp
import datetime
import logging

logger = logging.getLogger(__name__)

def create_user_with_otp(db: Session, user_in: UserCreate):
    """
    Creates a new user, generates an OTP, stores it, and sends it via email.
    """
    # Check if user with this email already exists
    existing_user = db.query(User).filter(User.email == user_in.email).first()
    if existing_user:
        logger.warning(f"Attempted to create user with existing email: {user_in.email}")
        # Optionally, you might want to re-send OTP or raise a specific error
        return None, "User with this email already exists."

    otp_code = generate_otp()
    otp_expires = otp_expiry()
    user = User(
        username=user_in.username,
        email=user_in.email,
        otp_code=otp_code,
        otp_expires_at=otp_expires,
        created_at=datetime.datetime.utcnow(),
        is_verified=0  # New users are unverified by default
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Send the OTP code to the user's email
    send_otp_email(user.email, otp_code)
    logger.info(f"User {user.email} created with OTP. OTP expires at {otp_expires}")
    return user, "User registered successfully, OTP sent."

def send_otp_email(email: str, otp: str):
    """
    Simulates sending an OTP email. Replace with your actual email service.
    """
    logger.info(f"[EMAIL SERVICE] Sending OTP {otp} to {email}")
    # In a real application, integrate with an email sending service (e.g., SendGrid, Mailgun, SMTP)
    # Example:
    # from your_email_service import send_email
    # send_email(to_email=email, subject="Your OTP Code", body=f"Your OTP is: {otp}")

def verify_user_otp(db: Session, user_email: str, input_otp: str):
    """
    Verifies the provided OTP code for a user.
    If valid, marks the user as verified and clears OTP fields.
    """
    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        logger.warning(f"OTP verification failed: User not found for email {user_email}")
        return False, "User not found."
    if user.is_verified == 1:
        logger.info(f"OTP verification skipped: User {user_email} already verified.")
        return True, "Email already verified."
    if not user.otp_code or not user.otp_expires_at:
        logger.warning(f"OTP verification failed: No OTP pending for user {user_email}")
        return False, "No OTP pending verification or OTP already used."

    # Check OTP and expiry using the utility function
    is_valid = verify_otp(input_otp, user.otp_code, user.otp_expires_at)
    if is_valid:
        user.otp_code = None
        user.otp_expires_at = None
        user.is_verified = 1  # Mark user as verified
        db.commit()
        db.refresh(user)
        logger.info(f"User {user_email} successfully verified.")
        return True, "Email verification successful."
    else:
        logger.warning(f"OTP verification failed: Invalid or expired OTP for user {user_email}")
        return False, "Invalid or expired OTP."

def check_user_can_login(db: Session, email: str) -> bool:
    """
    Checks if a user exists and their email is verified for login.
    """
    user = db.query(User).filter(User.email == email).first()
    if user and user.is_verified == 1:
        logger.info(f"User {email} is verified and can log in.")
        return True
    logger.warning(f"User {email} not found or not verified for login attempt.")
    return False
