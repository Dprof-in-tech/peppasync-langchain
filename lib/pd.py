import secrets
import string
from datetime import datetime, timedelta

def generate_otp(length=6) -> str:
    """Generate a numeric OTP code of given length (default: 6 digits)."""
    digits = string.digits
    otp = ''.join(secrets.choice(digits) for _ in range(length))
    return otp

def otp_expiry(minutes=5) -> datetime:
    """Return the expiry time for the OTP code (default: 5 minutes validity)."""
    return datetime.utcnow() + timedelta(minutes=minutes)

def verify_otp(input_otp: str, actual_otp: str, expiry_time: datetime) -> bool:
    """
    Verify the OTP:
    - Checks value and expiry.
    - Returns True if valid, False otherwise.
    """
    if datetime.utcnow() > expiry_time:
        return False
    return secrets.compare_digest(input_otp, actual_otp)
