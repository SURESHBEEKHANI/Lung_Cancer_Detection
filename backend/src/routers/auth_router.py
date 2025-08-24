from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database.database import get_db
from database.models import User
from schemas.schemas import UserCreate, UserLogin
from utils.security import hash_password, verify_password, create_access_token

auth_router = APIRouter(tags=["Authentication"], prefix="/auth")


# Signup route: frontend sends { name, email, password }
@auth_router.post("/signup")
def signup(user: UserCreate, db: Session = Depends(get_db)):
    # Check if user exists
    existing = db.query(User).filter((User.username == user.name) | (User.email == user.email)).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username or email already exists")

    # Hash the password
    hashed_password = hash_password(user.password)
    
    # Create new user (store name in username column)
    new_user = User(
        username=user.name,
        email=user.email,
        hashed_password=hashed_password
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Return token + user info
    access_token = create_access_token({"sub": str(new_user.id)})

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {"id": new_user.id, "email": new_user.email, "name": new_user.username}
    }


# Login route: expects { email, password }
@auth_router.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    # Authenticate user by email
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid credentials")

    access_token = create_access_token({"sub": str(db_user.id)})

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {"id": db_user.id, "email": db_user.email, "name": db_user.username}
    }
