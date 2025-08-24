from pydantic import BaseModel, EmailStr

# User signup schema: frontend sends { name, email, password }
class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str

# User login schema: frontend uses email/password
class UserLogin(BaseModel):
    email: EmailStr
    password: str

# Token schema for authentication responses (if you use JWT)
class Token(BaseModel):
    access_token: str
    token_type: str
