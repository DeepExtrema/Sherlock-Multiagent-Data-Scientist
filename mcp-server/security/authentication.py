"""
Authentication and Security Module for Deepline MLOps Platform

Implements JWT-based authentication, role-based access control, and secure credential management.
Addresses critical security gaps identified in the comprehensive audit.
"""

import os
import jwt
import bcrypt
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import logging
from cryptography.fernet import Fernet
import base64

logger = logging.getLogger(__name__)

# Security configuration
SECRET_KEY = os.getenv('JWT_SECRET_KEY', secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', '30'))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv('REFRESH_TOKEN_EXPIRE_DAYS', '7'))

# Encryption key for sensitive data
ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY')
if not ENCRYPTION_KEY:
    ENCRYPTION_KEY = Fernet.generate_key()
    logger.warning("ENCRYPTION_KEY not set. Generated new key. Set this environment variable for production.")

cipher = Fernet(ENCRYPTION_KEY)

# Security scheme
security = HTTPBearer()

# User roles and permissions
ROLES = {
    "admin": {
        "description": "Full system access",
        "permissions": ["*"]
    },
    "ml_engineer": {
        "description": "ML workflow management",
        "permissions": [
            "workflow:read", "workflow:write", "workflow:execute",
            "agent:read", "agent:execute", "data:read", "data:write",
            "model:read", "model:write", "experiment:read", "experiment:write"
        ]
    },
    "data_scientist": {
        "description": "Data analysis and model development",
        "permissions": [
            "workflow:read", "workflow:execute",
            "agent:read", "agent:execute", "data:read", "data:write",
            "model:read", "model:write", "experiment:read", "experiment:write"
        ]
    },
    "data_engineer": {
        "description": "Data pipeline management",
        "permissions": [
            "workflow:read", "workflow:write",
            "agent:read", "data:read", "data:write"
        ]
    },
    "viewer": {
        "description": "Read-only access",
        "permissions": [
            "workflow:read", "agent:read", "data:read", "model:read", "experiment:read"
        ]
    }
}

# Pydantic models
class User(BaseModel):
    username: str = Field(..., description="Unique username")
    email: str = Field(..., description="User email")
    full_name: str = Field(..., description="Full name")
    role: str = Field(..., description="User role")
    is_active: bool = Field(default=True, description="Account status")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r"^[^@]+@[^@]+\.[^@]+$")
    full_name: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=8)
    role: str = Field(..., description="User role")

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: Dict[str, any]

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None
    permissions: List[str] = []

# In-memory user store (replace with database in production)
users_db: Dict[str, User] = {}
user_passwords: Dict[str, str] = {}

class SecureConfig:
    """Secure configuration management with encryption"""
    
    @staticmethod
    def encrypt_value(value: str) -> str:
        """Encrypt a sensitive value"""
        if not value:
            return ""
        encrypted = cipher.encrypt(value.encode())
        return base64.b64encode(encrypted).decode()
    
    @staticmethod
    def decrypt_value(encrypted_value: str) -> str:
        """Decrypt a sensitive value"""
        if not encrypted_value:
            return ""
        try:
            encrypted_bytes = base64.b64decode(encrypted_value.encode())
            decrypted = cipher.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Failed to decrypt value: {e}")
            return ""
    
    @staticmethod
    def get_secure_env(key: str, default: str = "") -> str:
        """Get environment variable with optional decryption"""
        value = os.getenv(key, default)
        if value.startswith("ENC:"):
            return SecureConfig.decrypt_value(value[4:])
        return value

class AuthenticationManager:
    """Manages user authentication and token handling"""
    
    def __init__(self):
        self.blacklisted_tokens: set = set()
    
    def create_user(self, user_data: UserCreate) -> User:
        """Create a new user"""
        if user_data.username in users_db:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        if user_data.role not in ROLES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid role. Valid roles: {list(ROLES.keys())}"
            )
        
        # Hash password
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(user_data.password.encode(), salt)
        
        # Create user
        user = User(
            username=user_data.username,
            email=user_data.email,
            full_name=user_data.full_name,
            role=user_data.role
        )
        
        users_db[user_data.username] = user
        user_passwords[user_data.username] = hashed_password.decode()
        
        logger.info(f"Created user: {user_data.username} with role: {user_data.role}")
        return user
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password"""
        if username not in users_db:
            return None
        
        user = users_db[username]
        if not user.is_active:
            return None
        
        stored_password = user_passwords.get(username)
        if not stored_password:
            return None
        
        if bcrypt.checkpw(password.encode(), stored_password.encode()):
            # Update last login
            user.last_login = datetime.utcnow()
            users_db[username] = user
            return user
        
        return None
    
    def create_tokens(self, user: User) -> Token:
        """Create access and refresh tokens for user"""
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        refresh_token_expires = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        
        access_token = self._create_token(
            data={"sub": user.username, "role": user.role},
            expires_delta=access_token_expires
        )
        
        refresh_token = self._create_token(
            data={"sub": user.username, "type": "refresh"},
            expires_delta=refresh_token_expires
        )
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user={
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "role": user.role,
                "permissions": ROLES[user.role]["permissions"]
            }
        )
    
    def _create_token(self, data: dict, expires_delta: timedelta) -> str:
        """Create JWT token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + expires_delta
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def verify_token(self, token: str) -> TokenData:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            role: str = payload.get("role")
            
            if username is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token"
                )
            
            if token in self.blacklisted_tokens:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )
            
            return TokenData(
                username=username,
                role=role,
                permissions=ROLES.get(role, {}).get("permissions", [])
            )
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def revoke_token(self, token: str):
        """Add token to blacklist"""
        self.blacklisted_tokens.add(token)
        logger.info("Token revoked")

class PermissionManager:
    """Manages role-based access control"""
    
    @staticmethod
    def has_permission(user_permissions: List[str], required_permission: str) -> bool:
        """Check if user has required permission"""
        if "*" in user_permissions:
            return True
        
        # Check exact permission
        if required_permission in user_permissions:
            return True
        
        # Check wildcard permissions (e.g., "workflow:*" matches "workflow:read")
        permission_parts = required_permission.split(":")
        if len(permission_parts) == 2:
            wildcard_permission = f"{permission_parts[0]}:*"
            if wildcard_permission in user_permissions:
                return True
        
        return False
    
    @staticmethod
    def require_permission(required_permission: str):
        """Decorator to require specific permission"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # Extract token data from kwargs or request
                token_data = kwargs.get('token_data')
                if not token_data:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Permission check failed"
                    )
                
                if not PermissionManager.has_permission(
                    token_data.permissions, 
                    required_permission
                ):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Insufficient permissions. Required: {required_permission}"
                    )
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator

# Global instances
auth_manager = AuthenticationManager()
permission_manager = PermissionManager()

# Dependency functions
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> TokenData:
    """Get current authenticated user"""
    token_data = auth_manager.verify_token(credentials.credentials)
    return token_data

async def get_current_active_user(
    current_user: TokenData = Depends(get_current_user)
) -> TokenData:
    """Get current active user"""
    if not current_user.username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return current_user

def require_role(required_role: str):
    """Decorator to require specific role"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            token_data = kwargs.get('token_data')
            if not token_data:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Role check failed"
                )
            
            if token_data.role != required_role and token_data.role != "admin":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient role. Required: {required_role}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Rate limiting
class RateLimiter:
    """Simple rate limiter for API endpoints"""
    
    def __init__(self):
        self.requests: Dict[str, List[datetime]] = {}
        self.max_requests = 100  # requests per window
        self.window_seconds = 60  # 1 minute window
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed"""
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.window_seconds)
        
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Remove old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > window_start
        ]
        
        # Check if under limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[client_id].append(now)
        return True

rate_limiter = RateLimiter()

def rate_limit(client_id: str = None):
    """Rate limiting decorator"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract client ID from request or use default
            if not client_id:
                # In a real implementation, extract from request IP or user
                client_id = "default"
            
            if not rate_limiter.is_allowed(client_id):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Data governance
class DataGovernance:
    """Data governance and PII detection"""
    
    def __init__(self):
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            'mac_address': r'\b([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})\b'
        }
    
    def detect_pii(self, data: str) -> List[Dict[str, str]]:
        """Detect PII in data"""
        import re
        detected = []
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.finditer(pattern, data)
            for match in matches:
                detected.append({
                    'type': pii_type,
                    'value': match.group(),
                    'position': match.span()
                })
        
        return detected
    
    def anonymize_data(self, data: str, pii_matches: List[Dict[str, str]]) -> str:
        """Anonymize PII in data"""
        anonymized = data
        for match in pii_matches:
            anonymized = anonymized.replace(match['value'], f"[{match['type'].upper()}]")
        return anonymized
    
    def validate_data_compliance(self, data: Dict[str, any]) -> Dict[str, any]:
        """Validate data for compliance (GDPR, HIPAA, etc.)"""
        compliance_report = {
            'gdpr_compliant': True,
            'hipaa_compliant': True,
            'pii_detected': [],
            'recommendations': []
        }
        
        # Check for PII in data
        data_str = str(data)
        pii_matches = self.detect_pii(data_str)
        
        if pii_matches:
            compliance_report['pii_detected'] = pii_matches
            compliance_report['gdpr_compliant'] = False
            compliance_report['recommendations'].append(
                "PII detected. Consider anonymization or encryption."
            )
        
        # Add more compliance checks as needed
        
        return compliance_report

# Initialize data governance
data_governance = DataGovernance()

# Security middleware
class SecurityMiddleware:
    """Security middleware for request processing"""
    
    @staticmethod
    async def process_request(request: Request):
        """Process incoming request for security checks"""
        # Log request for audit
        logger.info(f"Request: {request.method} {request.url} from {request.client.host}")
        
        # Check for suspicious patterns
        user_agent = request.headers.get("user-agent", "")
        if "sqlmap" in user_agent.lower() or "nmap" in user_agent.lower():
            logger.warning(f"Suspicious user agent detected: {user_agent}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Add security headers
        request.state.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
        }

# Example usage functions
def create_default_admin():
    """Create default admin user if none exists"""
    if "admin" not in users_db:
        admin_user = UserCreate(
            username="admin",
            email="admin@deepline.com",
            full_name="System Administrator",
            password="Admin123!",  # Change in production
            role="admin"
        )
        auth_manager.create_user(admin_user)
        logger.info("Default admin user created")

def get_user_by_username(username: str) -> Optional[User]:
    """Get user by username"""
    return users_db.get(username)

def list_users() -> List[User]:
    """List all users"""
    return list(users_db.values())

def update_user_role(username: str, new_role: str):
    """Update user role"""
    if username not in users_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    if new_role not in ROLES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role. Valid roles: {list(ROLES.keys())}"
        )
    
    user = users_db[username]
    user.role = new_role
    users_db[username] = user
    
    logger.info(f"Updated user {username} role to {new_role}")

# Initialize default admin user
create_default_admin() 