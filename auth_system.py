"""
Simple Authentication System for VitalViewAI
Task 4: Privacy & Security - User Authentication with RBAC

Features:
- User registration and login
- Password hashing (bcrypt)
- JWT token generation
- Role-based access control
- Session management
"""

from datetime import datetime, timedelta
from typing import Optional, Dict
import jwt
import bcrypt
import json
from pathlib import Path


# JWT Configuration
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours


class UserDatabase:
    """
    Simple JSON-based user database
    In production: Use PostgreSQL/MongoDB
    """
    
    def __init__(self, db_path: str = "data/users.json"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.users = self._load_users()
    
    def _load_users(self) -> Dict:
        """Load users from JSON file"""
        if self.db_path.exists():
            with open(self.db_path, 'r') as f:
                return json.load(f)
        
        # Create default users for demo
        default_users = {
            "admin": {
                "username": "admin",
                "password_hash": bcrypt.hashpw("admin123".encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
                "role": "admin",
                "email": "admin@hospital.com",
                "full_name": "System Administrator",
                "created_at": datetime.now().isoformat()
            },
            "dr_smith": {
                "username": "dr_smith",
                "password_hash": bcrypt.hashpw("doctor123".encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
                "role": "clinician",
                "email": "smith@hospital.com",
                "full_name": "Dr. John Smith",
                "created_at": datetime.now().isoformat()
            },
            "nurse_alice": {
                "username": "nurse_alice",
                "password_hash": bcrypt.hashpw("nurse123".encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
                "role": "nurse",
                "email": "alice@hospital.com",
                "full_name": "Alice Johnson",
                "created_at": datetime.now().isoformat()
            },
            "researcher": {
                "username": "researcher",
                "password_hash": bcrypt.hashpw("research123".encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
                "role": "researcher",
                "email": "researcher@hospital.com",
                "full_name": "Research Team",
                "created_at": datetime.now().isoformat()
            }
        }
        
        self._save_users(default_users)
        return default_users
    
    def _save_users(self, users: Dict = None):
        """Save users to JSON file"""
        if users is None:
            users = self.users
        
        with open(self.db_path, 'w') as f:
            json.dump(users, f, indent=2)
    
    def get_user(self, username: str) -> Optional[Dict]:
        """Get user by username"""
        return self.users.get(username)
    
    def create_user(
        self,
        username: str,
        password: str,
        role: str,
        email: str,
        full_name: str
    ) -> bool:
        """Create new user"""
        if username in self.users:
            return False
        
        self.users[username] = {
            "username": username,
            "password_hash": bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
            "role": role,
            "email": email,
            "full_name": full_name,
            "created_at": datetime.now().isoformat()
        }
        
        self._save_users()
        return True
    
    def verify_password(self, username: str, password: str) -> bool:
        """Verify user password"""
        user = self.get_user(username)
        if not user:
            return False
        
        return bcrypt.checkpw(
            password.encode('utf-8'),
            user['password_hash'].encode('utf-8')
        )


class AuthenticationManager:
    """
    Handle authentication and JWT tokens
    """
    
    def __init__(self):
        self.user_db = UserDatabase()
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """
        Authenticate user with username and password
        
        Returns:
            User info if authenticated, None otherwise
        """
        if not self.user_db.verify_password(username, password):
            return None
        
        user = self.user_db.get_user(username)
        
        # Return user info without password hash
        return {
            "username": user['username'],
            "role": user['role'],
            "email": user['email'],
            "full_name": user['full_name']
        }
    
    def create_access_token(self, username: str, role: str) -> str:
        """
        Create JWT access token
        
        Returns:
            JWT token string
        """
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        payload = {
            "sub": username,
            "role": role,
            "exp": expire,
            "iat": datetime.utcnow()
        }
        
        token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
        return token
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """
        Verify JWT token
        
        Returns:
            Decoded token payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def login(self, username: str, password: str) -> Optional[Dict]:
        """
        Complete login flow
        
        Returns:
            {
                "access_token": "...",
                "token_type": "bearer",
                "user": {...}
            }
        """
        # Authenticate
        user = self.authenticate_user(username, password)
        if not user:
            return None
        
        # Create token
        token = self.create_access_token(user['username'], user['role'])
        
        return {
            "access_token": token,
            "token_type": "bearer",
            "user": user
        }
    
    def register_user(
        self,
        username: str,
        password: str,
        role: str,
        email: str,
        full_name: str
    ) -> bool:
        """Register new user"""
        return self.user_db.create_user(
            username=username,
            password=password,
            role=role,
            email=email,
            full_name=full_name
        )


# Global instance
auth_manager = AuthenticationManager()


# =============================================================================
# Demo Functions
# =============================================================================

def demo_authentication():
    """Demo authentication system"""
    print("\n" + "="*70)
    print("AUTHENTICATION SYSTEM DEMO")
    print("="*70)
    
    # Test login
    print("\n1. Testing Login:")
    result = auth_manager.login("dr_smith", "doctor123")
    
    if result:
        print(f"✅ Login successful!")
        print(f"   Username: {result['user']['username']}")
        print(f"   Role: {result['user']['role']}")
        print(f"   Token: {result['access_token'][:50]}...")
        
        # Test token verification
        print("\n2. Testing Token Verification:")
        payload = auth_manager.verify_token(result['access_token'])
        
        if payload:
            print(f"✅ Token valid!")
            print(f"   Username: {payload['sub']}")
            print(f"   Role: {payload['role']}")
        else:
            print("❌ Token invalid!")
    else:
        print("❌ Login failed!")
    
    # Test wrong password
    print("\n3. Testing Wrong Password:")
    result = auth_manager.login("dr_smith", "wrong_password")
    
    if result:
        print("❌ Should have failed!")
    else:
        print("✅ Correctly rejected wrong password")
    
    # Show all users
    print("\n4. Default Users:")
    print("   Username: admin | Password: admin123 | Role: admin")
    print("   Username: dr_smith | Password: doctor123 | Role: clinician")
    print("   Username: nurse_alice | Password: nurse123 | Role: nurse")
    print("   Username: researcher | Password: research123 | Role: researcher")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    demo_authentication()