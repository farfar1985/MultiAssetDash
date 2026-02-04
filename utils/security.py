"""
Security utilities for API key management.
QDT Nexus - Production Security Module

Author: Artemis (with Claude Code)
Date: 2026-02-03
"""

import hashlib
import secrets
import hmac
import os
from typing import Optional, Tuple
from functools import wraps
from flask import request, jsonify


class APIKeyManager:
    """
    Secure API key management with PBKDF2 hashing.
    
    Security features:
    - Cryptographically secure key generation
    - PBKDF2 with 100,000 iterations
    - Unique salt per key
    - Constant-time comparison (prevents timing attacks)
    """
    
    ITERATIONS = 100_000
    KEY_PREFIX = "qdt_"
    
    @staticmethod
    def generate_api_key() -> str:
        """
        Generate cryptographically secure API key.
        
        Returns:
            str: API key in format 'qdt_<44 random characters>'
        """
        return f"{APIKeyManager.KEY_PREFIX}{secrets.token_urlsafe(32)}"
    
    @staticmethod
    def generate_salt() -> str:
        """Generate cryptographically secure salt."""
        return secrets.token_hex(32)
    
    @staticmethod
    def hash_api_key(api_key: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """
        Hash API key using PBKDF2 with SHA-256.
        
        Args:
            api_key: The plaintext API key to hash
            salt: Optional salt (generated if not provided)
            
        Returns:
            Tuple of (hashed_key, salt)
        """
        if salt is None:
            salt = APIKeyManager.generate_salt()
        
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            api_key.encode('utf-8'),
            salt.encode('utf-8'),
            APIKeyManager.ITERATIONS
        )
        
        return hashed.hex(), salt
    
    @staticmethod
    def verify_api_key(provided_key: str, stored_hash: str, salt: str) -> bool:
        """
        Verify API key against stored hash using constant-time comparison.
        
        Args:
            provided_key: The API key to verify
            stored_hash: The stored hash to compare against
            salt: The salt used when hashing
            
        Returns:
            bool: True if key matches, False otherwise
        """
        computed_hash, _ = APIKeyManager.hash_api_key(provided_key, salt)
        return hmac.compare_digest(computed_hash, stored_hash)


def get_api_key_from_request() -> Optional[str]:
    """
    Extract API key from request.
    
    Checks (in order):
    1. X-API-Key header (preferred)
    2. Authorization: Bearer header
    3. api_key query parameter (deprecated, logged)
    
    Returns:
        str or None: The API key if found
    """
    # Preferred: X-API-Key header
    api_key = request.headers.get('X-API-Key')
    if api_key:
        return api_key
    
    # Alternative: Bearer token
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        return auth_header[7:]
    
    # Deprecated: Query parameter (log warning)
    api_key = request.args.get('api_key')
    if api_key:
        # TODO: Add logging - query param is deprecated
        # logger.warning(f"API key passed via query parameter (deprecated)")
        return api_key
    
    return None


def require_api_key(keys_loader_func):
    """
    Decorator factory to require API key authentication.
    
    Args:
        keys_loader_func: Function that returns dict of API keys
                         Format: {key_id: {key_hash, salt, enabled, user_id, rate_limit, ...}}
    
    Usage:
        @require_api_key(load_api_keys)
        def my_endpoint():
            # request.api_key_info contains the authenticated key's info
            pass
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            api_key = get_api_key_from_request()
            
            if not api_key:
                return jsonify({
                    "success": False,
                    "error": "API key required. Use X-API-Key header."
                }), 401
            
            # Load current keys
            keys_data = keys_loader_func()
            api_keys = keys_data.get("api_keys", {})
            
            # Find matching key by verifying hash
            key_info = None
            for key_id, info in api_keys.items():
                stored_hash = info.get('key_hash')
                salt = info.get('salt')
                
                if stored_hash and salt:
                    if APIKeyManager.verify_api_key(api_key, stored_hash, salt):
                        key_info = info
                        key_info['key_id'] = key_id
                        break
            
            if not key_info:
                return jsonify({
                    "success": False,
                    "error": "Invalid API key"
                }), 401
            
            # Check if key is enabled
            if not key_info.get('enabled', True):
                return jsonify({
                    "success": False,
                    "error": "API key is disabled"
                }), 403
            
            # Attach key info to request for downstream use
            request.api_key_info = key_info
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


# Rate limiting support
class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    pass


def check_rate_limit(user_id: str, limit: int, redis_client) -> Tuple[bool, int]:
    """
    Check if user has exceeded their rate limit.
    
    Args:
        user_id: Unique user identifier
        limit: Maximum requests per day
        redis_client: Redis client instance
        
    Returns:
        Tuple of (is_allowed, remaining_requests)
    """
    import time
    
    # Redis key for this user's daily count
    redis_key = f"rate_limit:{user_id}:{time.strftime('%Y-%m-%d')}"
    
    # Get current count
    current = redis_client.get(redis_key)
    current_count = int(current) if current else 0
    
    if current_count >= limit:
        return False, 0
    
    # Increment and set expiry
    pipe = redis_client.pipeline()
    pipe.incr(redis_key)
    pipe.expire(redis_key, 86400)  # 24 hours
    pipe.execute()
    
    return True, limit - current_count - 1
