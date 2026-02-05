"""
API Key Migration Script
========================
Migrates plaintext API keys to secure PBKDF2-hashed keys.

This is a one-time migration script. After running:
1. Existing plaintext keys are converted to hashed format
2. Original keys are backed up to api_keys.backup.json
3. New api_keys.json contains only hashed keys

Usage:
    python migrate_api_keys.py

Author: Artemis (with Claude Code)
Date: 2026-02-04
"""

import json
import os
import shutil
from datetime import datetime

# Import security utilities
from utils.security import APIKeyManager

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
API_KEYS_FILE = os.path.join(SCRIPT_DIR, 'api_keys.json')
BACKUP_FILE = os.path.join(SCRIPT_DIR, 'api_keys.backup.json')


def load_keys():
    """Load API keys from file."""
    if os.path.exists(API_KEYS_FILE):
        with open(API_KEYS_FILE, 'r') as f:
            return json.load(f)
    return {"api_keys": {}}


def save_keys(keys_data):
    """Save API keys to file."""
    with open(API_KEYS_FILE, 'w') as f:
        json.dump(keys_data, f, indent=2)


def is_already_hashed(key_info):
    """Check if a key entry is already in hashed format."""
    return 'key_hash' in key_info and 'salt' in key_info


def migrate_keys():
    """Migrate plaintext keys to hashed format."""
    keys_data = load_keys()
    api_keys = keys_data.get("api_keys", {})

    if not api_keys:
        print("[INFO] No API keys found to migrate.")
        return

    # Check if already migrated (all keys have hashes)
    needs_migration = False
    for key, info in api_keys.items():
        if not is_already_hashed(info):
            needs_migration = True
            break

    if not needs_migration:
        print("[INFO] All keys are already in hashed format. Nothing to migrate.")
        return

    # Backup existing file
    print(f"[INFO] Backing up existing keys to {BACKUP_FILE}")
    shutil.copy2(API_KEYS_FILE, BACKUP_FILE)

    # Migrate keys
    migrated_keys = {}
    plaintext_keys = []  # Store for user reference

    for plaintext_key, info in api_keys.items():
        if is_already_hashed(info):
            # Already migrated, keep as-is but use a new key_id
            key_id = info.get('key_id', plaintext_key[:16])
            migrated_keys[key_id] = info
            print(f"  [SKIP] Key {key_id}... already hashed")
        else:
            # Generate hash for plaintext key
            key_hash, salt = APIKeyManager.hash_api_key(plaintext_key)

            # Create new key entry with hash
            key_id = f"key_{len(migrated_keys) + 1:04d}"
            migrated_keys[key_id] = {
                "key_hash": key_hash,
                "salt": salt,
                "user_id": info.get("user_id", "unknown"),
                "assets": info.get("assets", ["*"]),
                "enabled": info.get("enabled", True),
                "rate_limit": info.get("rate_limit", 1000),
                "created": info.get("created", datetime.now().isoformat()),
                "migrated": datetime.now().isoformat()
            }

            # Store plaintext key for user reference
            plaintext_keys.append({
                "key_id": key_id,
                "plaintext_key": plaintext_key,
                "user_id": info.get("user_id", "unknown")
            })

            print(f"  [OK] Migrated key for user '{info.get('user_id', 'unknown')}' -> {key_id}")

    # Save migrated keys
    save_keys({"api_keys": migrated_keys})

    print(f"\n[SUCCESS] Migrated {len(plaintext_keys)} keys to secure format.")
    print(f"[INFO] Backup saved to: {BACKUP_FILE}")

    # Print key reference for users (they need to know their keys!)
    if plaintext_keys:
        print("\n" + "=" * 70)
        print("  IMPORTANT: Save these API keys - they cannot be recovered!")
        print("=" * 70)
        for key_info in plaintext_keys:
            print(f"\n  Key ID: {key_info['key_id']}")
            print(f"  User: {key_info['user_id']}")
            print(f"  API Key: {key_info['plaintext_key']}")
        print("\n" + "=" * 70)


def create_new_key(user_id, assets=None, rate_limit=1000):
    """Create a new secure API key."""
    keys_data = load_keys()
    api_keys = keys_data.get("api_keys", {})

    # Generate new secure key
    plaintext_key = APIKeyManager.generate_api_key()
    key_hash, salt = APIKeyManager.hash_api_key(plaintext_key)

    # Generate key ID
    key_id = f"key_{len(api_keys) + 1:04d}"

    # Add to keys
    api_keys[key_id] = {
        "key_hash": key_hash,
        "salt": salt,
        "user_id": user_id,
        "assets": assets if assets else ["*"],
        "enabled": True,
        "rate_limit": rate_limit,
        "created": datetime.now().isoformat()
    }

    save_keys({"api_keys": api_keys})

    print(f"\n[SUCCESS] Created new API key")
    print(f"  Key ID: {key_id}")
    print(f"  User: {user_id}")
    print(f"  API Key: {plaintext_key}")
    print(f"  Assets: {', '.join(api_keys[key_id]['assets'])}")
    print(f"\n  IMPORTANT: Save this key - it cannot be recovered!")

    return plaintext_key, key_id


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'create':
        # Create new key mode
        user_id = sys.argv[2] if len(sys.argv) > 2 else 'default_user'
        assets = sys.argv[3].split(',') if len(sys.argv) > 3 else ['*']
        create_new_key(user_id, assets)
    else:
        # Migration mode
        print("=" * 70)
        print("  API Key Migration - Plaintext to PBKDF2 Hashed")
        print("=" * 70)
        print()
        migrate_keys()
