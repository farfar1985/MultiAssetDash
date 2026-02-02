"""
API Key Management Script
=========================
Create and manage API keys for QDTNexus API.

Usage:
    python manage_api_keys.py create --user-id user1 --assets Crude_Oil Bitcoin SP500
    python manage_api_keys.py list
    python manage_api_keys.py delete --key your_api_key_here
    python manage_api_keys.py add-assets --key your_key --assets GOLD SP500
"""

import json
import os
import argparse
import secrets
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
API_KEYS_FILE = os.path.join(SCRIPT_DIR, 'api_keys.json')

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

def generate_api_key():
    """Generate a secure random API key."""
    return f"qdt_{secrets.token_urlsafe(32)}"

def create_key(user_id, assets, rate_limit=1000):
    """Create a new API key."""
    keys_data = load_keys()
    
    # Generate new key
    api_key = generate_api_key()
    
    keys_data["api_keys"][api_key] = {
        "user_id": user_id,
        "assets": assets if isinstance(assets, list) else assets.split(','),
        "created": datetime.now().isoformat(),
        "rate_limit": rate_limit
    }
    
    save_keys(keys_data)
    
    print(f"[OK] API key created successfully!")
    print(f"  User ID: {user_id}")
    print(f"  API Key: {api_key}")
    print(f"  Assets: {', '.join(keys_data['api_keys'][api_key]['assets'])}")
    print(f"  Rate Limit: {rate_limit} requests/day")
    print(f"\n  Use this key in API requests:")
    print(f"    Header: X-API-Key: {api_key}")
    print(f"    Or: ?api_key={api_key}")
    
    return api_key

def list_keys():
    """List all API keys."""
    keys_data = load_keys()
    api_keys = keys_data.get("api_keys", {})
    
    if not api_keys:
        print("[INFO] No API keys found")
        return
    
    print(f"\n{'='*70}")
    print(f"  API Keys ({len(api_keys)} total)")
    print(f"{'='*70}\n")
    
    for key, info in api_keys.items():
        print(f"  Key: {key[:20]}...")
        print(f"    User ID: {info.get('user_id', 'N/A')}")
        print(f"    Assets: {', '.join(info.get('assets', []))}")
        print(f"    Created: {info.get('created', 'N/A')}")
        print(f"    Rate Limit: {info.get('rate_limit', 1000)}/day")
        print()

def delete_key(api_key):
    """Delete an API key."""
    keys_data = load_keys()
    
    if api_key not in keys_data.get("api_keys", {}):
        print(f"[FAIL] API key not found: {api_key}")
        return False
    
    del keys_data["api_keys"][api_key]
    save_keys(keys_data)
    
    print(f"[OK] API key deleted: {api_key[:20]}...")
    return True

def add_assets(api_key, assets):
    """Add assets to an existing API key."""
    keys_data = load_keys()
    
    if api_key not in keys_data.get("api_keys", {}):
        print(f"[FAIL] API key not found: {api_key}")
        return False
    
    key_info = keys_data["api_keys"][api_key]
    current_assets = set(key_info.get("assets", []))
    
    new_assets = assets if isinstance(assets, list) else assets.split(',')
    current_assets.update(new_assets)
    
    key_info["assets"] = list(current_assets)
    save_keys(keys_data)
    
    print(f"[OK] Assets added to API key")
    print(f"  New assets list: {', '.join(key_info['assets'])}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Manage QDTNexus API keys")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create a new API key')
    create_parser.add_argument('--user-id', required=True, help='User ID')
    create_parser.add_argument('--assets', required=True, help='Comma-separated list of assets or "*" for all')
    create_parser.add_argument('--rate-limit', type=int, default=1000, help='Rate limit (requests per day)')
    
    # List command
    subparsers.add_parser('list', help='List all API keys')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete an API key')
    delete_parser.add_argument('--key', required=True, help='API key to delete')
    
    # Add assets command
    add_assets_parser = subparsers.add_parser('add-assets', help='Add assets to an API key')
    add_assets_parser.add_argument('--key', required=True, help='API key')
    add_assets_parser.add_argument('--assets', required=True, help='Comma-separated list of assets')
    
    args = parser.parse_args()
    
    if args.command == 'create':
        assets = '*' if args.assets == '*' else args.assets
        create_key(args.user_id, assets, args.rate_limit)
    elif args.command == 'list':
        list_keys()
    elif args.command == 'delete':
        delete_key(args.key)
    elif args.command == 'add-assets':
        add_assets(args.key, args.assets)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()

