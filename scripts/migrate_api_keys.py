#!/usr/bin/env python3
"""
API Key Migration Script - Plaintext to PBKDF2 Hashed Format
============================================================
QDT Nexus - qdtnexus.ai

Migrates existing plaintext API keys to secure PBKDF2 hashed format.

This script:
1. Creates a timestamped backup of api_keys.json
2. Converts plaintext keys to hashed format using APIKeyManager
3. Generates a one-time notification file with plaintext keys for users
4. Saves the new secure format

IMPORTANT: After migration, plaintext keys are no longer stored.
Users must save their keys from the notification file.

Usage:
    python scripts/migrate_api_keys.py [--dry-run] [--keys-file PATH]

Author: Artemis (with Claude Code)
Date: 2026-02-03
"""

import json
import os
import sys
import shutil
import argparse
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.security import APIKeyManager


class MigrationError(Exception):
    """Raised when migration fails."""
    pass


def load_json_file(file_path: Path) -> dict:
    """Load and parse JSON file."""
    if not file_path.exists():
        raise MigrationError(f"File not found: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise MigrationError(f"Invalid JSON in {file_path}: {e}")


def save_json_file(file_path: Path, data: dict, mode: int = 0o600) -> None:
    """Save data as JSON with secure permissions."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    os.chmod(file_path, mode)


def create_backup(source_path: Path, backup_dir: Path) -> Path:
    """
    Create timestamped backup of source file.

    Returns:
        Path to backup file
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_filename = f"api_keys_backup_{timestamp}.json"
    backup_path = backup_dir / backup_filename

    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, backup_path)
    os.chmod(backup_path, 0o600)

    return backup_path


def is_already_hashed(key_data: dict) -> bool:
    """Check if key entry is already in hashed format."""
    return 'key_hash' in key_data and 'salt' in key_data


def is_plaintext_key(key_id: str) -> bool:
    """Check if key_id looks like a plaintext API key."""
    return key_id.startswith(APIKeyManager.KEY_PREFIX) and len(key_id) > 40


def generate_key_id(user_id: str, index: int) -> str:
    """Generate a readable key ID."""
    safe_user_id = ''.join(c if c.isalnum() else '_' for c in user_id)[:20]
    timestamp = datetime.now().strftime('%Y%m%d')
    return f"key_{safe_user_id}_{timestamp}_{index:03d}"


def migrate_keys(keys_data: dict) -> tuple[dict, list[dict]]:
    """
    Migrate plaintext keys to hashed format.

    Args:
        keys_data: Original keys data with plaintext keys

    Returns:
        Tuple of (migrated_data, user_notifications)
    """
    api_keys = keys_data.get('api_keys', {})

    if not api_keys:
        raise MigrationError("No API keys found in file")

    migrated_keys = {}
    user_notifications = []
    key_index = 1

    for key_or_id, key_info in api_keys.items():
        # Skip already hashed keys
        if is_already_hashed(key_info):
            print(f"  [SKIP] {key_or_id[:20]}... already hashed")
            migrated_keys[key_or_id] = key_info
            continue

        # Check if this is a plaintext key (key is stored as the dict key)
        if is_plaintext_key(key_or_id):
            plaintext_key = key_or_id
            user_id = key_info.get('user_id', 'unknown')
            new_key_id = generate_key_id(user_id, key_index)
            key_index += 1
        else:
            # Key might be stored differently, skip
            print(f"  [WARN] Unrecognized key format: {key_or_id[:20]}...")
            migrated_keys[key_or_id] = key_info
            continue

        # Hash the plaintext key
        key_hash, salt = APIKeyManager.hash_api_key(plaintext_key)

        # Build new key entry
        migrated_entry = {
            'key_hash': key_hash,
            'salt': salt,
            'user_id': key_info.get('user_id', 'unknown'),
            'assets': key_info.get('assets', ['*']),
            'created': key_info.get('created', datetime.now().isoformat()),
            'migrated': datetime.now().isoformat(),
            'rate_limit': key_info.get('rate_limit', 1000),
            'enabled': key_info.get('enabled', True)
        }

        migrated_keys[new_key_id] = migrated_entry

        # Record for user notification
        user_notifications.append({
            'key_id': new_key_id,
            'api_key': plaintext_key,
            'user_id': key_info.get('user_id', 'unknown'),
            'assets': key_info.get('assets', ['*']),
            'rate_limit': key_info.get('rate_limit', 1000)
        })

        print(f"  [OK] Migrated: {plaintext_key[:15]}...{plaintext_key[-4:]} -> {new_key_id}")

    migrated_data = {
        'api_keys': migrated_keys,
        'schema_version': '2.0',
        'migrated_at': datetime.now().isoformat(),
        'hash_algorithm': 'PBKDF2-SHA256',
        'iterations': APIKeyManager.ITERATIONS
    }

    return migrated_data, user_notifications


def create_user_notification_file(
    notifications: list[dict],
    output_path: Path
) -> None:
    """
    Create a notification file with plaintext keys for users.

    WARNING: This file contains sensitive data and should be
    distributed securely and deleted after users save their keys.
    """
    content = []
    content.append("=" * 70)
    content.append("QDT NEXUS API KEY MIGRATION NOTICE")
    content.append("=" * 70)
    content.append("")
    content.append("IMPORTANT: Your API keys have been migrated to a secure hashed format.")
    content.append("This is your ONE-TIME opportunity to save your API key.")
    content.append("")
    content.append("After saving your key, DELETE THIS FILE for security.")
    content.append("")
    content.append("-" * 70)
    content.append("")

    for i, notification in enumerate(notifications, 1):
        content.append(f"Key #{i}")
        content.append(f"  Key ID:     {notification['key_id']}")
        content.append(f"  API Key:    {notification['api_key']}")
        content.append(f"  User ID:    {notification['user_id']}")
        content.append(f"  Assets:     {', '.join(notification['assets']) if isinstance(notification['assets'], list) else notification['assets']}")
        content.append(f"  Rate Limit: {notification['rate_limit']} requests/day")
        content.append("")
        content.append("  Usage:")
        content.append(f"    Header: X-API-Key: {notification['api_key']}")
        content.append("")
        content.append("-" * 70)
        content.append("")

    content.append("SECURITY NOTICE:")
    content.append("- Store your API key securely (password manager recommended)")
    content.append("- Never commit API keys to version control")
    content.append("- Delete this file after saving your key")
    content.append("- If you lose your key, contact support for a new one")
    content.append("")
    content.append(f"Generated: {datetime.now().isoformat()}")
    content.append("=" * 70)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content))

    os.chmod(output_path, 0o600)


def main():
    parser = argparse.ArgumentParser(
        description="Migrate API keys from plaintext to PBKDF2 hashed format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/migrate_api_keys.py --dry-run
    python scripts/migrate_api_keys.py
    python scripts/migrate_api_keys.py --keys-file /path/to/api_keys.json
        """
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be migrated without making changes'
    )
    parser.add_argument(
        '--keys-file',
        type=Path,
        default=PROJECT_ROOT / 'api_keys.json',
        help='Path to api_keys.json (default: <project>/api_keys.json)'
    )
    parser.add_argument(
        '--backup-dir',
        type=Path,
        default=PROJECT_ROOT / 'backups',
        help='Directory for backup files (default: <project>/backups)'
    )
    parser.add_argument(
        '--notification-dir',
        type=Path,
        default=PROJECT_ROOT / 'notifications',
        help='Directory for user notification files (default: <project>/notifications)'
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("QDT NEXUS API KEY MIGRATION")
    print("Plaintext -> PBKDF2 Hashed Format")
    print("=" * 70 + "\n")

    keys_file = args.keys_file.absolute()
    backup_dir = args.backup_dir.absolute()
    notification_dir = args.notification_dir.absolute()

    print(f"Keys file:        {keys_file}")
    print(f"Backup directory: {backup_dir}")
    print(f"Notifications:    {notification_dir}")
    print(f"Dry run:          {args.dry_run}")
    print()

    # Check if keys file exists
    if not keys_file.exists():
        print(f"[ERROR] Keys file not found: {keys_file}")
        print("        Create keys using: python manage_api_keys.py create --user-id <id> --assets <assets>")
        sys.exit(1)

    # Load existing keys
    print("[1/5] Loading existing keys...")
    try:
        keys_data = load_json_file(keys_file)
    except MigrationError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    api_keys = keys_data.get('api_keys', {})
    print(f"       Found {len(api_keys)} key(s)")

    # Check schema version
    if keys_data.get('schema_version') == '2.0':
        print("\n[INFO] Keys are already in hashed format (schema v2.0)")
        print("       No migration needed.")
        sys.exit(0)

    # Migrate keys
    print("\n[2/5] Migrating keys to hashed format...")
    try:
        migrated_data, user_notifications = migrate_keys(keys_data)
    except MigrationError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"       Migrated {len(user_notifications)} key(s)")

    if args.dry_run:
        print("\n[DRY RUN] Would perform the following actions:")
        print(f"  - Create backup at: {backup_dir}/api_keys_backup_<timestamp>.json")
        print(f"  - Update keys file: {keys_file}")
        print(f"  - Create notification: {notification_dir}/api_key_notification_<timestamp>.txt")
        print("\nMigrated data preview:")
        print(json.dumps(migrated_data, indent=2)[:500] + "...")
        sys.exit(0)

    # Create backup
    print("\n[3/5] Creating backup...")
    try:
        backup_path = create_backup(keys_file, backup_dir)
        print(f"       Backup saved: {backup_path}")
    except Exception as e:
        print(f"[ERROR] Failed to create backup: {e}")
        sys.exit(1)

    # Save migrated keys
    print("\n[4/5] Saving migrated keys...")
    try:
        save_json_file(keys_file, migrated_data)
        print(f"       Keys updated: {keys_file}")
    except Exception as e:
        print(f"[ERROR] Failed to save migrated keys: {e}")
        print(f"        Backup available at: {backup_path}")
        sys.exit(1)

    # Create user notification file
    print("\n[5/5] Creating user notification file...")
    if user_notifications:
        notification_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        notification_path = notification_dir / f"api_key_notification_{timestamp}.txt"

        try:
            create_user_notification_file(user_notifications, notification_path)
            print(f"       Notification saved: {notification_path}")
        except Exception as e:
            print(f"[ERROR] Failed to create notification: {e}")
            print("        Migration completed but notification file not created.")
            print("        Users' keys are in the backup file.")
    else:
        print("       No keys to notify (all were already hashed)")

    print("\n" + "=" * 70)
    print("MIGRATION COMPLETE")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Distribute notification file to users SECURELY")
    print("  2. Users must save their API keys")
    print("  3. DELETE the notification file after distribution")
    print("  4. Verify API authentication works with hashed keys")
    print()
    print(f"Backup location: {backup_path}")
    if user_notifications:
        print(f"Notification:    {notification_path}")
    print()


if __name__ == '__main__':
    main()
