# q_ensemble_sandbox/config_sandbox.py
# MULTI-PROJECT SANDBOXED CONFIG - Does NOT affect Crude Queen pipeline
import os

# --- Default Project (can be overridden via command line) ---
PROJECT_ID = "1860"
PROJECT_NAME = "Bitcoin"

# --- Paths (All relative to this sandbox folder) ---
SANDBOX_ROOT = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SANDBOX_ROOT, '..'))  # Parent for .env access

# Base data directory
DATA_BASE_DIR = os.path.join(SANDBOX_ROOT, 'data')

# Project-specific data directory (will be set by set_project())
DATA_DIR = None
MASTER_DATASET_PATH = None
PRICE_CACHE_PATH = None

def set_project(project_id: str, project_name: str):
    """
    Configure paths for a specific project.
    Call this before running any pipeline steps.
    """
    global PROJECT_ID, PROJECT_NAME, DATA_DIR, MASTER_DATASET_PATH, PRICE_CACHE_PATH
    
    PROJECT_ID = project_id
    PROJECT_NAME = project_name
    
    # Create project-specific data folder
    DATA_DIR = os.path.join(DATA_BASE_DIR, f"{project_id}_{project_name}")
    MASTER_DATASET_PATH = os.path.join(DATA_DIR, 'training_data.parquet')
    PRICE_CACHE_PATH = os.path.join(DATA_DIR, 'price_history_cache.json')
    
    # Ensure directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'horizons_wide'), exist_ok=True)
    
    print(f"[Config] Project set to: {PROJECT_NAME} (ID: {PROJECT_ID})")
    print(f"[Config] Data directory: {DATA_DIR}")

# Initialize with defaults
set_project(PROJECT_ID, PROJECT_NAME)

# --- API Configuration ---
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
except ImportError:
    print("WARNING: python-dotenv not installed. Install with: pip install python-dotenv")
    print("         Continuing without .env file...")

API_KEY = os.getenv('QML_API_KEY')
API_BASE_URL = "https://quantumcloud.ai"

if not API_KEY:
    print("WARNING: QML_API_KEY not found in environment variables.")
