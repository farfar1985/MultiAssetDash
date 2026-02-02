"""
Cleanup Script for QDT Ensemble Sandbox
========================================
Removes unused scripts and files based on CLEANUP_ANALYSIS.md

Run this script to clean up the folder before pushing to GitHub.
"""

import os
import shutil
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Files to DELETE
FILES_TO_DELETE = [
    # Analysis scripts (one-time use)
    "analyze_bitcoin.py",
    "analyze_strategies.py",
    "compare_combos_deep.py",
    "compare_ensemble_vs_raw.py",
    "calc_da.py",
    "calculate_dashboard_metrics.py",
    "calculate_signal_confidence.py",
    "recent_trades.py",
    
    # Grid search / optimization scripts (superseded)
    "grid_search_crude_oil.py",
    "grid_search_tuning.py",
    "deep_grid_search.py",
    "find_best_ensemble.py",
    "optimize_horizons.py",
    "test_new_optimal.py",
    
    # Old pipeline scripts (superseded by run_complete_pipeline.py)
    "run_complete_update.py",
    "run_daily_update.py",
    "run_full_update.py",
    "run_optimal_pipeline.py",
    "run_optimized_update.py",
    "run_batch.py",
    
    # Dashboard generation scripts (superseded by build_qdt_dashboard.py)
    "generate_optimized_dashboard.py",
    "generate_snake_chart_optimal.py",
    "build_marketing_hub.py",
    "build_marketing_hub_v2.py",
    
    # Other unused scripts
    "fetch_news_analysis.py",
    "generate_social_posts.py",
    
    # Log files
    "pipeline_log_*.txt",
    "update_log_*.txt",
    
    # Old output files
    "optimal_snake_chart.html",
    "QDT_Fresh.html",
    "QDT_Marketing_Hub.html",
    "QDT_Marketing_Hub_V2.html",
    
    # Cache files
    "news_analysis_cache.json",
    
    # Server config (not needed in repo)
    "nginx_config.txt",
    
    # Security risk file
    "root@45",
    
    # Validation scripts (keeping check_last_30_trades.py, validate_diverse_combo.py, validate_calculations.py)
    # These are useful for debugging
    
    # Deploy script (user said they know the command)
    "deploy_to_server.sh",
]

# Directories to DELETE
DIRS_TO_DELETE = [
    "ensemble_optimization",  # Entire folder
]

# Data files to clean up (in data/ directory)
DATA_FILES_TO_DELETE = [
    "data/sandbox_*.csv",
    "data/sandbox_*.json",
    "data/raw_children_1860/",
    "data/matrix_drift_chart.html",
    "data/sandbox_training_data.parquet",
]

def delete_file_or_dir(path):
    """Delete a file or directory."""
    full_path = os.path.join(SCRIPT_DIR, path)
    
    if not os.path.exists(full_path):
        return False, "Not found"
    
    try:
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
            return True, "Directory deleted"
        else:
            os.remove(full_path)
            return True, "File deleted"
    except Exception as e:
        return False, str(e)

def delete_by_pattern(pattern):
    """Delete files matching a pattern."""
    import glob
    matches = glob.glob(os.path.join(SCRIPT_DIR, pattern))
    deleted = 0
    errors = []
    
    for match in matches:
        try:
            if os.path.isdir(match):
                shutil.rmtree(match)
            else:
                os.remove(match)
            deleted += 1
        except Exception as e:
            errors.append(f"{os.path.basename(match)}: {str(e)}")
    
    return deleted, errors

def main():
    print("="*70)
    print("  QDT Ensemble Sandbox - Cleanup Script")
    print("="*70)
    print()
    
    deleted_count = 0
    error_count = 0
    
    # Delete individual files
    print("Deleting files...")
    for file_path in FILES_TO_DELETE:
        if '*' in file_path:
            # Pattern match
            deleted, errors = delete_by_pattern(file_path)
            if deleted > 0:
                print(f"  [OK] Deleted {deleted} file(s) matching: {file_path}")
                deleted_count += deleted
            if errors:
                for error in errors:
                    print(f"  [FAIL] {error}")
                    error_count += 1
        else:
            success, message = delete_file_or_dir(file_path)
            if success:
                print(f"  [OK] {file_path}")
                deleted_count += 1
            elif message != "Not found":
                print(f"  [FAIL] {file_path}: {message}")
                error_count += 1
    
    print()
    
    # Delete directories
    print("Deleting directories...")
    for dir_path in DIRS_TO_DELETE:
        success, message = delete_file_or_dir(dir_path)
        if success:
            print(f"  [OK] {dir_path}/")
            deleted_count += 1
        elif message != "Not found":
            print(f"  [FAIL] {dir_path}/: {message}")
            error_count += 1
    
    print()
    
    # Clean up data directory
    print("Cleaning up data directory...")
    for data_pattern in DATA_FILES_TO_DELETE:
        deleted, errors = delete_by_pattern(data_pattern)
        if deleted > 0:
            print(f"  [OK] Deleted {deleted} item(s) matching: {data_pattern}")
            deleted_count += deleted
        if errors:
            for error in errors:
                print(f"  [FAIL] {error}")
                error_count += 1
    
    print()
    print("="*70)
    print(f"  Cleanup Complete!")
    print(f"  Deleted: {deleted_count} items")
    if error_count > 0:
        print(f"  Errors: {error_count}")
    print("="*70)
    print()
    print("Next steps:")
    print("  1. Review the changes")
    print("  2. Test that the pipeline still works: python run_complete_pipeline.py --asset Crude_Oil")
    print("  3. Push to GitHub")

if __name__ == '__main__':
    # Safety check
    response = input("This will delete files. Are you sure? (yes/no): ")
    if response.lower() == 'yes':
        main()
    else:
        print("Cleanup cancelled.")

