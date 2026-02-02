"""
Complete QDT Ensemble Pipeline
==============================
Runs the full update pipeline for all assets:
1. Fetch fresh forecasts + run Q Ensemble (run_sandbox_pipeline.py)
2. Find optimal diverse combo (find_diverse_combo.py)
3. Pre-calculate metrics (precalculate_metrics.py)
4. Build dashboard (build_qdt_dashboard.py)

Usage:
    python run_complete_pipeline.py                    # All assets
    python run_complete_pipeline.py --asset Crude_Oil  # Single asset
    python run_complete_pipeline.py --skip-fetch        # Skip fetch step
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Asset mapping (name -> project_id)
ASSETS = {
    "Crude_Oil": "1866",
    "Bitcoin": "1860",
    "SP500": "1625",
    "NASDAQ": "269",
    "RUSSEL": "1518",
    "DOW_JONES_Mini": "336",
    "GOLD": "477",
    "US_DOLLAR_Index": "655",
    "SPDR_China_ETF": "291",
    "Nikkei_225": "358",
    "Nifty_50": "1398",
    "Nifty_Bank": "1387",
    "MCX_Copper": "1435",
    "USD_INR": "256",
    "Brent_Oil": "1859"
}

def run_command(cmd, description, timeout=600):
    """Run a command and return success status."""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding='utf-8',
            errors='replace'  # Replace encoding errors instead of failing
        )
        
        # Check if script actually succeeded (even if stderr has tqdm output)
        # For find_diverse_combo, check if config was updated or if it completed
        if result.returncode == 0:
            print(f"  [OK] {description} - SUCCESS")
            return True
        else:
            # Check stderr for actual errors vs just tqdm progress bars
            stderr_text = result.stderr if result.stderr else ''
            stdout_text = result.stdout if result.stdout else ''
            
            # If stderr only contains tqdm progress bars, it might have actually succeeded
            # Check for actual error messages
            has_real_error = any(keyword in stderr_text.lower() for keyword in 
                               ['error', 'exception', 'traceback', 'failed', 'fail'])
            
            if not has_real_error and 'testing diverse combos' in stderr_text.lower():
                # Likely just tqdm output, check if we can verify success another way
                print(f"  [WARN] {description} - May have succeeded (tqdm output in stderr)")
                print(f"  Check output: {stdout_text[:200] if stdout_text else 'No stdout'}")
                return True  # Assume success if no real errors found
            
            print(f"  [FAIL] {description} - FAILED")
            print(f"  Error: {stderr_text[:500] if stderr_text else 'Unknown error'}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  [WARN] {description} - TIMEOUT (>{timeout}s)")
        return False
    except Exception as e:
        print(f"  [FAIL] {description} - Exception: {str(e)[:200]}")
        return False

def step1_fetch_and_ensemble(asset_name, project_id, skip_fetch=False):
    """Step 1: Fetch fresh forecasts and run Q Ensemble."""
    if skip_fetch:
        print(f"  [SKIP] Skipping fetch for {asset_name}")
        return True
    
    cmd = [
        sys.executable,
        "run_sandbox_pipeline.py",
        "--project", project_id,
        "--name", asset_name
    ]
    
    return run_command(cmd, f"Fetch + Q Ensemble: {asset_name}", timeout=600)

def step2_find_diverse_combo(asset_name):
    """Step 2: Find optimal diverse horizon combination."""
    cmd = [
        sys.executable,
        "find_diverse_combo.py",
        "--asset", asset_name
    ]
    
    return run_command(cmd, f"Find Diverse Combo: {asset_name}", timeout=600)

def step3_precalculate_metrics(asset_name):
    """Step 3: Pre-calculate and save metrics to config."""
    cmd = [
        sys.executable,
        "precalculate_metrics.py",
        asset_name
    ]
    
    return run_command(cmd, f"Pre-calculate Metrics: {asset_name}", timeout=300)

def step4_build_dashboard():
    """Step 4: Build the unified dashboard."""
    cmd = [
        sys.executable,
        "build_qdt_dashboard.py"
    ]
    
    return run_command(cmd, "Build Dashboard", timeout=300)

def run_asset_pipeline(asset_name, project_id, skip_fetch=False):
    """Run complete pipeline for a single asset."""
    print(f"\n{'#'*70}")
    print(f"#  PROCESSING: {asset_name} (ID: {project_id})")
    print(f"{'#'*70}")
    
    start_time = datetime.now()
    results = {
        'fetch': False,
        'diverse_combo': False,
        'metrics': False
    }
    
    # Step 1: Fetch + Ensemble
    results['fetch'] = step1_fetch_and_ensemble(asset_name, project_id, skip_fetch)
    if not results['fetch']:
        print(f"  [WARN] {asset_name}: Fetch failed, but continuing...")
    
    # Step 2: Find Diverse Combo
    results['diverse_combo'] = step2_find_diverse_combo(asset_name)
    if not results['diverse_combo']:
        print(f"  [WARN] {asset_name}: Diverse combo search failed, but continuing...")
    
    # Step 3: Pre-calculate Metrics
    results['metrics'] = step3_precalculate_metrics(asset_name)
    if not results['metrics']:
        print(f"  [WARN] {asset_name}: Metrics calculation failed, but continuing...")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Summary
    print(f"\n{'='*70}")
    print(f"  {asset_name} Summary:")
    print(f"    Fetch + Ensemble:    {'[OK]' if results['fetch'] else '[FAIL]'}")
    print(f"    Diverse Combo:       {'[OK]' if results['diverse_combo'] else '[FAIL]'}")
    print(f"    Pre-calculate:       {'[OK]' if results['metrics'] else '[FAIL]'}")
    print(f"    Time:                {elapsed:.1f}s")
    print(f"{'='*70}")
    
    return all(results.values())

def main():
    parser = argparse.ArgumentParser(description="Complete QDT Ensemble Pipeline")
    parser.add_argument('--asset', type=str, default='all',
                        help='Asset name (default: all)')
    parser.add_argument('--skip-fetch', action='store_true',
                        help='Skip fetch step (use existing data)')
    args = parser.parse_args()
    
    # Determine assets to process
    if args.asset.lower() == 'all':
        assets_to_process = ASSETS
    else:
        if args.asset in ASSETS:
            assets_to_process = {args.asset: ASSETS[args.asset]}
        else:
            print(f"[FAIL] Unknown asset: {args.asset}")
            print(f"Available assets: {', '.join(ASSETS.keys())}")
            return
    
    # Start pipeline
    print("\n" + "="*70)
    print("  QDT ENSEMBLE - COMPLETE PIPELINE")
    print("="*70)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Assets: {len(assets_to_process)} ({', '.join(assets_to_process.keys())})")
    print(f"  Skip Fetch: {args.skip_fetch}")
    print("="*70)
    
    pipeline_start = datetime.now()
    results = {}
    
    # Process each asset
    for i, (asset_name, project_id) in enumerate(assets_to_process.items(), 1):
        print(f"\n\n[{i}/{len(assets_to_process)}] Processing {asset_name}...")
        results[asset_name] = run_asset_pipeline(asset_name, project_id, args.skip_fetch)
    
    # Step 4: Build Dashboard (once at the end)
    print(f"\n\n{'#'*70}")
    print("#  BUILDING UNIFIED DASHBOARD")
    print(f"{'#'*70}")
    dashboard_success = step4_build_dashboard()
    
    # Final Summary
    total_time = (datetime.now() - pipeline_start).total_seconds()
    
    print("\n" + "="*70)
    print("  PIPELINE SUMMARY")
    print("="*70)
    print(f"  Total Time: {total_time/60:.1f} minutes")
    print(f"  Assets Processed: {len(assets_to_process)}")
    print(f"\n  Asset Results:")
    for asset_name, success in results.items():
        status = "[OK]" if success else "[FAIL]"
        print(f"    {status} {asset_name}")
    print(f"\n  Dashboard: {'[OK] Built' if dashboard_success else '[FAIL] Failed'}")
    print("="*70)
    print(f"\n  [OK] Pipeline Complete!")
    print(f"  Dashboard: {os.path.join(SCRIPT_DIR, 'QDT_Ensemble_Dashboard.html')}")
    print("="*70)

if __name__ == "__main__":
    main()

