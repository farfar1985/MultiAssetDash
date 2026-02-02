# q_ensemble_sandbox/run_sandbox_pipeline.py
# MULTI-PROJECT SANDBOXED RUNNER
# Does NOT touch the Crude Queen pipeline
import os
import sys
import argparse

# Add sandbox root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def run_full_pipeline(project_id: str, project_name: str, skip_fetch: bool = False):
    """Run the full Q Ensemble pipeline for a specific project."""
    
    # Import and configure AFTER setting project
    import config_sandbox as cfg
    cfg.set_project(project_id, project_name)
    
    # Now import the modules (they'll use the updated config)
    import fetch_all_children
    import golden_engine
    import run_dynamic_quantile
    import matrix_drift_analysis
    import plot_persona_dashboard
    
    print("\n" + "="*60)
    print(f"  Q ENSEMBLE SANDBOX - {cfg.PROJECT_NAME}")
    print(f"  Project ID: {cfg.PROJECT_ID}")
    print(f"  Data Dir: {cfg.DATA_DIR}")
    print("="*60 + "\n")
    
    # 1. Fetch Data
    if not skip_fetch:
        print("\n[Step 1/5] Fetching Models from API...")
        result = fetch_all_children.fetch_all_models_bulk()
        if result is None:
            print("!!! PIPELINE ABORTED: Failed to fetch data.")
            return False
    else:
        print("\n[Step 1/5] Skipping Fetch (--skip-fetch flag)...")
    
    # 2. Prepare Horizon Data
    print("\n[Step 2/5] Preparing Horizon Data...")
    golden_engine.prepare_horizon_data()
    
    # 3. Run Dynamic Quantile Ensemble
    print("\n[Step 3/5] Running Q Ensemble (this may take a while)...")
    run_dynamic_quantile.run_all()
    
    # 4. Calculate Drift Signals
    print("\n[Step 4/5] Calculating Matrix Drift Signals...")
    matrix_drift_analysis.calculate_matrix_drift()
    
    # 5. Generate Dashboard
    print("\n[Step 5/5] Generating Snake Chart Dashboard...")
    plot_persona_dashboard.generate_persona_dashboard()
    
    print("\n" + "="*60)
    print(f"  {cfg.PROJECT_NAME} PIPELINE COMPLETE!")
    print("="*60)
    print(f"\nOutputs saved to: {cfg.DATA_DIR}")
    print(f"  - live_forecast.json           (Live predictions)")
    print(f"  - matrix_drift_signals.csv     (Historical signals)")
    print(f"  - persona_dashboard.html       (Snake Chart)")
    print(f"\nOpen the dashboard:")
    print(f"  {os.path.join(cfg.DATA_DIR, 'persona_dashboard.html')}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Q Ensemble for any project")
    parser.add_argument('--project', '-p', type=str, default='1860', 
                        help='Project ID (e.g., 1860)')
    parser.add_argument('--name', '-n', type=str, default='Bitcoin',
                        help='Project name for labels (e.g., Bitcoin, Gold, NatGas)')
    parser.add_argument('--skip-fetch', action='store_true', 
                        help='Skip downloading new data (use existing)')
    args = parser.parse_args()
    
    run_full_pipeline(
        project_id=args.project,
        project_name=args.name,
        skip_fetch=args.skip_fetch
    )
