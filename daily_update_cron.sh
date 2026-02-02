#!/bin/bash
# QDT Ensemble Dashboard - Daily Update Cron Script
# Runs at 6 AM daily via cron
# Updates all assets and rebuilds dashboard

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Log file with date
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/daily_update_$(date +%Y%m%d_%H%M%S).log"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "========================================"
log "QDT Ensemble Daily Update Started"
log "========================================"
log "Script Directory: $SCRIPT_DIR"
log "Log File: $LOG_FILE"

# Activate Python environment if venv exists
if [ -d "venv" ]; then
    log "Activating Python virtual environment..."
    source venv/bin/activate
elif [ -d "../venv" ]; then
    log "Activating Python virtual environment (parent directory)..."
    source ../venv/bin/activate
else
    log "[WARN] No virtual environment found, using system Python"
fi

# Check Python version
PYTHON_CMD=$(which python3 || which python)
log "Using Python: $PYTHON_CMD"
$PYTHON_CMD --version | tee -a "$LOG_FILE"

# Run the complete pipeline
log "Running complete pipeline (run_complete_pipeline.py)..."
$PYTHON_CMD run_complete_pipeline.py >> "$LOG_FILE" 2>&1
PIPELINE_EXIT_CODE=$?

if [ $PIPELINE_EXIT_CODE -eq 0 ]; then
    log "[OK] Pipeline completed successfully"
else
    log "[FAIL] Pipeline failed with exit code $PIPELINE_EXIT_CODE"
    # Continue anyway to try to deploy existing dashboard
fi

# Check if dashboard was generated
DASHBOARD_FILE="$SCRIPT_DIR/QDT_Ensemble_Dashboard.html"
if [ -f "$DASHBOARD_FILE" ]; then
    log "[OK] Dashboard file found: $DASHBOARD_FILE"
    
    # Copy to web server if path is configured
    # Update these paths based on your server setup
    WEB_DIR="/var/www/html"
    if [ -d "$WEB_DIR" ]; then
        log "Copying dashboard to web directory: $WEB_DIR"
        cp "$DASHBOARD_FILE" "$WEB_DIR/QDT_Ensemble_Dashboard.html"
        if [ $? -eq 0 ]; then
            log "[OK] Dashboard deployed to web server"
        else
            log "[FAIL] Failed to copy dashboard to web server"
        fi
    else
        log "[INFO] Web directory not found ($WEB_DIR), skipping deployment"
        log "[INFO] Dashboard available at: $DASHBOARD_FILE"
    fi
else
    log "[WARN] Dashboard file not found: $DASHBOARD_FILE"
fi

log "========================================"
log "Daily Update Completed"
log "========================================"
log "Check log file for details: $LOG_FILE"

# Exit with pipeline exit code
exit $PIPELINE_EXIT_CODE

