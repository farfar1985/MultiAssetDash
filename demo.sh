#!/bin/bash
#
# QDT Nexus Demo Launcher
# One-click demo for Rajiv & Ale
#
# Usage: ./demo.sh
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo -e "${BOLD}${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${BLUE}║           QDT Nexus - Demo Launcher                        ║${NC}"
echo -e "${BOLD}${BLUE}║           Quantum Decision Theory Dashboard                ║${NC}"
echo -e "${BOLD}${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down servers...${NC}"
    if [ ! -z "$API_PID" ]; then
        kill $API_PID 2>/dev/null || true
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    echo -e "${GREEN}Cleanup complete.${NC}"
}

trap cleanup EXIT

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}Error: Python virtual environment not found.${NC}"
    echo "Please run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Check if node_modules exists
if [ ! -d "frontend/node_modules" ]; then
    echo -e "${RED}Error: Frontend dependencies not installed.${NC}"
    echo "Please run: cd frontend && npm install"
    exit 1
fi

# Kill any existing processes on our ports
echo -e "${YELLOW}Checking for existing processes...${NC}"
pkill -f "python api_server.py" 2>/dev/null || true
pkill -f "next dev" 2>/dev/null || true
sleep 1

# Start API server
echo -e "${BLUE}Starting API server on port 5000...${NC}"
source .venv/bin/activate
python api_server.py > /tmp/qdt_api.log 2>&1 &
API_PID=$!
sleep 2

# Verify API server is running
if curl -s http://localhost:5000/api/v1/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ API server running${NC}"
else
    echo -e "${RED}✗ API server failed to start. Check /tmp/qdt_api.log${NC}"
    exit 1
fi

# Start frontend
echo -e "${BLUE}Starting frontend on port 3000...${NC}"
cd frontend
npm run dev > /tmp/qdt_frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

# Wait for frontend to be ready
echo -e "${YELLOW}Waiting for frontend to compile...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Frontend ready${NC}"
        break
    fi
    sleep 1
    echo -n "."
done
echo ""

# Verify frontend is running
if ! curl -s http://localhost:3000 > /dev/null 2>&1; then
    echo -e "${RED}✗ Frontend failed to start. Check /tmp/qdt_frontend.log${NC}"
    exit 1
fi

# Open browser
echo -e "${BLUE}Opening browser...${NC}"
DASHBOARD_URL="http://localhost:3000/dashboard/executive"

# Detect OS and open browser accordingly
if [[ "$OSTYPE" == "darwin"* ]]; then
    open "$DASHBOARD_URL"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if command -v xdg-open &> /dev/null; then
        xdg-open "$DASHBOARD_URL" 2>/dev/null || true
    elif command -v wslview &> /dev/null; then
        wslview "$DASHBOARD_URL" 2>/dev/null || true
    else
        echo -e "${YELLOW}Please open manually: ${DASHBOARD_URL}${NC}"
    fi
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    start "$DASHBOARD_URL"
fi

# Print success message and navigation guide
echo ""
echo -e "${BOLD}${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║                    DEMO READY!                             ║${NC}"
echo -e "${BOLD}${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BOLD}Dashboard URLs:${NC}"
echo ""
echo -e "  ${BOLD}Executive Dashboard${NC} (opened automatically)"
echo -e "  ${BLUE}http://localhost:3000/dashboard/executive${NC}"
echo ""
echo -e "  ${BOLD}Quant Dashboard${NC} - Advanced quantitative analysis"
echo -e "  ${BLUE}http://localhost:3000/dashboard/quant${NC}"
echo ""
echo -e "  ${BOLD}Hedge Fund Dashboard${NC} - Alpha generation & portfolio"
echo -e "  ${BLUE}http://localhost:3000/dashboard/hedgefund${NC}"
echo ""
echo -e "  ${BOLD}Procurement Dashboard${NC} - Strategic sourcing"
echo -e "  ${BLUE}http://localhost:3000/dashboard/procurement${NC}"
echo ""
echo -e "  ${BOLD}Hedging Dashboard${NC} - Risk management"
echo -e "  ${BLUE}http://localhost:3000/dashboard/hedging${NC}"
echo ""
echo -e "${BOLD}API Endpoints:${NC}"
echo ""
echo -e "  Health:  ${BLUE}http://localhost:5000/api/v1/health${NC}"
echo -e "  Assets:  ${BLUE}http://localhost:5000/api/v1/assets${NC} (requires API key)"
echo -e "  Signals: ${BLUE}http://localhost:5000/api/v1/signals/crude-oil${NC} (requires API key)"
echo ""
echo -e "${BOLD}${YELLOW}Press Ctrl+C to stop the demo servers${NC}"
echo ""

# Keep script running
wait
