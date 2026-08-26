@echo off
echo Starting Normative Threat Dashboard...

cd /d "C:\Users\omkar\.gemini\antigravity\scratch\backend-repo"

echo [1/2] Starting Backend Services and Simulator (Docker)...
docker-compose up -d

echo [2/2] Starting Frontend UI...
cd frontend
start npm run dev

echo.
echo ========================================================
echo The dashboard will automatically open in your web browser.
echo If it does not, go to: http://localhost:5173
echo ========================================================
pause
