@echo off
echo Starting Alert Summarizer Agent...
cd /d "%~dp0"
streamlit run app/streamlit_app.py --server.port 8501
pause
