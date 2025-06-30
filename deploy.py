#!/usr/bin/env python3
"""
Simple deployment script for Alert Summarizer Agent

python deploy.py setup    # Create venv and install dependencies
python deploy.py test     # Run tests
python deploy.py run      # Start the application

"""

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.absolute()
VENV_PATH = ROOT_DIR / "venv"

def create_venv():
    """Create virtual environment"""
    print("🔨 Creating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", str(VENV_PATH)], check=True)
        print("✅ Virtual environment created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def get_python_executable():
    """Get the Python executable path for the virtual environment"""
    if sys.platform == "win32":
        return VENV_PATH / "Scripts" / "python.exe"
    else:
        return VENV_PATH / "bin" / "python"

def install_requirements():
    """Install required packages"""
    python_exe = get_python_executable()
    if not python_exe.exists():
        print(f"❌ Python executable not found: {python_exe}")
        return False
    
    print("📦 Installing requirements...")
    try:
        subprocess.run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([str(python_exe), "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Requirements installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def run_tests():
    """Run the test suite"""
    python_exe = get_python_executable()
    print("🧪 Running tests...")
    try:
        result = subprocess.run([str(python_exe), "test_runner.py"], 
                              capture_output=True, text=True, cwd=str(ROOT_DIR))
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        
        if result.returncode == 0:
            print("✅ All tests passed!")
            return True
        else:
            print("❌ Some tests failed!")
            return False
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False

def run_app():
    """Run the Streamlit application"""
    python_exe = get_python_executable()
    print("🚀 Starting Streamlit application...")
    print("📱 The app will open at: http://localhost:8501")
    print("⚠️  Press Ctrl+C to stop")
    
    try:
        subprocess.run([
            str(python_exe), "-m", "streamlit", "run", "app/streamlit_app.py",
            "--server.port", "8501"
        ], cwd=str(ROOT_DIR))
    except KeyboardInterrupt:
        print("\n🛑 Application stopped")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python deploy.py [setup|test|run]")
        print("  setup - Create venv and install requirements")
        print("  test  - Run test suite")
        print("  run   - Start Streamlit app")
        return
    
    command = sys.argv[1].lower()
    
    if command == "setup":
        print("🔧 Setting up Alert Summarizer Agent...")
        if not VENV_PATH.exists() and not create_venv():
            return
        if not install_requirements():
            return
        print("✅ Setup complete! Run 'python deploy.py test' to verify.")
        
    elif command == "test":
        if not VENV_PATH.exists():
            print("❌ Run 'python deploy.py setup' first")
            return
        run_tests()
        
    elif command == "run":
        if not VENV_PATH.exists():
            print("❌ Run 'python deploy.py setup' first")
            return
        run_app()
        
    else:
        print(f"❌ Unknown command: {command}")

if __name__ == "__main__":
    main()
