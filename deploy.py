#!/usr/bin/env python3
"""
Deployment script for Alert Summarizer Agent
This script helps deploy and run the application with optimal settings.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

ROOT_DIR = Path(__file__).parent.absolute()
VENV_PATH = ROOT_DIR / "venv"
REQUIREMENTS_FILE = ROOT_DIR / "requirements.txt"
STREAMLIT_APP = ROOT_DIR / "app" / "streamlit_app.py"

def check_python():
    """Check if Python is available and get version"""
    try:
        result = subprocess.run([sys.executable, "--version"], capture_output=True, text=True)
        print(f"✅ Python found: {result.stdout.strip()}")
        return True
    except Exception as e:
        print(f"❌ Python check failed: {e}")
        return False

def check_venv():
    """Check if virtual environment exists"""
    if VENV_PATH.exists():
        print(f"✅ Virtual environment found at: {VENV_PATH}")
        return True
    else:
        print(f"❌ Virtual environment not found at: {VENV_PATH}")
        return False

def create_venv():
    """Create virtual environment"""
    print("🔨 Creating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", str(VENV_PATH)], check=True)
        print("✅ Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def get_python_executable():
    """Get the Python executable path for the virtual environment"""
    if os.name == 'nt':  # Windows
        return VENV_PATH / "Scripts" / "python.exe"
    else:  # Unix/Linux/macOS
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
        subprocess.run([str(python_exe), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)], check=True)
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def run_tests():
    """Run the test suite"""
    python_exe = get_python_executable()
    test_file = ROOT_DIR / "test_runner.py"
    
    print("🧪 Running tests...")
    try:
        result = subprocess.run([str(python_exe), str(test_file)], 
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

def run_streamlit():
    """Run the Streamlit application"""
    python_exe = get_python_executable()
    
    print("🚀 Starting Streamlit application...")
    print(f"📱 The app will open in your browser at: http://localhost:8501")
    print("⚠️  Press Ctrl+C to stop the application")
    
    try:
        # Set environment variables for better performance
        env = os.environ.copy()
        env['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid tokenizer warnings
        
        subprocess.run([
            str(python_exe), "-m", "streamlit", "run", str(STREAMLIT_APP),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ], cwd=str(ROOT_DIR), env=env)
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Failed to run Streamlit: {e}")

def check_models():
    """Check if required models are available"""
    python_exe = get_python_executable()
    
    print("🔍 Checking model availability...")
    check_script = '''
import warnings
warnings.filterwarnings("ignore")

try:
    from sentence_transformers import SentenceTransformer
    print("✅ sentence-transformers available")
    
    # Test loading a small model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Embedding model loaded successfully")
    
    from transformers import pipeline
    print("✅ transformers available")
    
    print("🎯 Models are ready for use!")
    
except Exception as e:
    print(f"❌ Model check failed: {e}")
'''
    
    try:
        result = subprocess.run([str(python_exe), "-c", check_script], 
                              capture_output=True, text=True, cwd=str(ROOT_DIR))
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Model check failed: {e}")
        return False

def show_help():
    """Show help information"""
    help_text = """
🚀 Alert Summarizer Agent - Deployment Script

Usage: python deploy.py [command]

Commands:
  setup     - Full setup (create venv, install requirements)
  test      - Run the test suite
  run       - Start the Streamlit application
  check     - Check system requirements and models
  help      - Show this help message

Examples:
  python deploy.py setup    # First-time setup
  python deploy.py test     # Run tests
  python deploy.py run      # Start the application
  python deploy.py check    # Check if everything is ready

📋 Requirements:
  - Python 3.8 or higher
  - At least 4GB RAM (for model loading)
  - Internet connection (for downloading models)

🎯 Quick Start:
  1. python deploy.py setup
  2. python deploy.py test
  3. python deploy.py run
"""
    print(help_text)

def main():
    """Main deployment function"""
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    print("🚀 Alert Summarizer Agent - Deployment")
    print("=" * 50)
    
    if command == "help":
        show_help()
    
    elif command == "setup":
        print("🔧 Setting up the application...")
        
        if not check_python():
            return
        
        if not check_venv():
            if not create_venv():
                return
        
        if not install_requirements():
            return
        
        print("\n✅ Setup completed successfully!")
        print("📋 Next steps:")
        print("  1. Run tests: python deploy.py test")
        print("  2. Start app: python deploy.py run")
    
    elif command == "test":
        if not check_venv():
            print("❌ Please run 'python deploy.py setup' first")
            return
        
        if not run_tests():
            return
        
        print("\n✅ Testing completed successfully!")
        print("🚀 Ready to run: python deploy.py run")
    
    elif command == "run":
        if not check_venv():
            print("❌ Please run 'python deploy.py setup' first")
            return
        
        run_streamlit()
    
    elif command == "check":
        print("🔍 Checking system requirements...")
        
        checks = [
            ("Python", check_python()),
            ("Virtual Environment", check_venv()),
            ("Models", check_models() if check_venv() else False),
        ]
        
        print("\n📊 System Check Results:")
        for name, status in checks:
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {name}")
        
        if all(status for _, status in checks):
            print("\n🎉 All checks passed! Ready to run the application.")
        else:
            print("\n⚠️  Some checks failed. Please run 'python deploy.py setup'")
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Run 'python deploy.py help' for available commands")

if __name__ == "__main__":
    main()
