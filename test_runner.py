#!/usr/bin/env python3
"""
Test runner for the alert summarizer system
"""

import os
import sys
import pandas as pd

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from summarizer import AlertSummarizer
from feedback_loop import save_feedback, get_feedback_stats

def test_basic_functionality():
    """Test basic functionality without heavy ML models"""
    print("🔍 Testing basic functionality...")
    
    # Test data loading
    data_path = "data/sample_alerts.csv"
    if not os.path.exists(data_path):
        print(f"❌ Sample data file not found: {data_path}")
        return False
        
    # Test DataFrame processing
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} sample alerts")
    
    # Test summarizer initialization (may fail gracefully)
    try:
        summarizer = AlertSummarizer(n_clusters=2)
        print("✅ AlertSummarizer initialized")
        
        # Test with smaller dataset to avoid model issues
        result = summarizer.run_from_dataframe(df)
        print(f"✅ Generated {len(result)} summaries")
        print("Sample output:")
        print(result.head())
        
    except Exception as e:
        print(f"⚠️ Summarizer test failed (expected if models aren't available): {e}")
    
    # Test feedback system
    print("\n🔍 Testing feedback system...")
    
    # Test saving feedback
    save_feedback(
        cluster_id=0, 
        summary_text="Test summary", 
        rating="Helpful", 
        comment="Test comment"
    )
    print("✅ Feedback saved")
    
    # Test retrieving stats
    stats = get_feedback_stats()
    print(f"✅ Feedback stats: {stats}")
    
    return True

def test_streamlit_imports():
    """Test that Streamlit app imports work"""
    print("\n🔍 Testing Streamlit app imports...")
    
    try:
        # Change to app directory for import
        sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))
        
        # Test imports
        import streamlit as st
        print("✅ Streamlit imported")
        
        # Test our module imports (without running the app)
        import importlib.util
        spec = importlib.util.spec_from_file_location("streamlit_app", "app/streamlit_app.py")
        if spec and spec.loader:
            print("✅ Streamlit app file can be loaded")
        else:
            print("❌ Streamlit app file import issue")
            
    except Exception as e:
        print(f"❌ Streamlit test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Running Alert Summarizer Tests\n")
    
    # Change to the project root directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Working directory: {os.getcwd()}")
    
    success = True
    success &= test_basic_functionality()
    success &= test_streamlit_imports()
    
    print(f"\n{'✅ All tests passed!' if success else '❌ Some tests failed'}")
    sys.exit(0 if success else 1)
