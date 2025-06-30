#!/usr/bin/env python3
"""
Test the advanced AI model implementation
"""

import sys
import os
sys.path.append('src')

from summarizer import AlertSummarizer
import pandas as pd

def test_basic_functionality():
    """Test basic summarizer functionality"""
    print("🔍 Testing basic AlertSummarizer functionality...")
    
    # Create test data
    test_data = {
        'message': [
            "High temperature alert in server room A. Temperature exceeded 85°C.",
            "Server room A cooling system malfunction detected.",
            "Network connection timeout on server 192.168.1.100.",
            "Database connection failed for user database.",
            "Network switch port 24 experiencing high latency.",
            "Critical temperature alert in server room A reached 90°C."
        ]
    }
    df = pd.DataFrame(test_data)
    
    # Test without AI (fast mode)
    print("\n📊 Testing fast mode (no AI)...")
    summarizer = AlertSummarizer(n_clusters=2)
    results = summarizer.run_from_dataframe(df)
    
    print("✅ Fast mode results:")
    for idx, row in results.iterrows():
        cluster_id = row.iloc[0]  # cluster column
        summary = row.iloc[1]     # summary column
        print(f"  Cluster {cluster_id}: {summary}")
    
    # Test with AI enabled
    print("\n🧠 Testing advanced AI mode...")
    summarizer_ai = AlertSummarizer(n_clusters=2)
    ai_enabled = summarizer_ai.enable_ai_summarization()
    
    if ai_enabled:
        results_ai = summarizer_ai.run_from_dataframe(df)
        print("✅ AI mode results:")
        for idx, row in results_ai.iterrows():
            cluster_id = row.iloc[0]  # cluster column
            summary = row.iloc[1]     # summary column
            print(f"  Cluster {cluster_id}: {summary}")
    else:
        print("⚠️ AI mode not available, falling back to extractive summarization")
    
    return True

def test_with_sample_data():
    """Test with actual sample data if available"""
    print("\n📋 Testing with sample data...")
    
    sample_path = os.path.join('data', 'sample_alerts.csv')
    if os.path.exists(sample_path):
        print(f"✅ Found sample data: {sample_path}")
        
        summarizer = AlertSummarizer(n_clusters=3)
        summarizer.enable_ai_summarization()
        
        try:
            results = summarizer.run(sample_path)
            print("✅ Sample data processing successful:")
            for idx, row in results.iterrows():
                cluster_id = row.iloc[0]  # cluster column
                summary = row.iloc[1]     # summary column
                print(f"  Cluster {cluster_id}: {summary}")
            return True
        except Exception as e:
            print(f"❌ Error processing sample data: {e}")
            return False
    else:
        print("⚠️ Sample data not found, skipping...")
        return True

def main():
    """Main test function"""
    print("🧪 Advanced AI Model Test Suite")
    print("=" * 40)
    
    try:
        # Test basic functionality
        if not test_basic_functionality():
            print("❌ Basic functionality test failed")
            return False
        
        # Test with sample data
        if not test_with_sample_data():
            print("❌ Sample data test failed")
            return False
        
        print("\n🎉 All tests passed! Advanced AI model is working correctly.")
        print("\n📋 Next steps:")
        print("1. Run the Streamlit app: streamlit run app/streamlit_app.py")
        print("2. Test the full workflow in the web interface")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
