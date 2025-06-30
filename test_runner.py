#!/usr/bin/env python3
"""
Test runner for the Alert Summarizer Agent
"""

import os
import sys
import json
import pandas as pd

# Add necessary paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
APP_DIR = os.path.join(ROOT_DIR, "app")
DATA_DIR = os.path.join(ROOT_DIR, "data")

sys.path.append(SRC_DIR)
sys.path.append(APP_DIR)

from summarizer_lite import AlertSummarizerLite
from summarizer import AlertSummarizer  # Import the full summarizer too
from feedback_loop import save_feedback, load_feedback, get_feedback_stats

SAMPLE_ALERT_PATH = os.path.join(DATA_DIR, "sample_alerts.csv")
FEEDBACK_LOG_PATH = os.path.join(DATA_DIR, "feedback_log.json")


def reset_feedback_log():
    """Reset feedback file for clean testing"""
    if os.path.exists(FEEDBACK_LOG_PATH):
        os.remove(FEEDBACK_LOG_PATH)
    with open(FEEDBACK_LOG_PATH, "w") as f:
        json.dump([], f)
    print("🧹 Cleared existing feedback log.")


def test_data_loading():
    """Test if sample alerts can be loaded"""
    print("🔍 Testing data loading...")
    assert os.path.exists(SAMPLE_ALERT_PATH), f"❌ Sample alert file missing: {SAMPLE_ALERT_PATH}"
    
    df = pd.read_csv(SAMPLE_ALERT_PATH)
    assert not df.empty, "❌ DataFrame is empty"
    assert "message" in df.columns, "❌ 'message' column is missing"
    
    print(f"✅ Loaded {len(df)} alerts with 'message' column.")
    return df


def test_summarization(df):
    """Test the summarization process"""
    print("\n🧠 Testing summarization...")
    
    summarizer = AlertSummarizerLite(n_clusters=2)
    
    try:
        result = summarizer.run_from_dataframe(df)
    except AttributeError:
        # fallback in case run_from_dataframe isn't defined
        print("⚠️ Falling back to run() method...")
        result = summarizer.run(SAMPLE_ALERT_PATH)

    assert not result.empty, "❌ Summarization returned an empty DataFrame"
    assert "summary" in result.columns, "❌ Missing 'summary' column in result"
    for summary in result["summary"]:
        assert isinstance(summary, str) and len(summary) > 10, f"❌ Invalid summary: {summary}"
    
    print(f"✅ {len(result)} summaries generated.")
    print("📋 Sample summary:\n", result.iloc[0]["summary"])
    return result


def test_feedback_system():
    """Test saving and retrieving feedback"""
    print("\n💬 Testing feedback loop...")

    save_feedback(
        cluster_id=0,
        summary_text="Test summary for cluster 0",
        rating="Helpful",
        comment="Works well"
    )

    feedback = load_feedback()
    assert isinstance(feedback, list), "❌ Feedback is not a list"
    assert any(f["summary"] == "Test summary for cluster 0" for f in feedback), "❌ Feedback entry not found"

    stats = get_feedback_stats()
    assert isinstance(stats, dict), "❌ Feedback stats not returned as dictionary"
    
    print(f"✅ Feedback entries: {len(feedback)}")
    print(f"📊 Feedback stats: {stats}")


def test_streamlit_app():
    """Check if Streamlit app loads correctly"""
    print("\n🖥️ Testing Streamlit app file...")

    app_path = os.path.join(APP_DIR, "streamlit_app.py")
    assert os.path.exists(app_path), f"❌ Streamlit app not found at {app_path}"

    try:
        import streamlit as st
        print("✅ Streamlit module imported")

        with open(app_path, "r", encoding='utf-8') as f:
            content = f.read()
            assert "st.file_uploader" in content, "❌ Missing expected Streamlit UI components"
            assert "st.expander" in content, "❌ Missing expandable summaries"
            assert "save_feedback" in content, "❌ Feedback logic not connected in UI"

        print("✅ Streamlit app structure is valid.")
    except Exception as e:
        raise AssertionError(f"❌ Streamlit test failed: {e}")


def run_all_tests():
    """Main test sequence"""
    print("🚀 Running Alert Summarizer Test Suite\n")
    os.chdir(ROOT_DIR)
    print(f"📂 Working directory: {os.getcwd()}")

    success = True

    try:
        reset_feedback_log()
        df = test_data_loading()
        test_summarization(df)
        test_feedback_system()
        test_streamlit_app()
    except AssertionError as e:
        print(e)
        success = False

    print("\n🧪 Test Results:")
    print("✅ All tests passed!" if success else "❌ Some tests failed.")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    run_all_tests()
