# app/streamlit_app.py

import sys
import os
import streamlit as st
import pandas as pd

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.summarizer import AlertSummarizer
from src.feedback_loop import save_feedback, get_feedback_stats

st.set_page_config(page_title="Alert Summarizer Agent", layout="wide")
st.title("⚡ Auto-Triage and Alert Summarizer Agent")

# Upload alert CSV
uploaded_file = st.file_uploader("Upload alert CSV", type=["csv"])
if uploaded_file:
    # Read the uploaded file directly
    df = pd.read_csv(uploaded_file)
    st.write("📋 Sample Alerts:")
    st.dataframe(df.head())

    # Save the uploaded file temporarily and pass the dataframe to summarizer
    summarizer = AlertSummarizer(n_clusters=3)
    summaries_df = summarizer.run_from_dataframe(df)

    st.markdown("---")
    st.subheader("🧠 Summarized Alert Clusters")

    for idx, row in summaries_df.iterrows():
        with st.expander(f"Cluster {row['cluster']}"):
            st.write(f"**Summary:** {row['summary']}")

            # Feedback form
            rating = st.radio(
                f"Was this summary helpful? (Cluster {row['cluster']})",
                ["Helpful", "Not Helpful"],
                key=f"rating_{idx}"
            )
            comment = st.text_area(
                f"Optional comment for Cluster {row['cluster']}",
                key=f"comment_{idx}"
            )
            if st.button(f"Submit Feedback for Cluster {row['cluster']}", key=f"submit_{idx}"):
                save_feedback(
                    cluster_id=int(row['cluster']),
                    summary_text=row['summary'],
                    rating=rating,
                    comment=comment
                )
                st.success("✅ Feedback submitted!")

    # Display aggregate feedback stats
    st.markdown("---")
    st.subheader("📊 Feedback Summary")
    stats = get_feedback_stats()
    if stats:
        for cluster_id, counts in stats.items():
            st.write(f"Cluster {cluster_id}: ✅ {counts['helpful']} helpful, ❌ {counts['not_helpful']} not helpful")
    else:
        st.write("No feedback submitted yet.")
