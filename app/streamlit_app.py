# app/streamlit_app.py

import streamlit as st
import pandas as pd
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from summarizer import AlertSummarizer
from feedback_loop import save_feedback, get_feedback_stats

st.set_page_config(page_title="Alert Summarizer Agent", layout="wide", page_icon="⚡")
st.title("⚡ Auto-Triage and Alert Summarizer Agent")
st.markdown("---")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    n_clusters = st.slider("Number of Alert Clusters", min_value=2, max_value=10, value=3)
    st.markdown("---")
    st.header("About")
    st.markdown("""
    This tool automatically:
    1. 📊 Groups similar alerts using AI clustering
    2. 🧠 Generates summaries for each group
    3. 💬 Collects feedback for improvement
    """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Upload alert CSV
    uploaded_file = st.file_uploader("Upload alert CSV file", type=["csv"], 
                                    help="CSV should contain a 'message' column with alert text")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validate CSV structure
            if "message" not in df.columns:
                st.error("❌ CSV file must contain a 'message' column")
                st.stop()
            
            # Display sample data
            st.success(f"✅ Loaded {len(df)} alerts")
            
            with st.expander("📋 Preview Alert Data", expanded=False):
                st.dataframe(df.head(10))
            
            # Process alerts
            if st.button("🚀 Analyze Alerts", type="primary"):
                with st.spinner("Processing alerts..."):
                    try:
                        summarizer = AlertSummarizer(n_clusters=n_clusters)
                        summaries_df = summarizer.run_from_dataframe(df)
                        
                        # Store results in session state
                        st.session_state.summaries_df = summaries_df
                        st.session_state.original_df = df
                        
                        st.success("✅ Analysis complete!")
                        
                    except Exception as e:
                        st.error(f"❌ Error during processing: {str(e)}")

        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")

with col2:
    # Display feedback statistics
    st.subheader("📊 Feedback Summary")
    stats = get_feedback_stats()
    if stats:
        for cluster_id, counts in stats.items():
            total = counts['helpful'] + counts['not_helpful']
            helpful_pct = (counts['helpful'] / total * 100) if total > 0 else 0
            st.metric(
                f"Cluster {cluster_id}",
                f"{helpful_pct:.0f}% helpful",
                f"{total} total reviews"
            )
    else:
        st.info("No feedback submitted yet")

# Display results if available
if 'summaries_df' in st.session_state:
    st.markdown("---")
    st.subheader("🧠 Alert Cluster Summaries")
    
    summaries_df = st.session_state.summaries_df
    original_df = st.session_state.original_df
    
    # Add alert count per cluster
    cluster_counts = original_df.groupby(
        original_df.index.map(lambda x: summaries_df.iloc[x % len(summaries_df)]['cluster'])
    ).size()
    
    for idx, row in summaries_df.iterrows():
        cluster_id = row['cluster']
        alert_count = cluster_counts.get(cluster_id, 0)
        
        with st.expander(f"📝 Cluster {cluster_id} ({alert_count} alerts)", expanded=True):
            st.write(f"**Summary:** {row['summary']}")
            
            # Show sample alerts from this cluster
            if st.checkbox(f"Show sample alerts from Cluster {cluster_id}", key=f"show_alerts_{idx}"):
                # Get alerts for this cluster (simplified approach)
                sample_alerts = original_df['message'].head(3).tolist()
                for i, alert in enumerate(sample_alerts, 1):
                    st.text(f"{i}. {alert}")
            
            # Feedback form
            col_rating, col_comment = st.columns([1, 2])
            
            with col_rating:
                rating = st.radio(
                    "Was this summary helpful?",
                    ["Helpful", "Not Helpful"],
                    key=f"rating_{idx}",
                    horizontal=True
                )
            
            with col_comment:
                comment = st.text_area(
                    "Optional comment",
                    key=f"comment_{idx}",
                    height=68,
                    placeholder="Any suggestions for improvement?"
                )
            
            if st.button(f"Submit Feedback", key=f"submit_{idx}", type="secondary"):
                success = save_feedback(
                    cluster_id=int(cluster_id),
                    summary_text=row['summary'],
                    rating=rating,
                    comment=comment if comment.strip() else None
                )
                if success:
                    st.success("✅ Thank you for your feedback!")
                    # Rerun to update stats
                    st.rerun()
                else:
                    st.error("❌ Failed to save feedback. Please try again.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit • Powered by AI")
