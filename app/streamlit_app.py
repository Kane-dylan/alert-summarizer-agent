# app/streamlit_app.py

import streamlit as st
import pandas as pd
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from summarizer_lite import AlertSummarizerLite
from summarizer import AlertSummarizer
from feedback_loop import save_feedback, get_feedback_stats

st.set_page_config(page_title="Alert Summarizer Agent", layout="wide", page_icon="⚡")
st.title("⚡ Auto-Triage and Alert Summarizer Agent")
st.markdown("---")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    n_clusters = st.slider("Number of Alert Clusters", min_value=2, max_value=10, value=3)
    
    # Model selection
    st.subheader("🧠 Summarization Model")
    use_full_model = st.checkbox(
        "Use Advanced AI Model", 
        value=False,
        help="Enable for better summaries using BART model (slower but higher quality)"
    )
    
    if use_full_model:
        st.info("⚡ Advanced model selected - first run may take longer to download models")
    else:
        st.info("🚀 Fast mode selected - uses TF-IDF clustering")
    
    st.markdown("---")
    
    # Sample data section
    st.header("📋 Sample Data")
    use_sample_data = st.checkbox("Use Sample Dataset", help="Load the built-in sample alerts for testing")
    
    if use_sample_data:
        sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sample_alerts.csv')
        if os.path.exists(sample_path):
            st.success("✅ Sample dataset available")
            if st.button("Load Sample Data"):
                st.session_state.use_sample = True
                st.rerun()
        else:
            st.error("❌ Sample dataset not found")
    
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
    # Handle sample data loading
    uploaded_file = None
    df = None
    
    if st.session_state.get('use_sample', False):
        sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sample_alerts.csv')
        try:
            df = pd.read_csv(sample_path)
            st.success(f"✅ Loaded sample dataset with {len(df)} alerts")
            st.session_state.use_sample = False  # Reset flag
        except Exception as e:
            st.error(f"❌ Error loading sample data: {str(e)}")
            st.session_state.use_sample = False
    else:
        # Upload alert CSV
        uploaded_file = st.file_uploader("Upload alert CSV file", type=["csv"], 
                                        help="CSV should contain a 'message' column with alert text")
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"❌ Error reading CSV file: {str(e)}")
    
    if df is not None:
        # Validate CSV structure
        if "message" not in df.columns:
            st.error("❌ CSV file must contain a 'message' column")
            st.stop()
        
        # Display dataset info
        st.success(f"✅ Loaded {len(df)} alerts")
        
        # Show dataset statistics for large datasets
        if len(df) > 100:
            st.info(f"📊 Large dataset detected ({len(df)} rows). Processing may take a moment.")
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Total Alerts", len(df))
            with col_stat2:
                avg_length = df['message'].str.len().mean()
                st.metric("Avg Message Length", f"{avg_length:.0f} chars")
            with col_stat3:
                unique_messages = df['message'].nunique()
                st.metric("Unique Messages", unique_messages)
        
        with st.expander("📋 Preview Alert Data", expanded=False):
            st.dataframe(df.head(10))
        
        # Process alerts
        if st.button("🚀 Analyze Alerts", type="primary"):
            with st.spinner("Processing alerts..."):
                try:
                    # Choose summarizer based on user selection
                    if use_full_model:
                        summarizer = AlertSummarizer(n_clusters=n_clusters, use_cache=True)
                        st.info("🧠 Using advanced AI model (BART) for high-quality summaries...")
                    else:
                        summarizer = AlertSummarizerLite(n_clusters=n_clusters)
                        st.info("🚀 Using fast TF-IDF model for quick results...")
                    
                    summaries_df = summarizer.run_from_dataframe(df)
                    
                    # Store results in session state
                    st.session_state.summaries_df = summaries_df
                    st.session_state.original_df = df
                    st.session_state.cluster_assignments = getattr(summaries_df, '_cluster_assignments', None)
                    st.session_state.n_clusters = n_clusters
                    st.session_state.model_type = "Advanced AI" if use_full_model else "Fast TF-IDF"
                    
                    st.success("✅ Analysis complete!")
                    
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
                    st.exception(e)
                    st.exception(e)

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
    
    # Display model info
    model_type = st.session_state.get('model_type', 'Unknown')
    st.subheader(f"🧠 Alert Cluster Summaries (Using {model_type} Model)")
    
    summaries_df = st.session_state.summaries_df
    original_df = st.session_state.original_df
    cluster_assignments = st.session_state.get('cluster_assignments', None)
    
    # Display results with proper cluster assignments
    for idx, row in summaries_df.iterrows():
        cluster_id = row['cluster']
        
        # Count alerts in this cluster
        if cluster_assignments is not None:
            cluster_alerts = cluster_assignments[cluster_assignments['cluster'] == cluster_id]
            alert_count = len(cluster_alerts)
        else:
            # Fallback calculation
            alerts_per_cluster = len(original_df) // len(summaries_df)
            alert_count = alerts_per_cluster + (1 if idx < len(original_df) % len(summaries_df) else 0)
        
        with st.expander(f"📝 Cluster {cluster_id} ({alert_count} alerts)", expanded=True):
            st.write(f"**Summary:** {row['summary']}")
            
            # Show sample alerts from this cluster
            if st.checkbox(f"Show sample alerts from Cluster {cluster_id}", key=f"show_alerts_{idx}"):
                if cluster_assignments is not None:
                    # Show actual alerts from this cluster
                    cluster_alerts = cluster_assignments[cluster_assignments['cluster'] == cluster_id]
                    sample_alerts = cluster_alerts['message'].head(3).tolist()
                else:
                    # Fallback: get sample alerts based on index
                    start_idx = idx * (len(original_df) // len(summaries_df))
                    end_idx = min(start_idx + 3, len(original_df))
                    sample_alerts = original_df['message'].iloc[start_idx:end_idx].tolist()
                
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