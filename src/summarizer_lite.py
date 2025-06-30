# src/summarizer_lite.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import os

class AlertSummarizerLite:
    """Lightweight alert summarizer that works without heavy ML models"""
    
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')

    def load_alerts(self, csv_path):
        df = pd.read_csv(csv_path)
        if "message" not in df.columns:
            raise ValueError("CSV must have a 'message' column.")
        return df

    def embed_alerts(self, messages):
        """Create TF-IDF embeddings for messages"""
        return self.vectorizer.fit_transform(messages).toarray()

    def cluster_alerts(self, embeddings):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        return labels

    def group_by_cluster(self, df, labels):
        df['cluster'] = labels
        grouped = df.groupby('cluster')['message'].apply(lambda x: ' '.join(x)).reset_index()
        return grouped

    def summarize_grouped_alerts(self, grouped_messages):
        """Create simple extractive summaries"""
        summaries = []
        for text in grouped_messages['message']:
            # Simple extractive summary - take first sentence and most common words
            sentences = text.split('.')
            # Take the first sentence as summary
            if sentences:
                summary = sentences[0].strip()
                if len(summary) < 10:  # If too short, take first two sentences
                    summary = '. '.join(sentences[:2]).strip()
                summaries.append(summary)
            else:
                summaries.append("No summary available")
        
        grouped_messages['summary'] = summaries
        return grouped_messages[['cluster', 'summary']]

    def run_from_dataframe(self, df):
        """Run the summarization process directly from a DataFrame"""
        if "message" not in df.columns:
            raise ValueError("DataFrame must have a 'message' column.")
        
        print("[INFO] Creating TF-IDF embeddings...")
        embeddings = self.embed_alerts(df['message'].tolist())

        print("[INFO] Clustering alerts...")
        labels = self.cluster_alerts(embeddings)

        print("[INFO] Grouping and summarizing...")
        grouped = self.group_by_cluster(df, labels)
        summarized = self.summarize_grouped_alerts(grouped)
        
        # Store cluster assignments in a proper way
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = labels
        
        # Use setattr to avoid pandas warning
        setattr(summarized, '_cluster_assignments', df_with_clusters)

        return summarized

    def run(self, csv_path):
        print("[INFO] Loading alerts...")
        df = self.load_alerts(csv_path)
        return self.run_from_dataframe(df)
