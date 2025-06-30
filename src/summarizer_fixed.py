import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import pipeline
import numpy as np
import os
import pickle
import hashlib

# Global cache for models to avoid reloading
_model_cache = {}

class AlertSummarizer:
    def __init__(self, n_clusters=3, use_cache=True):
        self.n_clusters = n_clusters
        self.use_cache = use_cache
        self._embedding_model = None
        self._summarizer = None
        
        # Create cache directory
        self.cache_dir = os.path.join(os.path.dirname(__file__), '..', '.model_cache')
        os.makedirs(self.cache_dir, exist_ok=True)

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            # Check if model is in global cache first
            cache_key = "embedding_model"
            if cache_key in _model_cache:
                print("[INFO] Loading embedding model from cache...")
                self._embedding_model = _model_cache[cache_key]
            else:
                print("[INFO] Loading embedding model...")
                self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                if self.use_cache:
                    _model_cache[cache_key] = self._embedding_model
        return self._embedding_model

    @property
    def summarizer(self):
        if self._summarizer is None:
            # Check if model is in global cache first
            cache_key = "summarizer_model"
            if cache_key in _model_cache:
                print("[INFO] Loading summarization model from cache...")
                self._summarizer = _model_cache[cache_key]
            else:
                print("[INFO] Loading summarization model...")
                self._summarizer = pipeline(
                    "summarization", 
                    model="facebook/bart-large-cnn",
                    device=-1  # Use CPU to avoid GPU issues
                )
                if self.use_cache:
                    _model_cache[cache_key] = self._summarizer
        return self._summarizer

    def load_alerts(self, csv_path):
        df = pd.read_csv(csv_path)
        if "message" not in df.columns:
            raise ValueError("CSV must have a 'message' column.")
        return df

    def embed_alerts(self, messages):
        return self.embedding_model.encode(messages)

    def cluster_alerts(self, embeddings):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        labels = kmeans.fit_predict(embeddings)
        return labels

    def group_by_cluster(self, df, labels):
        df['cluster'] = labels
        grouped = df.groupby('cluster')['message'].apply(lambda x: ' '.join(x)).reset_index()
        return grouped

    def summarize_grouped_alerts(self, grouped_messages):
        summaries = []
        for text in grouped_messages['message']:
            try:
                summary = self.summarizer(text, max_length=80, min_length=20, do_sample=False)
                summaries.append(summary[0]['summary_text'])
            except Exception as e:
                summaries.append(f"Error summarizing: {str(e)}")
        grouped_messages['summary'] = summaries
        return grouped_messages[['cluster', 'summary']]

    def run(self, csv_path):
        print("[INFO] Loading alerts...")
        df = self.load_alerts(csv_path)

        print("[INFO] Embedding messages...")
        embeddings = self.embed_alerts(df['message'].tolist())

        print("[INFO] Clustering alerts...")
        labels = self.cluster_alerts(embeddings)

        print("[INFO] Grouping and summarizing...")
        grouped = self.group_by_cluster(df, labels)
        summarized = self.summarize_grouped_alerts(grouped)

        return summarized

    def run_from_dataframe(self, df):
        """Run the summarization process directly from a DataFrame"""
        if "message" not in df.columns:
            raise ValueError("DataFrame must have a 'message' column.")
        
        print("[INFO] Embedding messages...")
        embeddings = self.embed_alerts(df['message'].tolist())

        print("[INFO] Clustering alerts...")
        labels = self.cluster_alerts(embeddings)

        print("[INFO] Grouping and summarizing...")
        grouped = self.group_by_cluster(df, labels)
        summarized = self.summarize_grouped_alerts(grouped)
        
        # Store cluster assignments in the original dataframe for reference
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = labels
        
        # Add cluster assignments as a private attribute to avoid pandas warning
        setattr(summarized, '_cluster_assignments', df_with_clusters)

        return summarized
