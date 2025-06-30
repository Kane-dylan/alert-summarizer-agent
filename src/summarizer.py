# src/summarizer.py

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers.pipelines import pipeline
import numpy as np

class AlertSummarizer:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        try:
            print("[INFO] Loading embedding model...")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("[INFO] Loading summarization model...")
            self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            print("[INFO] Models loaded successfully!")
        except Exception as e:
            print(f"[ERROR] Failed to load models: {str(e)}")
            # Fallback to a simpler model or disable features
            self.embedding_model = None
            self.summarizer = None

    def load_alerts(self, csv_path):
        df = pd.read_csv(csv_path)
        if "message" not in df.columns:
            raise ValueError("CSV must have a 'message' column.")
        return df

    def embed_alerts(self, messages):
        if self.embedding_model is None:
            # Simple fallback - create random embeddings for demo
            import random
            return [[random.random() for _ in range(384)] for _ in messages]
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
                if self.summarizer is None:
                    # Simple fallback - just truncate the text
                    if len(text) > 100:
                        summary = text[:100] + "..."
                    else:
                        summary = text
                else:
                    summary = self.summarizer(text, max_length=80, min_length=20, do_sample=False)
                    summary = summary[0]['summary_text']
                summaries.append(summary)
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
        """Run the summarizer on a pandas DataFrame directly"""
        print("[INFO] Processing DataFrame...")
        
        if "message" not in df.columns:
            raise ValueError("DataFrame must have a 'message' column.")

        print("[INFO] Embedding messages...")
        embeddings = self.embed_alerts(df['message'].tolist())

        print("[INFO] Clustering alerts...")
        labels = self.cluster_alerts(embeddings)

        print("[INFO] Grouping and summarizing...")
        grouped = self.group_by_cluster(df, labels)
        summarized = self.summarize_grouped_alerts(grouped)

        return summarized
