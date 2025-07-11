import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Global cache for models
_model_cache = {}

class AlertSummarizer:
    def __init__(self, n_clusters=3, use_cache=True):
        self.n_clusters = n_clusters
        self.use_cache = use_cache
        self._embedding_model = None
        self._summarizer = None
        self._use_ai = False  # Start with AI disabled for stability

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            cache_key = "embedding_model"
            if cache_key in _model_cache:
                print("[INFO] Loading embedding model from cache...")
                self._embedding_model = _model_cache[cache_key]
            else:
                print("[INFO] Loading embedding model...")
                try:
                    self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                    if self.use_cache:
                        _model_cache[cache_key] = self._embedding_model
                    print("[INFO] Embedding model loaded successfully")
                except Exception as e:
                    print(f"[ERROR] Failed to load embedding model: {e}")
                    raise
        return self._embedding_model

    def enable_ai_summarization(self):
        """Enable AI summarization (call this after initialization)"""
        if self._summarizer is None:
            cache_key = "summarizer_model"
            if cache_key in _model_cache:
                print("[INFO] Loading AI summarizer from cache...")
                self._summarizer = _model_cache[cache_key]
                self._use_ai = True
            else:
                print("[INFO] Loading AI summarization model...")
                try:
                    from transformers import pipeline
                    import torch
                    
                    # Set environment for stability
                    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
                    
                    self._summarizer = pipeline(
                        "summarization",
                        model="sshleifer/distilbart-cnn-6-6",
                        device=-1,
                        torch_dtype=torch.float32
                    )
                    
                    if self.use_cache:
                        _model_cache[cache_key] = self._summarizer
                    
                    self._use_ai = True
                    print("[INFO] AI summarizer loaded successfully")
                    return True
                    
                except Exception as e:
                    print(f"[WARNING] AI summarizer failed: {e}")
                    print("[INFO] Will use extractive summarization")
                    self._use_ai = False
                    return False
        else:
            self._use_ai = True
            return True

    def _extractive_summary(self, text, max_sentences=2):
        """Simple extractive summarization fallback"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            return "No summary available"
        
        # Take first few sentences
        summary = '. '.join(sentences[:max_sentences])
        if not summary.endswith('.'):
            summary += '.'
        return summary

    def load_alerts(self, csv_path):
        df = pd.read_csv(csv_path)
        if "message" not in df.columns:
            raise ValueError("CSV must have a 'message' column.")
        return df

    def embed_alerts(self, messages):
        return self.embedding_model.encode(messages)

    def cluster_alerts(self, embeddings):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0, n_init=10)
        return kmeans.fit_predict(embeddings)

    def group_by_cluster(self, df, labels):
        df['cluster'] = labels
        grouped = df.groupby('cluster')['message'].apply(lambda x: ' '.join(x)).reset_index()
        return grouped

    def summarize_grouped_alerts(self, grouped_messages):
        summaries = []
        for text in grouped_messages['message']:
            try:
                # Limit text length to avoid memory issues
                if len(text) > 1000:
                    text = text[:1000] + "..."
                
                if self._use_ai and self._summarizer is not None:
                    # Use AI summarization
                    result = self._summarizer(text, max_length=80, min_length=20, do_sample=False)
                    summaries.append(result[0]['summary_text'])
                else:
                    # Use extractive summarization
                    summaries.append(self._extractive_summary(text))
                    
            except Exception as e:
                print(f"[WARNING] Summarization failed: {e}")
                summaries.append(self._extractive_summary(text))
        
        grouped_messages['summary'] = summaries
        return grouped_messages[['cluster', 'summary']]

    def run(self, csv_path):
        print("[INFO] Loading alerts...")
        df = self.load_alerts(csv_path)
        return self.run_from_dataframe(df)

    def run_from_dataframe(self, df):
        if "message" not in df.columns:
            raise ValueError("DataFrame must have a 'message' column.")
        
        print("[INFO] Embedding messages...")
        embeddings = self.embed_alerts(df['message'].tolist())

        print("[INFO] Clustering alerts...")
        labels = self.cluster_alerts(embeddings)

        print("[INFO] Grouping and summarizing...")
        grouped = self.group_by_cluster(df, labels)
        summarized = self.summarize_grouped_alerts(grouped)
        
        # Store cluster assignments for proper display
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = labels
        setattr(summarized, '_cluster_assignments', df_with_clusters)

        return summarized
