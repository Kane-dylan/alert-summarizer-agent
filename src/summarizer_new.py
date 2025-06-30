import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Global cache for models to avoid reloading
_model_cache = {}

class AlertSummarizer:
    def __init__(self, n_clusters=3, use_cache=True):
        self.n_clusters = n_clusters
        self.use_cache = use_cache
        self._embedding_model = None
        self._summarizer = None
        self._use_ai_summarization = True
        
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
                try:
                    self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                    if self.use_cache:
                        _model_cache[cache_key] = self._embedding_model
                    print("[INFO] ✅ Embedding model loaded successfully")
                except Exception as e:
                    print(f"[ERROR] Failed to load embedding model: {e}")
                    raise
        return self._embedding_model

    @property
    def summarizer(self):
        if self._summarizer is None and self._use_ai_summarization:
            # Check if model is in global cache first
            cache_key = "summarizer_model"
            if cache_key in _model_cache:
                print("[INFO] Loading summarization model from cache...")
                self._summarizer = _model_cache[cache_key]
            else:
                print("[INFO] Loading AI summarization model...")
                try:
                    # Import here to avoid blocking the module import
                    from transformers import pipeline
                    
                    # Use a smaller, more reliable model
                    self._summarizer = pipeline(
                        "summarization", 
                        model="sshleifer/distilbart-cnn-6-6",
                        device=-1,  # Use CPU
                        framework="pt"  # Use PyTorch
                    )
                    
                    if self.use_cache:
                        _model_cache[cache_key] = self._summarizer
                    print("[INFO] ✅ AI summarization model loaded successfully")
                    
                except Exception as e:
                    print(f"[WARNING] Failed to load AI summarization model: {e}")
                    print("[INFO] Falling back to extractive summarization...")
                    self._use_ai_summarization = False
                    self._summarizer = None
        
        return self._summarizer

    def _extractive_summary(self, text, max_sentences=2):
        """Fallback extractive summarization"""
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return "No summary available"
        
        # Return first few sentences
        summary_sentences = sentences[:max_sentences]
        return '. '.join(summary_sentences) + '.'

    def load_alerts(self, csv_path):
        df = pd.read_csv(csv_path)
        if "message" not in df.columns:
            raise ValueError("CSV must have a 'message' column.")
        return df

    def embed_alerts(self, messages):
        return self.embedding_model.encode(messages)

    def cluster_alerts(self, embeddings):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0, n_init=10)
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
                # Limit text length to avoid memory issues
                if len(text) > 1000:
                    text = text[:1000] + "..."
                
                if self.summarizer is not None:
                    # Use AI summarization
                    summary = self.summarizer(text, max_length=80, min_length=20, do_sample=False)
                    summaries.append(summary[0]['summary_text'])
                else:
                    # Use extractive summarization
                    summary = self._extractive_summary(text)
                    summaries.append(summary)
                    
            except Exception as e:
                print(f"[WARNING] Summarization failed for text: {e}")
                # Fallback to extractive summary
                summary = self._extractive_summary(text)
                summaries.append(summary)
        
        grouped_messages['summary'] = summaries
        return grouped_messages[['cluster', 'summary']]

    def run(self, csv_path):
        print("[INFO] Loading alerts...")
        df = self.load_alerts(csv_path)
        return self.run_from_dataframe(df)

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
