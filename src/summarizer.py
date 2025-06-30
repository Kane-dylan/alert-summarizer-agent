# src/summarizer.py

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import pipeline
import numpy as np

class AlertSummarizer:
    def __init__(self, n_clusters=3):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        # For testing and demo purposes, use fallback mode to avoid large model downloads
        print("[INFO] Using fallback summarization mode for faster testing")
        self.summarizer = None
        self.use_hf_model = False
        self.n_clusters = n_clusters

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
                if self.use_hf_model and self.summarizer:
                    summary = self.summarizer(text, max_length=80, min_length=20, do_sample=False)
                    summaries.append(summary[0]['summary_text'])
                else:
                    # Improved fallback: Better extractive summary
                    sentences = text.split('. ')
                    if len(sentences) > 0:
                        # Get first meaningful sentence
                        first_sentence = sentences[0].strip()
                        if not first_sentence.endswith('.'):
                            first_sentence += '.'
                        
                        # Extract key terms more intelligently
                        words = text.lower().split()
                        key_words = []
                        technical_terms = []
                        
                        for word in words:
                            clean_word = word.strip('.,!?;:')
                            if len(clean_word) > 4 and clean_word not in ['alert', 'warning', 'error', 'detected', 'system']:
                                if any(char.isdigit() for char in clean_word) or '-' in clean_word:
                                    technical_terms.append(clean_word)
                                else:
                                    key_words.append(clean_word)
                        
                        # Combine technical terms and key words
                        important_terms = list(set(technical_terms[:3] + key_words[:3]))
                        
                        if important_terms:
                            summary = f"{first_sentence} Key components affected: {', '.join(important_terms)}"
                        else:
                            summary = first_sentence
                        
                        summaries.append(summary)
                    else:
                        summaries.append(f"Multiple alerts detected: {text[:80]}...")
            except Exception as e:
                print(f"[WARNING] Error summarizing cluster: {e}")
                summaries.append(f"Alert cluster summary: {text[:60]}...")
                
        grouped_messages['summary'] = summaries
        return grouped_messages[['cluster', 'summary']]

    def run_from_dataframe(self, df):
        """Process alerts from a DataFrame instead of CSV file"""
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
