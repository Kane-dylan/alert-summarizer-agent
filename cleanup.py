#!/usr/bin/env python3
"""
Cleanup script for Alert Summarizer Agent
Removes unnecessary files and optimizes the codebase
"""

import os
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.absolute()

def clean_pycache():
    """Remove all __pycache__ directories"""
    print("🧹 Cleaning __pycache__ directories...")
    count = 0
    for pycache_dir in ROOT_DIR.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache_dir)
            count += 1
            print(f"  Removed: {pycache_dir}")
        except Exception as e:
            print(f"  Failed to remove {pycache_dir}: {e}")
    print(f"✅ Removed {count} __pycache__ directories")

def clean_backup_files():
    """Remove backup files and temporary files"""
    print("🧹 Cleaning backup and temporary files...")
    patterns = ["*_backup.py", "*_old.py", "*_fixed.py", "test_advanced_model.py"]
    count = 0
    
    for pattern in patterns:
        for file in ROOT_DIR.rglob(pattern):
            try:
                file.unlink()
                count += 1
                print(f"  Removed: {file}")
            except Exception as e:
                print(f"  Failed to remove {file}: {e}")
    print(f"✅ Removed {count} backup/temporary files")

def clean_model_cache():
    """Clean model cache directory"""
    print("🧹 Cleaning model cache...")
    cache_dir = ROOT_DIR / ".model_cache"
    if cache_dir.exists():
        try:
            shutil.rmtree(cache_dir)
            print(f"  Removed: {cache_dir}")
        except Exception as e:
            print(f"  Failed to remove {cache_dir}: {e}")
    print("✅ Model cache cleaned")

def clean_logs():
    """Clean log files"""
    print("🧹 Cleaning log files...")
    log_patterns = ["*.log", "*.out"]
    count = 0
    
    for pattern in log_patterns:
        for file in ROOT_DIR.rglob(pattern):
            try:
                file.unlink()
                count += 1
                print(f"  Removed: {file}")
            except Exception as e:
                print(f"  Failed to remove {file}: {e}")
    print(f"✅ Removed {count} log files")

def optimize_requirements():
    """Create optimized requirements.txt"""
    print("📦 Optimizing requirements.txt...")
    
    # Core requirements only
    requirements = [
        "pandas>=2.0.0",
        "numpy>=1.21.0", 
        "scikit-learn>=1.3.0",
        "streamlit>=1.45.0",
        "sentence-transformers>=2.2.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "torch>=2.0.0"
    ]
    
    requirements_file = ROOT_DIR / "requirements.txt"
    try:
        with open(requirements_file, 'w') as f:
            for req in requirements:
                f.write(req + '\n')
        print(f"✅ Updated {requirements_file}")
    except Exception as e:
        print(f"❌ Failed to update requirements: {e}")

def show_project_size():
    """Show project size information"""
    print("📊 Project size analysis...")
    
    total_size = 0
    file_count = 0
    
    for file in ROOT_DIR.rglob("*"):
        if file.is_file():
            size = file.stat().st_size
            total_size += size
            file_count += 1
    
    # Convert to MB
    total_mb = total_size / (1024 * 1024)
    
    print(f"  Total files: {file_count}")
    print(f"  Total size: {total_mb:.2f} MB")
    
    # Show largest directories
    dir_sizes = {}
    for item in ROOT_DIR.iterdir():
        if item.is_dir() and item.name not in ['.git', 'venv']:
            size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
            dir_sizes[item.name] = size / (1024 * 1024)
    
    print("  Directory sizes:")
    for name, size in sorted(dir_sizes.items(), key=lambda x: x[1], reverse=True):
        print(f"    {name}: {size:.2f} MB")

def create_simple_summarizer():
    """Create a simplified, robust summarizer"""
    print("🔧 Creating simplified summarizer...")
    
    summarizer_content = '''import pandas as pd
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
                    print("[INFO] ✅ Embedding model loaded")
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
                    print("[INFO] ✅ AI summarizer loaded")
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
        """Simple extractive summarization"""
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
                # Limit text length
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
        
        # Store cluster assignments
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = labels
        setattr(summarized, '_cluster_assignments', df_with_clusters)

        return summarized
'''
    
    summarizer_file = ROOT_DIR / "src" / "summarizer.py"
    try:
        with open(summarizer_file, 'w') as f:
            f.write(summarizer_content)
        print(f"✅ Created simplified {summarizer_file}")
    except Exception as e:
        print(f"❌ Failed to create summarizer: {e}")

def main():
    """Main cleanup function"""
    print("🧹 Alert Summarizer Agent - Cleanup & Optimization")
    print("=" * 55)
    
    # Show current state
    show_project_size()
    print()
    
    # Cleanup operations
    clean_pycache()
    clean_backup_files()
    clean_model_cache()
    clean_logs()
    optimize_requirements()
    create_simple_summarizer()
    
    print()
    print("🎉 Cleanup completed!")
    print()
    
    # Show final state
    show_project_size()
    
    print()
    print("📋 Next steps:")
    print("1. Run: python deploy.py setup")
    print("2. Test: python deploy.py test")
    print("3. Start: python deploy.py run")

if __name__ == "__main__":
    main()
