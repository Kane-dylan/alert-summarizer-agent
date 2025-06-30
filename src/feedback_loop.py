# src/feedback_loop.py

import json
import os
from datetime import datetime

# Get the absolute path to the data directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
FEEDBACK_FILE = os.path.join(PROJECT_ROOT, "data", "feedback_log.json")

def load_feedback():
    try:
        if not os.path.exists(FEEDBACK_FILE):
            # Create data directory if it doesn't exist
            os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
            return []
        with open(FEEDBACK_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print("[WARNING] Feedback file corrupted, creating new one")
        return []
    except Exception as e:
        print(f"[ERROR] Failed to load feedback: {e}")
        return []

def save_feedback(cluster_id, summary_text, rating, comment=None):
    try:
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "cluster_id": cluster_id,
            "summary": summary_text,
            "rating": rating,  # "Helpful" or "Not Helpful"
            "comment": comment
        }
        all_feedback = load_feedback()
        all_feedback.append(feedback_entry)

        # Ensure data directory exists
        os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
        
        with open(FEEDBACK_FILE, "w") as f:
            json.dump(all_feedback, f, indent=2)

        print(f"[INFO] Feedback saved for Cluster {cluster_id}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save feedback: {e}")
        return False

def get_feedback_stats():
    feedback = load_feedback()
    review_flags = {}
    for entry in feedback:
        cluster_id = entry["cluster_id"]
        if cluster_id not in review_flags:
            review_flags[cluster_id] = {"not_helpful": 0, "helpful": 0}
        if entry["rating"] == "Not Helpful":
            review_flags[cluster_id]["not_helpful"] += 1
        else:
            review_flags[cluster_id]["helpful"] += 1
    return review_flags
