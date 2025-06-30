import json
import os
from datetime import datetime

FEEDBACK_FILE = "data/feedback_log.json"

def load_feedback():
    # Ensure data directory exists
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    
    if not os.path.exists(FEEDBACK_FILE):
        return []
    with open(FEEDBACK_FILE, "r") as f:
        return json.load(f)

def save_feedback(cluster_id, summary_text, rating, comment=None):
    feedback_entry = {
        "timestamp": datetime.now().isoformat(),
        "cluster_id": cluster_id,
        "summary": summary_text,
        "rating": rating,  # "Helpful" or "Not Helpful"
        "comment": comment
    }
    all_feedback = load_feedback()
    all_feedback.append(feedback_entry)

    with open(FEEDBACK_FILE, "w") as f:
        json.dump(all_feedback, f, indent=2)

    print(f"[INFO] Feedback saved for Cluster {cluster_id}")

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
