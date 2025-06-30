# ⚡ Alert Summarizer Agent

An AI-powered agent that automatically clusters, summarizes, and prioritizes operational asset alerts — helping engineering and operations teams focus on what matters most.

Built as part of an application for the **AI Systems Internship at Xempla**, this project demonstrates how large language models (LLMs) and embeddings can be used in real-world operations & maintenance (O&M) contexts to reduce alert fatigue, surface high-impact issues, and learn from operator feedback.

---

## 📌 Features

### ✅ Intelligent Alert Summarization

- Clusters similar alerts using sentence embeddings.
- Summarizes each alert cluster into clear, human-readable insights using Hugging Face's `facebook/bart-large-cnn` model.

### ✅ Feedback Loop

- Users can rate summaries as "Helpful" or "Not Helpful" and leave optional comments.
- Feedback is stored and analyzed to monitor and improve summarization quality over time.

### ✅ Lightweight Streamlit UI

- Upload alert files, view summaries, and submit feedback — all in an intuitive interface.

---

## 🖼️ Demo

➡️ [Loom Video Link Placeholder]  
_This short video explains the project, use case, and demo of the prototype._

---

## 🧠 Use Case Alignment: Why It Matters

This agent is inspired by real problems in O&M environments:

- **Alert overload**: Engineers are flooded with repetitive or low-impact alerts.
- **Slow triage**: Identifying what matters takes time and judgment.
- **Lack of learning**: Systems don't improve based on user feedback.

**This agent solves all three**, aligning with Xempla's mission of building intelligent, autonomous O&M platforms.

---

## 📂 Project Structure

```
alert-summarizer-agent/
│
├── data/
│   ├── sample_alerts.csv       # Sample input alerts
│   └── feedback_log.json       # Stored user feedback
│
├── notebooks/
│   └── alert_summarizer.ipynb  # Jupyter notebook for testing core logic
│
├── src/
│   ├── summarizer.py           # Main summarization logic
│   └── feedback_loop.py        # Feedback storage and stats
│
├── app/
│   └── streamlit_app.py        # UI to interact with summaries and submit feedback
│
├── .env                        # Hugging Face API token (not committed)
├── requirements.txt            # Python dependencies
└── README.md                   # Project overview
```

---

## 🚀 How to Run

### 1. Clone the repo & set up environment

```bash
git clone https://github.com/yourusername/alert-summarizer-agent.git
cd alert-summarizer-agent
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your Hugging Face API token to .env file

```ini
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

### 3. Run Streamlit app

```bash
streamlit run app/streamlit_app.py
```

## 🧪 Sample Input Format (sample_alerts.csv)

```csv
timestamp,message
2024-06-30 10:30:00,High temperature in AHU-3 zone
2024-06-30 10:31:00,Repeated vibration warning on Pump-2
2024-06-30 10:32:00,Power surge detected in Panel A
```

## 🛠 Tech Stack

| Area             | Tools Used                     |
| ---------------- | ------------------------------ |
| Language Models  | Hugging Face Transformers      |
| Embeddings       | sentence-transformers (MiniLM) |
| Vector Grouping  | KMeans Clustering              |
| Frontend         | Streamlit                      |
| Feedback Storage | JSON                           |

## 📈 What's Next (Ideas for Iteration)

- Use historical resolution data to suggest automated fixes.
- Rank summaries by severity using external KPIs or sensor data.
- Fine-tune a summarization model on facility maintenance logs.
- Deploy as a Slack bot for real-time feedback collection.

## 🙌 About the Creator

This project was created as part of my application to the AI Systems Intern role at Xempla. I'm passionate about the intersection of LLMs, real-time operations, and sustainability, and would be excited to help shape the future of autonomous O&M systems.
