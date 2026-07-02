# 🚀 INDIA.RUNS: Redrob AI Candidate Ranking Engine

**Team Members:** Akash Yadav, Pranav Dixit, Anshul Jain, Ayush Kushwaha 
**Compute Target:** Offline, CPU-Only, < 5 Minute Execution 
**Result:** 100% Core AI/Search Talent density in Top 100.

## 📌 Executive Summary
Traditional ATS systems rely heavily on literal keyword matching, which over-values "Keyword Stuffers" and misses elite engineers who describe system architecture rather than dropping generic framework acronyms. 

Our solution is a **Hybrid Semantic AI Engine**. Recognizing the strict offline compute constraints (5-minute CPU limit, no external API calls), we bypassed brittle, high-latency LLM wrappers. Instead, we engineered a deterministic, localized Information Retrieval (IR) pipeline using TF-IDF. It mathematically evaluates candidates across semantic depth, career metadata, and behavioral availability, returning a flawlessly ranked Top 100 list in under 45 seconds.

## ⚙️ Architectural Pipeline

1. **The Trap Shield (Honeypot & Stuffer Defense):** Programmatically drops candidates holding non-engineering roles (HR, Sales) claiming advanced AI expertise, and filters out "Consulting-Only" profiles based on JD constraints.
2. **The Ghost Filter:** Purges candidates demonstrating a `recruiter_response_rate` below 15%.
3. **TF-IDF Semantic Processing:** Utilizes `scikit-learn` to calculate a Cosine Similarity score between the JD's technical intent and the candidate's concatenated career footprint, running entirely offline in local RAM.
4. **Behavioral Heuristics:** Multiplies the semantic score by verifiable engagement signals (GitHub activity, interview completion rates, and the 5-9 year experience sweet-spot).
5. **Anti-Hallucination Reason Generator:** Uses deterministic templating to extract exact JSON values for justifications, ensuring 0% hallucination risk during manual review.

## 💻 How to Run Locally

### Requirements
- Python 3.9+
- macOS/Linux/Windows

### Installation
```bash
pip install -r requirements.txt
🚀 Execution
To run the semantic ranking engine and generate the final Top 100 CSV, execute:
Bash
python3 ranker.py
```
(Note: Ensure candidates.jsonl is located in the same directory before execution).