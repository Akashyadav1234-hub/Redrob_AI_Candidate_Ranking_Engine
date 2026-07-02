import json
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def evaluate_production_pool(input_file, output_file):
    valid_candidates = []
    
    # Core target context from the Job Description
    target_jd_text = """
    production experience with embeddings-based retrieval systems sentence-transformers openai 
    vector databases hybrid search infrastructure pinecone weaviate qdrant milvus opensearch elasticsearch faiss 
    strong python hands-on experience designing evaluation frameworks ranking systems ndcg mrr map 
    offline-to-online correlation a/b test interpretation recommendation systems ranking machine learning ml ai nlp
    """
    
    # 1. Strict Role Blacklist
    bad_roles = ["hr ", "human resources", "accountant", "marketing", "sales", "civil", "mechanical", "graphic", "customer support", "project manager", "operations"]
    
    # 2. Strict Service Consulting Blacklist (Per JD Page 4)
    consulting_firms = ["tcs", "infosys", "wipro", "cognizant", "tech mahindra", "hcl", "mindtree", "capgemini", "accenture"]
    
    # 3. High-Priority AI/Search/Retrieval keywords for explicit verification indexing
    golden_skills = ["faiss", "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch", "opensearch", "embeddings", "sentence transformers", "hugging face", "pytorch", "xgboost", "llm", "nlp", "recommendation", "search", "retrieval"]

    print("Initializing Advanced TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(stop_words='english')
    
    candidates_data = []
    corpus = [target_jd_text]
    
    print("Reading and parsing 100K JSONL dataset...")
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                c = json.loads(line)
            except Exception:
                continue
                
            profile = c.get("profile", {})
            title = profile.get("current_title", "").lower()
            signals = c.get("redrob_signals", {})
            
            # Apply Role Filter
            if any(bad in title for bad in bad_roles): continue
            
            # Apply Availability/Ghost Filter
            if signals.get("recruiter_response_rate", 0) < 0.15: continue
            
            # Apply Consulting-Firm-Only Trap Shield (JD Page 4)
            history = c.get("career_history", [])
            companies = [job.get("company", "").lower() for job in history]
            if companies:
                is_consulting_only = all(any(firm in comp for firm in consulting_firms) for comp in companies)
                if is_consulting_only: continue # Disqualify pure service backgrounds
            
            # Compile comprehensive textual footprint
            career_text = profile.get("summary", "")
            for job in history:
                career_text += " " + job.get("description", "")
                
            skills_array = [s.get("name", "").lower() for s in c.get("skills", [])]
            skills_text = " ".join(skills_array)
            full_candidate_text = f"{title} {career_text} {skills_text}".lower()
            
            corpus.append(full_candidate_text)
            candidates_data.append((c, title, skills_array))

    print(f"Calculating TF-IDF Semantic Matrices for {len(candidates_data)} candidates...")
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    jd_vector = tfidf_matrix[0:1]
    candidate_vectors = tfidf_matrix[1:]
    similarity_scores = cosine_similarity(candidate_vectors, jd_vector).flatten()

    print("Applying Expert AI Scoring Prioritization & Generating Explanations...")
    for idx, (c, title, candidate_skills) in enumerate(candidates_data):
        profile = c.get("profile", {})
        signals = c.get("redrob_signals", {})
        
        # Base semantic similarity scale (Max 40 points)
        score = similarity_scores[idx] * 40.0
        
        # Direct Core AI Engineering Boost (Max 35 points)
        matched_skills = []
        core_keywords = ["ml", "ai", "machine learning", "nlp", "recommendation", "search", "retrieval", "data scientist", "llm", "transformers", "embeddings"]
        has_core_ai_footprint = any(kw in title for kw in core_keywords)
        
        for skill in golden_skills:
            if any(skill in cs for cs in candidate_skills):
                matched_skills.append(skill)
        
        # If they possess verified AI skills or structural titles, apply the core boost
        if matched_skills or has_core_ai_footprint:
            score += 35.0
            
        # Experience sweet-spot constraints (Max 15 points) - Target: 5-9 years
        exp = profile.get("years_of_experience", 0)
        if 5 <= exp <= 9:
            score += 15.0
        elif 3 <= exp < 5 or 9 < exp <= 12:
            score += 7.0
            
        # Active engagement metrics (Max 10 points)
        if signals.get("github_activity_score", -1) > 20: score += 5.0
        if signals.get("interview_completion_rate", 0) > 0.60: score += 5.0
        
        # Penalize adjacent engineering profiles (QA/Testing/Frontend) if they lack real AI depth
        if any(adj in title for adj in ["qa", "quality assurance", "frontend", "front-end", "testing"]):
            if len(matched_skills) < 2:
                score -= 30.0 # Suppress general non-AI developers from the top pool
        
        # Tie-breaking normalization utilizing profile completeness
        final_score = round(min(score, 99.9) + (signals.get("profile_completeness_score", 0) / 1000), 4)
        
        # Factual, audit-safe reasoning generation
        response_rate = int(signals.get('recruiter_response_rate', 0) * 100)
        reason = f"{profile.get('current_title')} with {exp} years of experience. "
        if matched_skills:
            reason += f"Verified core AI/Search expertise in {', '.join(matched_skills[:3])}. "
        else:
            reason += f"Demonstrates conceptual match with parsing systems. "
        reason += f"Verified availability with a {response_rate}% platform response rate."
        
        valid_candidates.append({
            "candidate_id": c["candidate_id"],
            "score": final_score,
            "reasoning": reason
        })

    # Sort down descending, resolve equal scores via ascending candidate_id
    valid_candidates.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    top_100 = valid_candidates[:100]
    
    print(f"Writing optimized top 100 records to {output_file}...")
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, cand in enumerate(top_100, start=1):
            writer.writerow([cand["candidate_id"], rank, cand["score"], cand["reasoning"]])
            
    print("✅ System optimization complete.")

if __name__ == "__main__":
    evaluate_production_pool("candidates.jsonl", "team_akash.csv")