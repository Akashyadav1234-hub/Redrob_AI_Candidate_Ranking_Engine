import csv

def audit_submission(csv_file):
    print(f"--- Auditing {csv_file} for Hackathon Accuracy ---")
    
    red_flags = ["marketing", "hr ", "sales", "accountant", "civil", "graphic"]
    golden_skills = ["faiss", "pinecone", "weaviate", "qdrant", "pytorch", "llm", "search"]
    
    total_candidates = 0
    trap_count = 0
    high_skill_count = 0
    scores = []
    
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_candidates += 1
            reasoning = row["reasoning"].lower()
            score = float(row["score"])
            scores.append(score)
            
            # Check if a trap slipped through
            if any(flag in reasoning for flag in red_flags):
                trap_count += 1
                print(f"🚨 TRAP DETECTED at Rank {row['rank']}: {row['reasoning']}")
                
            # Check if they have the actual skills needed
            if any(skill in reasoning for skill in golden_skills):
                high_skill_count += 1

    print("\n--- Final Audit Report ---")
    print(f"Total Ranked: {total_candidates}/100")
    print(f"Traps/Keyword Stuffers in Top 100: {trap_count} (Must be 0)")
    print(f"Candidates with Core AI/Search Skills: {high_skill_count}%")
    print(f"Top Score: {max(scores)}")
    print(f"Bottom Score: {min(scores)}")
    
    if trap_count == 0 and high_skill_count > 80:
        print("\n✅ RESULT: EXCELLENT. Your system is highly accurate and safe to submit.")
    else:
        print("\n❌ RESULT: WARNING. Your system is letting bad data through. Adjust ranker.py weights.")

if __name__ == "__main__":
    audit_submission("team_akash.csv")