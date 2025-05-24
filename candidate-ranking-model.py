import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from fastapi import FastAPI
import uvicorn
from pyngrok import ngrok
import nest_asyncio
import threading
import time

# Load CSV files
employer_df = pd.read_csv("Employers.csv")
candidates_df = pd.read_csv("Candidates.csv")

# Convert employer data to dictionary
employer = employer_df.to_dict(orient="records")[0]

# Job dictionary with valid keys
job = {
    "description": employer.get("Job Title", ""),
    "requirements": employer.get("Required Skills", ""),
    "location": employer.get("Location", ""),
    "salary": int(str(employer.get("Salary", "0")).replace("₹", "").replace(",", "").split("-")[0]),
    "required_education": employer.get("Education Requirement", "")
}

# Rename candidate columns if necessary
candidates_df = candidates_df.rename(columns={"Technical Skills": "skills", "Experience": "experience"})

# Convert candidates to dictionary list
candidates = candidates_df.to_dict(orient="records")

# Ensure every candidate has 'skills' and 'experience'
for candidate in candidates:
    candidate['skills'] = candidate.get("skills", "")
    candidate['experience'] = candidate.get("experience", "")

class CandidateRankingSystem:
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=1)
        self.scaler = MinMaxScaler()

    def preprocess_data(self, job_desc, requirements, candidate_profiles):
        """Preprocess job and candidate data"""
        job_text = job_desc + " " + requirements
        all_texts = [job_text] + [prof.get('skills', "") + " " + prof.get('experience', "") for prof in candidate_profiles]

        # Generate TF-IDF matrix
        tfidf_matrix = self.tfidf.fit_transform(all_texts)

        # Compute similarity
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        return similarities.flatten()

    def rank_candidates(self, job, candidates):
        """Rank candidates for a given job"""
        text_scores = self.preprocess_data(job['description'], job['requirements'], candidates)

        # Normalize scores
        if len(text_scores) > 0:
            text_scores = self.scaler.fit_transform(text_scores.reshape(-1,1)).flatten()

        scores = []
        for i, candidate in enumerate(candidates):
            final_score = 0.4 * text_scores[i]

            scores.append({
                'candidate_id': candidate.get('id', "Unknown"),
                'name': candidate.get('name', "Unknown"),
                'score': final_score
            })

        ranked_candidates = sorted(scores, key=lambda x: x['score'], reverse=True)
        return ranked_candidates

# Initialize FastAPI
app = FastAPI()

# Define a sample endpoint
@app.get("/")
def read_root():
    return {"message": "FastAPI running!"}

# Endpoint to rank candidates
@app.get("/rank_candidates")
def rank_candidates():
    ranking_system = CandidateRankingSystem()
    ranked_candidates = ranking_system.rank_candidates(job, candidates)
    ranked_df = pd.DataFrame(ranked_candidates)
    ranked_df.to_csv("Ranked_Candidates.csv", index=False)
    return {"message": "Ranking complete! Check Ranked_Candidates.csv"}

# Allow asyncio to work in Colab
nest_asyncio.apply()

# Set your Ngrok authtoken (replace with your actual token)
ngrok.set_auth_token("2tL6xRFkC2XiobjTqasHSr0ehnc_87d6A5YrhiTkvpLYbpoZY")

# Function to start the FastAPI server
def start_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Start the server in a separate thread
server_thread = threading.Thread(target=start_server)
server_thread.start()

# Wait for the server to start
time.sleep(2)

# Expose the server to the public using ngrok
try:
    ngrok_tunnel = ngrok.connect(8000)  # Use the same port as Uvicorn
    print(f"Public URL: {ngrok_tunnel.public_url}")
except Exception as e:
    print(f"Error creating Ngrok tunnel: {e}")