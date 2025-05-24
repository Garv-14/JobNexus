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
candidates_df = candidates_df.rename(columns={
    "Technical Skills": "skills",
    "Experience": "experience",
    "Expected Salary": "expected_salary",
    "Education": "education",
    "Willing to Relocate": "willing_to_relocate"
})

# Convert candidates to dictionary list
candidates = candidates_df.to_dict(orient="records")

# Ensure every candidate has required fields
for candidate in candidates:
    candidate['skills'] = candidate.get("skills", "")
    candidate['experience'] = candidate.get("experience", "")
    candidate['expected_salary'] = candidate.get("expected_salary", None)
    candidate['education'] = candidate.get("education", "")
    candidate['willing_to_relocate'] = candidate.get("willing_to_relocate", False)

class CandidateRankingSystem:
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.scaler = MinMaxScaler()
        
    def preprocess_data(self, job_desc, requirements, candidate_profiles):
        """Preprocess job and candidate data"""
        # Convert text fields to TF-IDF vectors
        job_text = job_desc + " " + requirements
        all_texts = [job_text] + [prof['skills'] + " " + prof['experience'] for prof in candidate_profiles]
        tfidf_matrix = self.tfidf.fit_transform(all_texts)
        
        # Calculate text similarity scores
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        
        return similarities.flatten()
    
    def calculate_location_score(self, job_location, candidate_location, willing_to_relocate):
        """Calculate location match score"""
        if job_location.lower() == candidate_location.lower():
            return 1.0
        elif willing_to_relocate:
            return 0.7
        return 0.3
    
    def calculate_salary_score(self, job_salary, candidate_salary):
        """Calculate salary match score"""
        if not candidate_salary:  # Handle missing salary data
            return 0.5
        
        salary_diff = abs(job_salary - candidate_salary)
        return 1 - (salary_diff / job_salary) if salary_diff < job_salary else 0
    
    def calculate_education_score(self, required_education, candidate_education):
        """Calculate education match score"""
        education_levels = {
            'high school': 1,
            'bachelor': 2,
            'master': 3,
            'phd': 4
        }
        
        req_level = education_levels.get(required_education.lower(), 0)
        cand_level = education_levels.get(candidate_education.lower(), 0)
        
        if cand_level >= req_level:
            return 1.0
        return 0.5
    
    def rank_candidates(self, job, candidates):
        """Rank candidates for a given job"""
        # Calculate text similarity scores
        text_scores = self.preprocess_data(
            job['description'],
            job['requirements'],
            candidates
        )
        
        # Calculate other score components for each candidate
        scores = []
        for i, candidate in enumerate(candidates):
            location_score = self.calculate_location_score(
                job['location'],
                candidate.get('location', ''),
                candidate.get('willing_to_relocate', False)
            )
            
            salary_score = self.calculate_salary_score(
                job['salary'],
                candidate.get('expected_salary', None)
            )
            
            education_score = self.calculate_education_score(
                job['required_education'],
                candidate.get('education', '')
            )
            
            # Combine scores with weights
            final_score = (
                0.4 * text_scores[i] +  # Skills and experience match
                0.2 * location_score +   # Location match
                0.2 * salary_score +     # Salary match
                0.2 * education_score    # Education match
            )
            
            scores.append({
                'candidate_id': candidate['id'],
                'name': candidate['name'],
                'score': final_score,
                'text_match': text_scores[i],
                'location_match': location_score,
                'salary_match': salary_score,
                'education_match': education_score
            })
        
        # Sort candidates by final score
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
    return {"ranked_candidates": ranked_candidates}

# Allow asyncio to work in Colab
nest_asyncio.apply()

# Set your Ngrok authtoken (replace with your actual token)
ngrok.set_auth_token("2tL6xRFkC2XiobjTqasHSr0ehnc_87d6A5YrhiTkvpLYbpoZY")

# Function to start the FastAPI server
def start_server():
    uvicorn.run(app, host="0.0.0.0", port=8001)

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