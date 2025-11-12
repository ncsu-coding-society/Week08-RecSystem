import requests
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# import environment variables
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

# get data from movie api
def get_recent_movies():
    current_year = 2025
    start_year = 2015
    movies = []

    for page in range(1, 11):
        url = (
            f"{BASE_URL}/discover/movie"
            f"?api_key={API_KEY}"
            f"&language=en-US"
            f"&sort_by=popularity.desc"
            f"&include_adult=false"
            f"&include_video=false"
            f"&primary_release_date.gte={start_year}-01-01"
            f"&primary_release_date.lte={current_year}-12-31"
            f"&page={page}"
        )

        r = requests.get(url)

        data = r.json()
        for m in data.get("results", []):
            movies.append({
                "id": m["id"],
                "title": m["title"],
                "description": m.get("overview", "")
            })
    return pd.DataFrame(movies)

def get_movie_genres(movie_id):
    url = f"{BASE_URL}/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    r = requests.get(url)
    if r.status_code != 200:
        return ""
    data = r.json()
    return ", ".join([g["name"] for g in data.get("genres", [])])

# build dataset
movies_df = get_recent_movies()
movies_df["genres"] = movies_df["id"].apply(get_movie_genres)
movies_df["content"] = (
    movies_df["description"]+ " Genres: " + movies_df["genres"]
)

# load model and tokenizer

# embed dataset

# fetch recommendations

# testings / interface