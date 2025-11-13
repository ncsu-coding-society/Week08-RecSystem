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
    start_year = 2000
    movies = []

    # increasing page range will give you more movies
    for page in range(1, 25):
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
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# embed dataset
def get_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.squeeze().numpy()

movies_df["embedding"] = movies_df["content"].apply(get_embedding)
print(movies_df["embedding"])
# fetch recommendations

def rec_movie_from_dataset(title, num=5):
    idx = movies_df[movies_df["title"]==title].index[0]
    query_vec = movies_df.loc[idx, "embedding"].reshape(1,-1)
    all_vecs = np.stack(movies_df["embedding"].to_numpy())

    # compute cos similarity
    similarities = cosine_similarity(query_vec, all_vecs).flatten()
    movies_df["similarity"] = similarities

    # sort by similarity
    top = movies_df.sort_values("similarity", ascending=False).iloc[1:num+1]

    print(f"\nMovies similar to '{title}':\n")
    for _,row in top.iterrows():
        print(f"- {row['title']} (similarity: {row['similarity']}) \n{row['description']}\n")


def rec_movie_from_input(movie, num=5):

    query_vec = get_embedding(movie).reshape(1,-1)
    all_vecs = np.stack(movies_df["embedding"].to_numpy())

    # compute cos similarity
    similarities = cosine_similarity(query_vec, all_vecs).flatten()
    movies_df["similarity"] = similarities

    # sort by similarity
    top = movies_df.sort_values("similarity", ascending=False).iloc[1:num+1]

    print(f"\nMovies you might like':\n")
    for _,row in top.iterrows():
        print(f"- {row['title']} (similarity: {row['similarity']}) \n{row['description']}\n")



# testings / interface
# for i in range(10):
#     movie = input("Enter movie description:")
#     rec_movie_from_input(movie)

for i in range(10):
    movie = input("Enter movie name:")
    rec_movie_from_dataset(movie)