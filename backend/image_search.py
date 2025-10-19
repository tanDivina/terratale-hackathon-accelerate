import os
import zipfile
import pandas as pd
import requests
from PIL import Image
from elasticsearch import Elasticsearch
from elasticsearch.helpers import parallel_bulk
from transformers import CLIPProcessor, CLIPModel
import torch

# --- Configuration ---
ELASTIC_API_KEY = os.environ.get("ELASTIC_API_KEY")
ELASTIC_ENDPOINT_URL = os.environ.get("ELASTIC_ENDPOINT_URL")
INDEX_NAME = "images"
MODEL_NAME = "openai/clip-vit-base-patch32"

# --- Model Loading ---
# Load the model and processor only once
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

# --- Elasticsearch Client ---
def get_es_client():
    if ELASTIC_ENDPOINT_URL:
        return Elasticsearch(ELASTIC_ENDPOINT_URL, api_key=ELASTIC_API_KEY, request_timeout=600)
    else:
        return None

# --- Indexing ---
def index_images():
    """Downloads the Unsplash dataset, generates embeddings, and indexes it into Elasticsearch."""
    es = get_es_client()
    if not es:
        print("Skipping image indexing because Elastic credentials are not set.")
        return

    try:
        if not es.ping():
            print("Could not connect to Elasticsearch. Please check your credentials.")
            return
    except Exception as e:
        print(f"An error occurred while connecting to Elasticsearch: {e}")
        return

    # Create the index
    try:
        INDEX_MAPPING = {
            "properties": {
                "image_embedding": {
                    "type": "dense_vector",
                    "dims": 512,
                    "index": True,
                    "similarity": "cosine",
                },
                "photo_id": {"type": "keyword"},
                "photo_image_url": {"type": "keyword"},
                "photo_description": {"type": "text"},
                "label": {"type": "keyword"},
            }
        }
        if not es.indices.exists(index=INDEX_NAME):
            es.indices.create(index=INDEX_NAME, mappings=INDEX_MAPPING)
    except Exception as e:
        print(f"An error occurred while creating the index: {e}")
        return

    # Download and extract the data
    unsplash_zip_file = "unsplash-research-dataset-lite-1.2.0.zip"
    if not os.path.exists(unsplash_zip_file):
        os.system(f"curl -L https://unsplash.com/data/lite/1.2.0 -o {unsplash_zip_file}")
        with zipfile.ZipFile(unsplash_zip_file, "r") as zip_ref:
            zip_ref.extractall("data/unsplash/")

    # Load the data
    df_unsplash = pd.read_csv("data/unsplash/photos.tsv000", sep="\t", header=0)
    df_unsplash.fillna("", inplace=True)
    # Add a placeholder label column for now. User will replace this with actual labels.
    df_unsplash["label"] = ""

    # Generate embeddings and index in batches
    batch_size = 50
    for i in range(0, len(df_unsplash), batch_size):
        batch_df = df_unsplash.iloc[i:i+batch_size]
        images = [Image.open(requests.get(url, stream=True).raw) for url in batch_df["photo_image_url"]]
        
        inputs = processor(text=None, images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=inputs.pixel_values.to(device))
        
        actions = []
        for j, row in enumerate(batch_df.iterrows()):
            index, data = row
            actions.append({
                "_index": INDEX_NAME,
                "_id": data["photo_id"],
                "_source": {
                    "photo_id": data["photo_id"],
                    "photo_image_url": data["photo_image_url"],
                    "photo_description": data["photo_description"],
                    "label": data["label"],
                    "image_embedding": image_features[j].cpu().numpy().tolist(),
                }
            })
        
        try:
            parallel_bulk(client=es, actions=actions)
            print(f"Indexed batch {i // batch_size + 1}")
        except Exception as e:
            print(f"An error occurred during indexing: {e}")

# --- Searching ---
def search_images(query: str):
    """Searches for images in Elasticsearch based on a text query."""
    es = get_es_client()
    if not es:
        return {"error": "Elasticsearch credentials not configured."}

    inputs = processor(text=[query], padding=True, return_tensors="pt")
    with torch.no_grad():
        text_features = model.get_text_features(input_ids=inputs.input_ids.to(device), attention_mask=inputs.attention_mask.to(device))

    query_embedding = text_features[0].cpu().numpy().tolist()

    knn_query = {
        "field": "image_embedding",
        "k": 5,
        "num_candidates": 100,
        "query_vector": query_embedding,
    }

    response = es.search(index=INDEX_NAME, knn=knn_query, source=["photo_image_url", "photo_description", "label"])
    return response.body["hits"]["hits"]