import torch
import requests
from PIL import Image
from io import BytesIO
import pinecone
import sqlite3
import os
from clip_utils import CustomCLIPModel, Config

def encode_image(image_url, model, device):
    '''
    Encodes an image using CLIP model and 
    returns the embedding as a numpy array.
    '''
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Error downloading image from {image_url}")
        print(e)
        return None
    
    image_processed = Config.clip_transform(image).to(device)

    with torch.no_grad():
        image_features = model.embed_image(image_processed)
    return image_features.cpu().numpy()

def load_data(sqlite_db='artworks.db'):
    '''
    Load art data from the 
    '''
    connection = sqlite3.connect(sqlite_db)
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM artworks')
    art_data = cursor.fetchall()
    cursor.close()
    connection.close()
    return art_data

def populate_pinecone(data, model, device):
    '''
    Populate Pinecone with image embeddings
    '''
    # Initialize Pinecone
    pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment='gcp-starter')
    index_name = "art-module-project"

    # Check if the index exists, and create it if not
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=768, metric="cosine")  # Adjust the dimension according to your model's output

    index = pinecone.Index(index_name)

    # Process and insert the images into Pinecone
    for idx, (image_url, description) in enumerate(data):
        image_embedding = encode_image(image_url, model, device)
        if image_embedding is None:
            print(f"Skipping image {image_url}")
            continue
        index.upsert(vectors=[(str(idx), image_embedding.flatten())])

        if idx % 25 == 0:
            print(f"Embedding {idx} uploaded to Pinecone. for {image_url}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Load data
    art_data = load_data("notebooks/artworks.db")

    # Load Finetuned model
    model = CustomCLIPModel()
    model.load_state_dict(torch.load("models/art_clip_model.pth", map_location=device))
    print("Model loaded.")

    # Populate Pinecone
    populate_pinecone(art_data, model, device)

