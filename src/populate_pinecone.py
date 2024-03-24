import clip
import torch
import requests
from PIL import Image
from io import BytesIO
import pinecone
import sqlite3
import os

# Function to encode an image using CLIP
def encode_image(image_url, model, preprocess, device="cuda"):
    '''
    Encodes an image using CLIP model and 
    returns the image features as a numpy array.
    '''
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    image_preprocessed = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_preprocessed)
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

def populate_pinecone(model, preprocess, device):
    '''
    Populate Pinecone with image embeddings
    '''
    # Initialize Pinecone
    pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENVIRONMENT'))
    index_name = "ART_MODULE_PROJ"

    # Check if the index exists, and create it if not
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=768, metric="cosine")  # Adjust the dimension according to your model's output

    index = pinecone.Index(index_name)

    # Process and insert the images into Pinecone
    for idx, (image_url, description) in enumerate(museum_curation):
        image_embedding = encode_image(image_url, model, preprocess, device)
        index.upsert(vectors=[(str(idx), image_embedding.flatten())])

        print(f"Embedding {idx} uploaded to Pinecone. for {image_url}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    museum_curation = load_data()
    #TODO: Change the model to the one we trained
    model, preprocess = clip.load("ViT-B/32", device=device)
    print("Model loaded.")

    populate_pinecone(model, preprocess, device)

