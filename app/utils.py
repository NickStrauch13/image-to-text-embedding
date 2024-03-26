from clip_utils import CustomCLIPModel
from pinecone import Pinecone
import os
from dotenv import load_dotenv
import torch
import requests


def embed_text_clip(text_prompt, model_path='models/art_clip_model.pth', device='cuda' if torch.cuda.is_available() else 'cpu'):
    '''
    Embed the text description using the custom CLIP model.
    '''
    # Load the model
    model = CustomCLIPModel()
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Get the image embedding
    with torch.no_grad():
        text_emb = model.embed_text(text_prompt)
    
    return text_emb.cpu().numpy().flatten().tolist()


def get_images_from_pinecone(input_query, num_images=3):
    '''
    Get images similar to the input query from Pinecone.
    '''
    # Embed the input query using the CLIP model
    query_embedding = embed_text_clip(input_query)

    # Query Pinecone for similar images to the embedding
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index("art-module-project")

    results = index.query(
        vector=query_embedding,
        top_k=num_images,
        include_metadata=True
    )

    # print(results)
    # Extract the image URLs from the results
    image_urls = [match['metadata']['image_url'] for match in results['matches']]

    # For now, return dummy URLs
    return image_urls