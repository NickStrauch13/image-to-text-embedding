[![CI](https://github.com/NickStrauch13/image-to-text-embedding/actions/workflows/python-ci.yml/badge.svg)](https://github.com/NickStrauch13/image-to-text-embedding/actions/workflows/python-ci.yml)

# Training Multimodel Image and Text Embeddings for Art Museum Search

Transfer learning both with [CLIP](https://github.com/openai/CLIP) and google ViT to learn a latent space for both text and images for use retrieving images from a database based on natrual language prompts.

## Data

Data was sourced from [Art Institute of Chicago](https://www.artic.edu/open-access/public-api) open api and stored in a local Sqlite3 database. The data consisted of images along with descriptions found at the museum.

## Training

Using Kaggle notebooks, found in `notebooks/`, we finetuned CLIP using the labeled art data. The custom CLIP model configuration can be found in `src/clip_utils.py`. The trained weights are [here](https://drive.google.com/file/d/1rGNPti0CtLzVJ4-zq7YGHHu1tSdUEWTD/view?usp=sharing).

## Pipeline

Once we had a trained model, we can use it for embedding both images and text. In order to build the art retrieval app, we first embedded all of the art images and saved those in to a Pinecone vector database. Now that the images are in the latent space, on inference we embed the text prompt in to the same space and pull the closest few images from pinecone based on cosine similarity.

## To Use

To run our application, you will need access to our Pinecone db or make one using the finetuned embedding model. After creating the database, updating .env, and downloading requirements simply run ```python app/server.py``` to locally host the application.

## Results

We found improvement with our finetuned model over the base CLIP model in our art specific task. On the test set, our model had a top-5 retrieval accuracy of about 0.36 while the base CLIP model had a top-5 accuracy of about 0.06.

