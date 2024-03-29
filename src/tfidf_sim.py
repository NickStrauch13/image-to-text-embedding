import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

'''
This code snippet demonstrates a simple TF-IDF similarity search using a SQLite database.
It computes the cosine similarity between a keyword vector and the TF-IDF vectors of artwork descriptions.
The similarity output metric is the cosine similarity score.
Smaller scores indicate less similarity, while larger scores indicate more similarity.
The larger the score, the smaller the angle between the two vectors, and the more similar the descriptions are.
'''

def search(keyword):
    '''
    Perform a TF-IDF similarity search on the artwork descriptions.
    '''
    # Start timing
    start_time = time.time()

    # Transform the keyword to the same vector space as the descriptions
    keyword_vec = vectorizer.transform([keyword])

    # Compute cosine similarity between the keyword vector and all descriptions
    similarity_scores = cosine_similarity(keyword_vec, tfidf_matrix)

    # Convert similarity scores to a DataFrame
    scores_df = pd.DataFrame(similarity_scores.T, columns=['similarity'])

    # Merge the similarity scores with the original DataFrame
    # This adds the 'image_url' and 'description' columns to the scores DataFrame
    merged_df = pd.concat([df, scores_df], axis=1)

    # Sort the merged DataFrame by similarity scores
    sorted_scores = merged_df.sort_values(by='similarity', ascending=False)

    # End timing
    end_time = time.time()
    
    # Print elapsed time
    print(f"Search took {end_time - start_time:.4f} seconds")

    # Return the sorted scores for review
    return sorted_scores

if __name__ == "__main__":
    # Connect to the SQLite database
    db_path = '../notebooks/artworks.db'  # Adjusted path to the database
    conn = sqlite3.connect(db_path)
    query = "SELECT image_url, description FROM artworks"
    df = pd.read_sql_query(query, conn)

    # Ensure the database connection is closed after the data is loaded
    conn.close()

    # Initialize a TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the descriptions to a vector
    tfidf_matrix = vectorizer.fit_transform(df['description'])

    # Example search
    keyword = "beach"
    results = search(keyword)
    print(results[['image_url', 'description', 'similarity']].head())  # Print only the top results for brevity
