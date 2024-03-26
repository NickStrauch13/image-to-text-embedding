import sqlite3
import pandas as pd
import time

'''
This code snippet demonstrates a simple keyword-based similarity search using a SQLite database.
It simply counts the occurrences of a keyword in the description of each artwork and sorts the results.
The similarity output metric is the count of keyword occurrences.
'''

# Function to count keyword occurrences in the description
def keyword_occurrences(description, keyword):
    '''
    Count the occurrences of a keyword in the description.
    '''
    return description.lower().count(keyword.lower())

def search(keyword):
    '''
    Perform a keyword-based similarity search on the artwork descriptions.
    '''
    # Start timing
    start_time = time.time()

    # Apply the keyword_occurrences function to each description
    df['similarity'] = df['description'].apply(lambda x: keyword_occurrences(x, keyword))

    # Sort the DataFrame by the 'similarity' column
    sorted_scores = df.sort_values(by='similarity', ascending=False)

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
    # Example search
    keyword = "beach"
    results = search(keyword)
    print(results[['image_url', 'description', 'similarity']].head())  # Print only the top results for brevity
