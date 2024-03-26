import requests

def fetch_artworks():
    '''
    Fetch artworks from the Met's public API.
    '''
    # Base URL for the Met's public API
    search_url = "https://collectionapi.metmuseum.org/public/collection/v1/search"
    object_url = "https://collectionapi.metmuseum.org/public/collection/v1/objects/"

    # Search for artworks that are paintings in the public domain
    search_params = {
        "hasImages": True,
        "isPublicDomain": True,
        "q": "paintings"
    }

    # Fetch a list of artwork object IDs
    search_response = requests.get(search_url, params=search_params)
    search_data = search_response.json()
    object_ids = search_data.get("objectIDs", [])[:10]  # Limit to first 10 results for example

    artworks = []

    for object_id in object_ids:
        # Fetch detailed information for each artwork
        object_response = requests.get(f"{object_url}{object_id}")
        object_data = object_response.json()

        # Extract relevant data
        title = object_data.get("title")
        artist = object_data.get("artistDisplayName")
        image_url = object_data.get("primaryImage")
        description = object_data.get("galleryText", "No description available.")  # Some artworks might not have a gallery text

        artwork_info = {
            "title": title,
            "artist": artist,
            "image_url": image_url,
            "description": description
        }

        artworks.append(artwork_info)

    return artworks

if __name__ == "__main__":
    # Fetch and print the artworks
    artworks = fetch_artworks()
    for artwork in artworks:
        print(f"Title: {artwork['title']}, Artist: {artwork['artist']}, Image URL: {artwork['image_url']}, Description: {artwork['description']}\n")
