import requests
from bs4 import BeautifulSoup
import csv

# Function to fetch artworks from the API
def fetch_artworks_aic():
    search_url = "https://api.artic.edu/api/v1/artworks/search"
    object_url = "https://api.artic.edu/api/v1/artworks/"
    search_params = {"query": "Impressionism", "limit": 5, "fields": "id,title,image_id"}
    response = requests.get(search_url, params=search_params)
    data = response.json()
    
    artworks = []
    for artwork in data["data"]:
        object_response = requests.get(f"{object_url}{artwork['id']}", params={"fields": "id,title,image_id"})
        object_data = object_response.json()["data"]
        title = object_data["title"]
        image_id = object_data["image_id"]
        image_url = f"https://www.artic.edu/iiif/2/{image_id}/full/843,/0/default.jpg"
        artworks.append({"title": title, "image_url": image_url, "id": artwork['id']})
    return artworks

# Function to scrape artwork descriptions
def scrape_description(artwork_id):
    detail_url = f"https://www.artic.edu/artworks/{artwork_id}"  # Hypothetical detail page URL
    response = requests.get(detail_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    description_selector = '#content > article > div.o-article__body.o-blocks > div.o-blocks > p' # Hypothetical selector
    description_element = soup.select_one(description_selector)
    return description_element.text.strip() if description_element else "Description not found."

# Main function to fetch artworks, scrape descriptions, and save to CSV
def main():
    artworks = fetch_artworks_aic()
    with open('artwork_data.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['title', 'image_url', 'description']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for artwork in artworks:
            print(f"Fetching description for artwork: {artwork['title']}")
            description = scrape_description(artwork['id'])
            artwork['description'] = description
            writer.writerow({'title': artwork['title'], 'image_url': artwork['image_url'], 'description': description})

if __name__ == "__main__":
    main()
