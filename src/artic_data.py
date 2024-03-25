import requests
from bs4 import BeautifulSoup
import csv
import re
from lxml import html


def slugify(title):
    '''
    Generate a slug from the title.
    '''
    # Convert to lowercase and replace spaces and non-word characters with dashes
    slug = re.sub(r'\W+', '-', title.lower())
    return slug

def fetch_artworks_aic():
    '''
    Fetch artworks from the Art Institute of Chicago API.
    '''
    search_url = "https://api.artic.edu/api/v1/artworks/search"
    object_url = "https://api.artic.edu/api/v1/artworks/"
    # Define the search parameters
    search_params = {"query": "Impressionism", "limit": 80, "fields": "id,title,image_id"}
    response = requests.get(search_url, params=search_params)
    data = response.json()
    
    artworks = []
    for artwork in data["data"]:
        object_response = requests.get(f"{object_url}{artwork['id']}", params={"fields": "id,title,image_id"})
        object_data = object_response.json()["data"]
        title = object_data["title"]
        image_id = object_data["image_id"]
        image_url = f"https://www.artic.edu/iiif/2/{image_id}/full/843,/0/default.jpg"
        slug = slugify(title)  # Generate a slug from the title
        artworks.append({"title": title, "image_url": image_url, "id": artwork['id'], "slug": slug})
    return artworks


def scrape_description(artwork_id, slug):
    '''
    Scrape the artwork description from the Art Institute of Chicago website.
    '''
    detail_url = f"https://www.artic.edu/artworks/{artwork_id}/{slug}"  # Updated to include the slug
    response = requests.get(detail_url)
    if response.status_code == 200:
        # Parse the response content with lxml
        tree = html.fromstring(response.content)
        
        # Define the XPath to the description element
        description_xpath = '//*[@id="content"]/article/div[5]/div[1]/p/text()'
        
        description_elements = tree.xpath(description_xpath)
        if description_elements:
            return ' '.join([elem.strip() for elem in description_elements])  # Joining text elements if there are multiple
        else:
            return "Description not found."
    else:
        return "Failed to fetch the webpage."

def main():
    '''
    Main function to fetch and save artwork data.
    '''
    artworks = fetch_artworks_aic()
    # Save the artwork data to a CSV file
    with open('artwork_data.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['title', 'image_url', 'description']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Loop through the artworks and fetch the description for each
        for artwork in artworks:
            print(f"Fetching description for artwork: {artwork['title']}")
            description = scrape_description(artwork['id'], artwork['slug'])
            # Check if the description exists before saving
            if description and description != "Description not found.":
                writer.writerow({'title': artwork['title'], 'image_url': artwork['image_url'], 'description': description})
            else:
                print(f"No valid description found for {artwork['title']}; skipping.")

if __name__ == "__main__":
    main()
