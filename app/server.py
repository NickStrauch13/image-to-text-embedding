from flask import Flask, render_template, request, jsonify
from utils import get_images_from_pinecone

app = Flask(__name__)

@app.route('/')
def home():
    # Serve the search HTML page
    return render_template('artlookup.html')

@app.route('/search', methods=['GET'])
def search():
    '''
    Search for images similar to the query and return the URLs.
    '''
    # Get the search query from the request's query string
    query = request.args.get('query')
    
    # Get image URLs for the query
    image_urls = get_images_from_pinecone(query)
    
    # Return the list of URLs as JSON
    return jsonify(image_urls)

if __name__ == '__main__':
    app.run(debug=True)
