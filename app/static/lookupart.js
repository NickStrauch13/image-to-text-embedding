document.getElementById('searchForm').addEventListener('submit', function(e) {
    e.preventDefault();
    const query = document.getElementById('searchQuery').value;
    const numImages = document.getElementById('numImages').value; // Get the number of images
    fetch(`/search?query=${encodeURIComponent(query)}&numImages=${encodeURIComponent(numImages)}`)
        .then(response => response.json())
        .then(data => {
            const resultsSection = document.getElementById('resultsSection');
            resultsSection.innerHTML = ''; // Clear previous results
            data.forEach(url => {
                const imgElement = document.createElement('img');
                imgElement.src = url;
                resultsSection.appendChild(imgElement);
            });
        })
        .catch(error => console.error('Error fetching images:', error));
});
