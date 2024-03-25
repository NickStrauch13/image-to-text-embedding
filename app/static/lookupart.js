document.getElementById('searchForm').addEventListener('submit', function(e) {
    e.preventDefault();
    const query = document.getElementById('searchQuery').value;
    fetch(`/search?query=${encodeURIComponent(query)}`)
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