async function getRecommendations() {
    const movieTitle = document.getElementById("movie-title").value.trim();
    const recommendationsList = document.getElementById("recommendations");

    recommendationsList.innerHTML = ""; // Clear previous recommendations

    if (movieTitle) {
        // Show loading indicator
        const loadingIndicator = document.createElement("li");
        loadingIndicator.textContent = "Fetching recommendations...";
        recommendationsList.appendChild(loadingIndicator);

        try {
            const response = await fetch('/recommend', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ title: movieTitle })
            });

            const data = await response.json();
            recommendationsList.innerHTML = ""; // Clear loading indicator

            if (response.ok && data.status === 'success') {
                if (data.recommendations.length > 0) {
                    data.recommendations.forEach(movie => {
                        const listItem = document.createElement("li");
                        listItem.textContent = movie;
                        listItem.classList.add("recommendation-item"); // Add a class for styling and functionality
                        listItem.addEventListener("click", () => {
                            document.getElementById("movie-title").value = movie; // Paste the clicked title into the input
                        });
                        recommendationsList.appendChild(listItem);
                    });
                } else {
                    recommendationsList.innerHTML = "<li>No recommendations found for this movie.</li>";
                }
            } else {
                recommendationsList.innerHTML = `<li>${data.message || 'An error occurred.'}</li>`;
            }
        } catch (error) {
            console.error("Error fetching recommendations:", error);
            recommendationsList.innerHTML = "<li>Failed to fetch recommendations. Please try again later.</li>";
        }
    } else {
        alert("Please enter a movie title!");
    }
}

document.addEventListener("DOMContentLoaded", () => {
    const movieInput = document.getElementById("movie-title");
    const selectedMovie = sessionStorage.getItem("selectedMovie"); // Retrieve the selected movie
    if (selectedMovie) {
        movieInput.value = selectedMovie; // Populate the input field
        sessionStorage.removeItem("selectedMovie"); // Clear sessionStorage
    }

    // Attach event listener for the "Get Recommendations" button
    document.getElementById("get-recommendations-btn").addEventListener("click", getRecommendations);
});
