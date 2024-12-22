async function getRecommendations() {
    const movieTitle = document.getElementById("movie-title").value.trim();
    const recommendationsList = document.getElementById("recommendations");
    const errorMessage = document.getElementById("error-message");

    // Clear previous results and error messages
    recommendationsList.innerHTML = "";
    errorMessage.textContent = "";

    if (movieTitle) {
        // Show a loading indicator
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

            // Process the response
            const data = await response.json();
            recommendationsList.innerHTML = ""; // Clear loading indicator

            if (response.ok && data.status === 'success') {
                // Check if recommendations are available
                if (data.recommendations && Array.isArray(data.recommendations) && data.recommendations.length > 0) {
                    data.recommendations.forEach(movie => {
                        const listItem = document.createElement("li");
                        listItem.textContent = movie; // Display movie title and year
                        listItem.classList.add("recommendation-item");
                        listItem.addEventListener("click", () => {
                            document.getElementById("movie-title").value = movie.split('(')[0].trim(); // Set input field with movie title
                        });
                        recommendationsList.appendChild(listItem);
                    });
                } else {
                    recommendationsList.innerHTML = "<li>No recommendations found for this movie.</li>";
                }
            } else if (data.status === 'error') {
                errorMessage.textContent = data.message || "No recommendations available.";
                if (data.suggestions && Array.isArray(data.suggestions)) {
                    const suggestionList = document.createElement("ul");
                    suggestionList.textContent = "Did you mean:";
                    data.suggestions.forEach(suggestion => {
                        const suggestionItem = document.createElement("li");
                        suggestionItem.textContent = suggestion;
                        suggestionItem.classList.add("suggestion-item");
                        suggestionItem.addEventListener("click", () => {
                            document.getElementById("movie-title").value = suggestion;
                            getRecommendations(); // Retry with the suggestion
                        });
                        suggestionList.appendChild(suggestionItem);
                    });
                    recommendationsList.appendChild(suggestionList);
                }
            } else {
                recommendationsList.innerHTML = "<li>An error occurred. Please try again later.</li>";
            }
        } catch (error) {
            console.error("Error fetching recommendations:", error);
            recommendationsList.innerHTML = "<li>Failed to fetch recommendations. Please try again later.</li>";
        }
    } else {
        errorMessage.textContent = "Please enter a movie title!";
    }
}

// Attach the event listener when the DOM is fully loaded
document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("recommend-form");
    form.addEventListener("submit", (event) => {
        event.preventDefault();
        getRecommendations();
    });
});
