async function getRecommendations() {
    const movieTitle = document.getElementById("movie-title").value;
    const recommendationsList = document.getElementById("recommendations");

    recommendationsList.innerHTML = ""; // Clear previous recommendations

    if (movieTitle) {
        try {
            const response = await fetch('/recommend', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ title: movieTitle })
            });

            const data = await response.json();

            if (data.status === 'success') {
                data.recommendations.forEach(movie => {
                    const listItem = document.createElement("li");
                    listItem.textContent = movie;
                    recommendationsList.appendChild(listItem);
                });
            } else {
                recommendationsList.innerHTML = `<li>${data.message}</li>`;
            }
        } catch (error) {
            console.error("Error fetching recommendations:", error);
            recommendationsList.innerHTML = "<li>Failed to fetch recommendations.</li>";
        }
    } else {
        alert("Please enter a movie title!");
    }
}
