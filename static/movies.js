// Function to handle movie title clicks using event delegation
document.addEventListener("DOMContentLoaded", () => {
    const moviesList = document.querySelector(".movies-list");

    if (moviesList) {
        moviesList.addEventListener("click", (event) => {
            const target = event.target;

            if (target.classList.contains("movie-item")) {
                const movieTitle = target.getAttribute("data-title");

                // Store selected movie title in sessionStorage
                sessionStorage.setItem("selectedMovie", movieTitle);

                // Redirect to the main page or any other target
                window.location.href = "/"; // Adjust this URL if needed
            }
        });
    }
});
