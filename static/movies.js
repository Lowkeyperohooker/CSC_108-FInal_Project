document.addEventListener("DOMContentLoaded", () => {
    const moviesList = document.querySelector(".movies-list");

    if (moviesList) {
        moviesList.addEventListener("click", (event) => {
            const target = event.target;

            if (target.classList.contains("movie-item")) {
                const movieTitle = target.getAttribute("data-title");

                if (movieTitle) {
                    // Log the movie title to confirm it's being set
                    console.log("Selected movie:", movieTitle);

                    // Store the selected movie title in sessionStorage
                    sessionStorage.setItem("selectedMovie", movieTitle);

                    // Redirect to the index page
                    window.location.href = "/";
                }
            }
        });
    }
});
