// Function to handle movie title click
document.addEventListener("DOMContentLoaded", () => {
    const movieItems = document.querySelectorAll(".movie-item");

    movieItems.forEach(item => {
        item.addEventListener("click", () => {
            const movieTitle = item.getAttribute("data-title");
            
            // Use sessionStorage to pass data between pages
            sessionStorage.setItem("selectedMovie", movieTitle);

            // Redirect back to the main page
            window.location.href = "/"; // Adjust this URL if needed
        });
    });
});

