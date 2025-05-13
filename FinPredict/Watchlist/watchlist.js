document.addEventListener("DOMContentLoaded", async () => {
    const watchlistContainer = document.getElementById("watchlist");
    const messageElement = document.getElementById("message");

    // Function to check authentication
    async function checkAuth() {
        try {
            const response = await fetch("/api/auth/check-auth");
            const data = await response.json();
            return data.authenticated;
        } catch (error) {
            console.error("Authentication check failed:", error);
            return false;
        }
    }

    // Function to fetch watchlist from API
    async function fetchWatchlist() {
        try {
            const response = await fetch("/api/auth/watchlist");
            
            if (!response.ok) {
                if (response.status === 401) {
                    // User not authenticated
                    messageElement.textContent = "Please log in to view your watchlist";
                    return null;
                }
                throw new Error("Failed to fetch watchlist");
            }
            
            const data = await response.json();
            return data.watchlist || [];
        } catch (error) {
            console.error("Error fetching watchlist:", error);
            messageElement.textContent = "Error loading watchlist. Please try again later.";
            return null;
        }
    }

    // Function to remove stock from watchlist
    async function removeFromWatchlist(symbol) {
        try {
            const response = await fetch("/api/auth/watchlist/remove", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ symbol })
            });
            
            if (!response.ok) {
                throw new Error("Failed to remove stock from watchlist");
            }
            
            // Refresh watchlist
            displayWatchlist();
        } catch (error) {
            console.error("Error removing stock:", error);
            alert("Error removing stock from watchlist");
        }
    }

    // Function to display watchlist
    async function displayWatchlist() {
        watchlistContainer.innerHTML = "";
        
        // First check if user is authenticated
        const isAuthenticated = await checkAuth();
        if (!isAuthenticated) {
            messageElement.textContent = "Please log in to view your watchlist";
            // Redirect to login if needed
            // window.location.href = "../Login/login.html";
            return;
        }
        
        // Fetch watchlist from API
        const watchlist = await fetchWatchlist();
        
        if (!watchlist) {
            return; // Error already handled in fetchWatchlist
        }
        
        if (watchlist.length === 0) {
            messageElement.textContent = "Your watchlist is empty.";
            return;
        }
        
        messageElement.textContent = ""; // Clear message

        watchlist.forEach((stock) => {
            const listItem = document.createElement("li");
            listItem.classList.add("watchlist-item");

            // Stock text
            const stockText = document.createElement("span");
            stockText.textContent = `${stock.company} (${stock.symbol})`;

            // Predict Button (Redirects to Prediction Page)
            const predictBtn = document.createElement("button");
  

            // Remove button
            const removeBtn = document.createElement("button");
            removeBtn.textContent = "Remove";
            removeBtn.classList.add("remove-btn");
            removeBtn.addEventListener("click", (event) => {
                event.preventDefault(); // Prevent navigation issues
                removeFromWatchlist(stock.symbol);
            });

            // Append elements
            listItem.appendChild(stockText);
            listItem.appendChild(predictBtn);
            listItem.appendChild(removeBtn);
            watchlistContainer.appendChild(listItem);
        });
    }

    // Check authentication and load watchlist on page load
    const isAuthenticated = await checkAuth();
    if (isAuthenticated) {
        displayWatchlist();
    } else {
        messageElement.textContent = "Please log in to view your watchlist";
        // Optionally redirect to login page
        // window.location.href = "../Login/login.html";
    }

    // Add logout functionality
    document.getElementById("logout-btn").addEventListener("click", async () => {
        try {
            const response = await fetch("/api/auth/logout");
            if (response.ok) {
                window.location.href = "../Login/login.html";
            }
        } catch (error) {
            console.error("Logout error:", error);
        }
    });
});