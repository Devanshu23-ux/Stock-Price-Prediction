// Function to fetch stock data from JSON file
async function fetchStocks() {
    try {
        const response = await fetch('./static/nse-listed-stocks.json');
        if (!response.ok) {
            throw new Error("Failed to load stock data");
        }
        const stockData = await response.json();
        console.log("Stock Data Loaded:", stockData); // Debugging
        return stockData;
    } catch (error) {
        console.error('Error fetching stock data:', error);
        return {};
    }
}

// Function to add stock to the watchlist
function addToWatchlist(company, symbol) {
    let watchlist = JSON.parse(localStorage.getItem("watchlist")) || [];

    // Check if stock is already in watchlist
    if (!watchlist.some(stock => stock.symbol === symbol)) {
        watchlist.push({ company, symbol });
        localStorage.setItem("watchlist", JSON.stringify(watchlist));
        alert(`${company} added to watchlist`);
    } else {
        alert("Stock already in watchlist");
    }
}

// Function to redirect to the prediction page
function goToPrediction(company, symbol) {
    const url = `/predict?company=${encodeURIComponent(company)}&symbol=${encodeURIComponent(symbol)}`;
    window.location.href = url;
}

document.addEventListener("DOMContentLoaded", async () => {
    const stockDict = await fetchStocks();
    const searchInput = document.getElementById("search");
    const resultsDiv = document.getElementById("stock-list");

    searchInput.addEventListener("input", () => {
        const query = searchInput.value.trim().toLowerCase();
        resultsDiv.innerHTML = ""; // Clear previous results

        if (query) {
            const results = Object.entries(stockDict).filter(([company, symbol]) => 
                company.toLowerCase().includes(query) || symbol.toLowerCase().includes(query)
            );

            if (results.length > 0) {
                results.forEach(([company, symbol]) => {
                    // Create list item for stock
                    const listItem = document.createElement("li");
                    listItem.classList.add("stock-item");

                    // Stock text (company + symbol)
                    const stockText = document.createElement("span");
                    stockText.textContent = `${company} (${symbol})`;

                    // Predict button
                    const predictBtn = document.createElement("button");
                    predictBtn.textContent = "Predict";
                    predictBtn.classList.add("predict-btn");
                    predictBtn.addEventListener("click", () => goToPrediction(company, symbol));

                    // // Add to Watchlist button
                    // const watchlistBtn = document.createElement("button");
                    // watchlistBtn.textContent = "Add to Watchlist";
                    // watchlistBtn.classList.add("watchlist-btn");
                    // watchlistBtn.addEventListener("click", () => addToWatchlist(company, symbol));
                    

                    // Add to Watchlist button
const watchlistBtn = document.createElement("button");
watchlistBtn.classList.add("watchlist-btn");

// Create the heart icon
const heartIcon = document.createElement("i");
heartIcon.classList.add("fa-solid", "fa-heart");

// Add text and icon to the button
watchlistBtn.appendChild(heartIcon);
watchlistBtn.appendChild(document.createTextNode(''));

// Event listener for adding to watchlist
watchlistBtn.addEventListener("click", () => addToWatchlist(company, symbol));



                    // Append elements
                    listItem.appendChild(stockText);
                    listItem.appendChild(predictBtn);
                    listItem.appendChild(watchlistBtn);
                    resultsDiv.appendChild(listItem);
                });
            } else {
                resultsDiv.textContent = "No matching stocks found";
            }
        }
    });
});
