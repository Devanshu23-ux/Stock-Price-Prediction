document.addEventListener('DOMContentLoaded', function() {
    const predictBtn = document.getElementById('predict-btn');
    const stockSelect = document.getElementById('stocks');
    const predictionsDiv = document.getElementById('predictions');
    const loadingDiv = document.getElementById('loading');
    let chart;
    
    predictBtn.addEventListener('click', async function() {
        // Get selected tickers
        const selectedTickers = Array.from(stockSelect.selectedOptions).map(option => option.value);
        
        if (selectedTickers.length === 0) {
            alert('Please select at least one stock')// // Get stock details from URL parameters
            // const urlParams = new URLSearchParams(window.location.search);
            // const company = urlParams.get('company');
            // const symbol = urlParams.get('symbol');
            
            // if (company && symbol) {
            //     document.getElementById("stock-name").innerText = `Selected Stock: ${company} (${symbol})`;
            // } else {
            //     document.getElementById("stock-name").innerText = "No stock selected!";
            // }
            
            // // Function to fetch predicted prices dynamically (Replace with actual API call)
            // async function fetchPredictedPrices(symbol, timeRange) {
            //     try {
            //         // Placeholder API call (Replace with actual API endpoint)
            //         const response = await fetch(`https://your-api.com/predict?stock=${symbol}&range=${timeRange}`);
            //         if (!response.ok) throw new Error("Failed to fetch prediction data");
            
            //         const data = await response.json();
            //         return data.prices; // Expecting an array of { day: "Monday", price: "102" }
            //     } catch (error) {
            //         console.error("Error fetching prediction data:", error);
            //         return [];
            //     }
            // }
            
            // document.getElementById("predict-btn").addEventListener("click", async function() {
            //     const selectedTime = document.getElementById("time-range").value;
            //     const predictionTable = document.getElementById("prediction-table");
            //     predictionTable.innerHTML = "<tr><td colspan='2'>Loading...</td></tr>"; // Show loading state
            
            //     // Fetch predictions dynamically
            //     const predictedData = await fetchPredictedPrices(symbol, selectedTime);
            //     predictionTable.innerHTML = ""; // Clear previous data
            
            //     if (predictedData.length > 0) {
            //         predictedData.forEach(({ day, price }) => {
            //             let tr = document.createElement("tr");
            //             tr.innerHTML = `<td>${day}</td><td>$${price}</td>`;
            //             predictionTable.appendChild(tr);
            //         });
            //     } else {
            //         predictionTable.innerHTML = "<tr><td colspan='2'>No prediction data available</td></tr>";
            //     }
            // });
            
            //----------------------------------------------------------------------
            
            // Get stock details from URL parameters
            const urlParams = new URLSearchParams(window.location.search);
            const company = urlParams.get('company');
            const symbol = urlParams.get('symbol');
            
            // Display selected stock name
            document.addEventListener("DOMContentLoaded", () => {
                if (company && symbol) {
                    document.getElementById("stock-name").innerText = `Selected Stock: ${company} (${symbol})`;
                } else {
                    document.getElementById("stock-name").innerText = "No stock selected!";
                }
            });
            
            // Mock function to generate a fake predicted price
            function getPredictedPrice() {
                return (Math.random() * (500 - 100) + 100).toFixed(2); // Generates a price between 100 and 500
            }
            
            // Handle predict button click
            document.getElementById("predict-btn").addEventListener("click", () => {
                const selectedDate = document.getElementById("date-picker").value;
                const predictionTable = document.getElementById("prediction-table");
            
                if (!selectedDate) {
                    alert("Please select a date first!");
                    return;
                }
            
                if (!company || !symbol) {
                    alert("No stock selected!");
                    return;
                }
            
                // Clear previous results
                predictionTable.innerHTML = "";
            
                // Generate a fake highest price for that date
                const predictedPrice = getPredictedPrice();
            
                // Create a new row in the table
                let tr = document.createElement("tr");
                tr.innerHTML = `
                    <td>${company} (${symbol})</td>
                    <td>${selectedDate}</td>
                    <td>$${predictedPrice}</td>
                `;
                predictionTable.appendChild(tr);
            });
            ;
            return;
        }
        
        // Show loading indicator
        loadingDiv.style.display = 'flex';
        predictionsDiv.innerHTML = '';
        
        try {
            // Send request to backend
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ tickers: selectedTickers }),
            });
            
            const predictions = await response.json();
            
            // Display predictions
            let html = '<table><tr><th>Stock</th><th>Predicted High (₹)</th></tr>';
            const labels = [];
            const values = [];
            
            for (const [ticker, price] of Object.entries(predictions)) {
                html += `<tr><td>${ticker}</td><td>₹${parseFloat(price).toFixed(2)}</td></tr>`;
                labels.push(ticker);
                values.push(parseFloat(price));
            }
            
            html += '</table>';
            predictionsDiv.innerHTML = html;
            
            // Create chart
            if (chart) {
                chart.destroy();
            }
            
            const ctx = document.getElementById('chart').getContext('2d');
            chart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Predicted High Price (₹)',
                        data: values,
                        backgroundColor: 'rgba(54, 162, 235, 0.6)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    scales: {
                        y: {
                            beginAtZero: false
                        }
                    }
                }
            });
            
        } catch (error) {
            predictionsDiv.innerHTML = `<p class="error">Error: ${error.message}</p>`;
        } finally {
            // Hide loading indicator
            loadingDiv.style.display = 'none';
        }
    });
});