document.querySelector("form").addEventListener("submit", async (e) => {
    e.preventDefault();

    // Correct field name: Change 'name' to 'username'
    const username = document.querySelector('input[type="text"]').value; 
    const email = document.querySelector('input[type="email"]').value;
    const password = document.querySelectorAll('input[type="password"]')[0].value;
    const confirmPassword = document.querySelectorAll('input[type="password"]')[1].value;

    console.log("Username:", username);
    console.log("Email:", email);
    console.log("Password:", password);
    console.log("Confirm Password:", confirmPassword);

    if (password !== confirmPassword) {
        alert("Passwords do not match!");
        return;
    }

    try {
        const response = await fetch("http://localhost:5000/api/auth/signup", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username, email, password }), // Corrected: Use 'username' instead of 'name'
        });

        console.log("Response Status:", response.status);
        
        const data = await response.json();
        console.log("Response Data:", data);

        if (response.ok) {
            alert("Signup Successful!");
            window.location.href = "login.html"; // Redirect to login page
        } else {
            alert(data.message);
        }
    } catch (error) {
        console.error("Error:", error);
        alert("An error occurred during signup.");
    }
});

//________________________________________________________
