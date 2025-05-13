// document.addEventListener("DOMContentLoaded", () => {
//     let index = 0;
//     const slides = document.querySelectorAll(".slides img");

//     function showSlide() {
//         slides.forEach((slide, i) => {
//             slide.style.opacity = i === index ? "1" : "0";
//         });
//         index = (index + 1) % slides.length;
//     }

//     setInterval(showSlide, 3000); // Change image every 3 seconds
//     showSlide(); // Initial call

//     // Profile Dropdown
//     const profileBtn = document.getElementById("profile-btn");
//     const logoutMenu = document.getElementById("logout-menu");

//     profileBtn.addEventListener("click", (event) => {
//         event.preventDefault();
//         logoutMenu.style.display = logoutMenu.style.display === "block" ? "none" : "block";
//     });

//     // Close dropdown when clicking outside
//     document.addEventListener("click", (event) => {
//         if (!profileBtn.contains(event.target) && !logoutMenu.contains(event.target)) {
//             logoutMenu.style.display = "none";
//         }
//     });
// });


// ------------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", async () => {
    let index = 0;
    const slides = document.querySelectorAll(".slides img");

    function showSlide() {
        slides.forEach((slide, i) => {
            slide.style.opacity = i === index ? "1" : "0";
        });
        index = (index + 1) % slides.length;
    }

    setInterval(showSlide, 3000); // Change image every 3 seconds
    showSlide(); // Initial call

    // Profile Dropdown
    const profileBtn = document.getElementById("profile-btn");
    const logoutMenu = document.getElementById("logout-menu");

    profileBtn.addEventListener("click", (event) => {
        event.preventDefault();
        logoutMenu.style.display = logoutMenu.style.display === "block" ? "none" : "block";
    });

    // Close dropdown when clicking outside
    document.addEventListener("click", (event) => {
        if (!profileBtn.contains(event.target) && !logoutMenu.contains(event.target)) {
            logoutMenu.style.display = "none";
        }
    });

    // ✅ Check authentication status
    async function checkAuth() {
        try {
            const response = await fetch("/api/auth/check-auth");
            const data = await response.json();

            if (data.authenticated) {
                // ✅ User is logged in: Show "Logout" option
                profileBtn.innerHTML = `<i class="fa-solid fa-user"></i> ${data.user.username}`;
                logoutMenu.innerHTML = `<a href="#" id="logout-btn">Logout</a>`;
                
                // Add logout functionality
                document.getElementById("logout-btn").addEventListener("click", async () => {
                    await fetch("/api/auth/logout");
                    window.location.href = "/login.html"; // Redirect to login page
                });
            } else {
                // ❌ User is not logged in: Show "Login" & "Signup" options
                logoutMenu.innerHTML = `
                    <a href="/Login/login.html">Login</a>
                    <a href="/Login/signup.html">Signup</a>
                `;
            }
        } catch (error) {
            console.error("Error checking authentication:", error);
        }
    }

    checkAuth(); // 🔄 Run authentication check on page load
});
