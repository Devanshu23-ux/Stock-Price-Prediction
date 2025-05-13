// require("dotenv").config();
// // const express = require("express");
// // const mongoose = require("mongoose");
// // const path = require("path");

// // // Import authentication routes
// // const authRoutes = require("./authRoutes");

// // const app = express();
// // const PORT = process.env.PORT || 5000;
// // const mongoURI = process.env.MONGO_URI;

// // console.log("MONGO_URI:", mongoURI); // Debugging line

// // if (!mongoURI) {
// //     console.error("❌ MONGO_URI is missing in .env file");
// //     process.exit(1);
// // }

// // // Connect to MongoDB
// // mongoose
// //     .connect(mongoURI)
// //     .then(() => console.log("✅ MongoDB Connected"))
// //     .catch((err) => {
// //         console.error("❌ MongoDB Connection Error:", err);
// //         process.exit(1);
// //     });

// // const db = mongoose.connection;
// // db.on("disconnected", () => console.log("⚠️ MongoDB Disconnected"));

// // // Middleware to parse JSON & handle forms
// // app.use(express.json());
// // app.use(express.urlencoded({ extended: true }));

// // // Serve static frontend files
// // app.use(express.static(path.join(__dirname)));

// // // ✅ Register API Routes at /api/auth
// // app.use("/api/auth", authRoutes);

// // // Default route (For testing)
// // app.get("/", (req, res) => {
// //     res.sendFile(path.join(__dirname, "Login", "login.html"));
// // });

// // // 404 Handler
// // app.use((req, res) => {
// //     res.status(404).json({ message: "Route Not Found" });
// // });

// // // Start the server
// // app.listen(PORT, () => {
// //     console.log(`🚀 Server running on port ${PORT}`);
// // });

// // _____________________________________________________________________________________

require("dotenv").config();
const express = require("express");
const mongoose = require("mongoose");
const session = require("express-session");
const path = require("path");

// Import authentication routes
const authRoutes = require("./authRoutes");

const app = express();
const PORT = process.env.PORT || 5000;
const mongoURI = process.env.MONGO_URI;

console.log("MONGO_URI:", mongoURI); // Debugging line

if (!mongoURI) {
    console.error("❌ MONGO_URI is missing in .env file");
    process.exit(1);
}

// Connect to MongoDB
mongoose
    .connect(mongoURI)
    .then(() => console.log("✅ MongoDB Connected"))
    .catch((err) => {
        console.error("❌ MongoDB Connection Error:", err);
        process.exit(1);
    });

const db = mongoose.connection;
db.on("disconnected", () => console.log("⚠️ MongoDB Disconnected"));

// Middleware to parse JSON & handle forms
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Configure session
app.use(session({
    secret: process.env.SESSION_SECRET || "your_secret_key",
    resave: false,
    saveUninitialized: true,
    cookie: { secure: false } // Set to true if using HTTPS
}));

// Middleware to check if the user is logged in
function isAuthenticated(req, res, next) {
    if (req.session.user) {
        return next(); // User is authenticated, continue
    }
    res.redirect("/Login/login.html"); // Redirect to login if not authenticated
}

// Serve static frontend files
app.use(express.static(path.join(__dirname)));

// ✅ Register API Routes at /api/auth
app.use("/api/auth", authRoutes);

// Protected Routes (Require Login)
// app.get("/index.html", isAuthenticated, (req, res) => {
//     res.sendFile(path.join(__dirname, "index.html"));
// });
// app.get("/Watchlist/watchlist.html", isAuthenticated, (req, res) => {
//     res.sendFile(path.join(__dirname, "Watchlist", "watchlist.html"));
// });
// app.get("/search/search.html", isAuthenticated, (req, res) => {
//     res.sendFile(path.join(__dirname, "search", "search.html"));
// });

// Default route (Redirect to login)
app.get("/", (req, res) => {
    if (req.session.user) {
        res.redirect("/index.html"); // Redirect logged-in users to dashboard
    } else {
        res.redirect("/Login/login.html"); // Redirect to login page
    }
});

// 404 Handler
app.use((req, res) => {
    res.status(404).json({ message: "Route Not Found" });
});

// Start the server
app.listen(PORT, () => {
    console.log(`🚀 Server running on port ${PORT}`);
});
