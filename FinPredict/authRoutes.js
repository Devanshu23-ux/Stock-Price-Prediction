const express = require("express");
const router = express.Router();
const User = require("./user"); // Ensure this model exists
const bcrypt = require("bcrypt");

// Signup Route
router.post("/signup", async (req, res) => {
    try {
        console.log("🔹 Signup Request Received:", req.body);

        const { username, email, password } = req.body;

        if (!username || !email || !password) {
            return res.status(400).json({ message: "All fields are required" });
        }

        // Check if user already exists
        const existingUser = await User.findOne({ email });
        if (existingUser) {
            return res.status(400).json({ message: "User already exists" });
        }

        // Hash password
        const hashedPassword = await bcrypt.hash(password, 10);

        // Create new user
        const newUser = new User({ username, email, password: hashedPassword });
        await newUser.save();

        res.status(201).json({ success: true, message: "User registered successfully" });
    } catch (error) {
        console.error("❌ Signup Error:", error);
        res.status(500).json({ success: false, message: "Server Error", error: error.message });
    }
});

// Login Route with Session Management
router.post("/login", async (req, res) => {
    try {
        const { email, password } = req.body;

        if (!email || !password) {
            return res.status(400).json({ success: false, message: "Email and password are required" });
        }

        const user = await User.findOne({ email });

        if (!user) {
            return res.status(400).json({ success: false, message: "User not found" });
        }

        const isMatch = await bcrypt.compare(password, user.password);
        if (!isMatch) {
            return res.status(400).json({ success: false, message: "Invalid credentials" });
        }

        // Store user session
        req.session.user = { 
            _id: user._id, // Make sure to include _id
            id: user._id,  // For backward compatibility
            username: user.username, 
            email: user.email 
        };

        res.json({ success: true, message: "Login successful", user: req.session.user });
    } catch (error) {
        console.error("❌ Login Error:", error);
        res.status(500).json({ success: false, message: "Server Error", error });
    }
});

// Logout Route
router.get("/logout", (req, res) => {
    req.session.destroy((err) => {
        if (err) {
            return res.status(500).json({ success: false, message: "Logout failed" });
        }
        res.clearCookie("connect.sid"); // Clear session cookie
        res.json({ success: true, message: "Logout successful" });
    });
});

// Check Authentication Route
router.get("/check-auth", (req, res) => {
    if (req.session.user) {
        res.json({ success: true, authenticated: true, user: req.session.user });
    } else {
        res.json({ success: false, authenticated: false, message: "Not logged in" });
    }
});

// Get user's watchlist
router.get("/watchlist", async (req, res) => {
    try {
        if (!req.session.user) {
            return res.status(401).json({ success: false, message: "Not authenticated" });
        }
        
        const user = await User.findById(req.session.user._id);
        if (!user) {
            return res.status(404).json({ success: false, message: "User not found" });
        }
        
        res.json({ success: true, watchlist: user.watchlist || [] });
    } catch (error) {
        console.error("❌ Error fetching watchlist:", error);
        res.status(500).json({ success: false, message: "Server error", error: error.message });
    }
});

// Add stock to watchlist
router.post("/watchlist/add", async (req, res) => {
    try {
        if (!req.session.user) {
            return res.status(401).json({ success: false, message: "Not authenticated" });
        }
        
        const { company, symbol } = req.body;
        
        if (!company || !symbol) {
            return res.status(400).json({ success: false, message: "Company name and symbol are required" });
        }
        
        const user = await User.findById(req.session.user._id);
        if (!user) {
            return res.status(404).json({ success: false, message: "User not found" });
        }
        
        // Initialize watchlist if it doesn't exist
        if (!user.watchlist) {
            user.watchlist = [];
        }
        
        // Check if stock already exists in watchlist
        const stockExists = user.watchlist.some(item => item.symbol === symbol);
        
        if (stockExists) {
            return res.status(400).json({ success: false, message: "Stock already in watchlist" });
        }
        
        // Add to watchlist
        user.watchlist.push({ company, symbol });
        
        await user.save();
        
        res.json({ 
            success: true, 
            message: "Stock added to watchlist",
            watchlist: user.watchlist
        });
    } catch (error) {
        console.error("❌ Error adding to watchlist:", error);
        res.status(500).json({ success: false, message: "Server error", error: error.message });
    }
});

// Remove stock from watchlist
router.post("/watchlist/remove", async (req, res) => {
    try {
        if (!req.session.user) {
            return res.status(401).json({ success: false, message: "Not authenticated" });
        }
        
        const { symbol } = req.body;
        
        if (!symbol) {
            return res.status(400).json({ success: false, message: "Stock symbol is required" });
        }
        
        const user = await User.findById(req.session.user._id);
        if (!user) {
            return res.status(404).json({ success: false, message: "User not found" });
        }
        
        // Remove from watchlist
        if (user.watchlist) {
            user.watchlist = user.watchlist.filter(item => item.symbol !== symbol);
        }
        
        await user.save();
        
        res.json({ 
            success: true, 
            message: "Stock removed from watchlist",
            watchlist: user.watchlist
        });
    } catch (error) {
        console.error("❌ Error removing from watchlist:", error);
        res.status(500).json({ success: false, message: "Server error", error: error.message });
    }
});

module.exports = router;