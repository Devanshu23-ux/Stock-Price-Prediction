const mongoose = require("mongoose");

const userSchema = new mongoose.Schema({
    username: { type: String, required: true, unique: true }, // ✅ Ensure 'username' is used
    email: { type: String, required: true, unique: true },
    password: { type: String, required: true },
    watchlist: { type: Array, default: [] }
});

module.exports = mongoose.model("User", userSchema);


