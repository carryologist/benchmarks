const express = require('express');
const authMiddleware = require('./middleware/auth');
const itemsRouter = require('./routes/items');

const app = express();
app.use(express.json());

// Simple in-memory user store
const users = [
  { id: 1, username: 'admin', password: 'admin123' }
];

// Login endpoint — sets token in x-auth-token header
app.post('/login', (req, res) => {
  const { username, password } = req.body;
  const user = users.find(u => u.username === username && u.password === password);
  if (!user) {
    return res.status(401).json({ error: 'Invalid credentials' });
  }
  const token = Buffer.from(`${user.id}:${Date.now()}`).toString('base64');
  // BUG: Sets token in x-auth-token but middleware checks authorization
  res.set('x-auth-token', token);
  res.json({ message: 'Login successful', token });
});

// Protected routes
app.use('/items', authMiddleware, itemsRouter);

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));

module.exports = app;
