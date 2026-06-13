const express = require('express');
const router = express.Router();

// In-memory items store
let items = [
  { id: 1, name: 'Item 1', userId: 1 },
  { id: 2, name: 'Item 2', userId: 1 },
  { id: 3, name: 'Item 3', userId: 2 },
];
let nextId = 4;

// GET /items — list items for authenticated user
router.get('/', (req, res) => {
  const userItems = items.filter(item => item.userId === req.userId);
  res.json(userItems);
});

// POST /items — create new item
router.post('/', (req, res) => {
  const { name } = req.body;
  if (!name) {
    return res.status(400).json({ error: 'Name is required' });
  }
  const newItem = { id: nextId++, name, userId: req.userId };
  items.push(newItem);
  res.status(201).json(newItem);
});

// DELETE /items/:id — delete item
router.delete('/:id', (req, res) => {
  const itemId = parseInt(req.params.id);
  const item = items.find(i => i.id === itemId && i.userId === req.userId);
  if (!item) {
    return res.status(404).json({ error: 'Item not found' });
  }
  items = items.filter(i => i.id !== itemId);
  res.json({ message: 'Item deleted' });
});

module.exports = router;
