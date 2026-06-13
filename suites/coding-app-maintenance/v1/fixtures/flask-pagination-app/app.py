"""Simple Flask REST API with items — no pagination (to be added by the model)."""
from flask import Flask, jsonify, request

app = Flask(__name__)

# Seed 50 items
items = [{"id": i, "name": f"Item {i}", "description": f"Description for item {i}"} for i in range(1, 51)]


@app.route("/items", methods=["GET"])
def get_items():
    """Return all items. TODO: Add pagination."""
    return jsonify(items)


@app.route("/items/<int:item_id>", methods=["GET"])
def get_item(item_id):
    """Return a single item by ID."""
    item = next((i for i in items if i["id"] == item_id), None)
    if item is None:
        return jsonify({"error": "Item not found"}), 404
    return jsonify(item)


@app.route("/items", methods=["POST"])
def create_item():
    """Create a new item."""
    data = request.get_json()
    if not data or "name" not in data:
        return jsonify({"error": "Name is required"}), 400
    new_id = max(i["id"] for i in items) + 1 if items else 1
    new_item = {
        "id": new_id,
        "name": data["name"],
        "description": data.get("description", ""),
    }
    items.append(new_item)
    return jsonify(new_item), 201


if __name__ == "__main__":
    app.run(debug=True, port=5001)
