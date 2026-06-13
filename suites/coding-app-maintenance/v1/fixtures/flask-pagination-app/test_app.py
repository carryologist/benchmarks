"""Tests for the items API."""
import pytest
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_get_all_items(client):
    """Should return all 50 items."""
    resp = client.get("/items")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data) == 50


def test_get_single_item(client):
    """Should return item by ID."""
    resp = client.get("/items/1")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["id"] == 1
    assert data["name"] == "Item 1"


def test_get_missing_item(client):
    """Should return 404 for non-existent item."""
    resp = client.get("/items/9999")
    assert resp.status_code == 404


def test_create_item(client):
    """Should create a new item."""
    resp = client.post("/items", json={"name": "New Item", "description": "Test"})
    assert resp.status_code == 201
    data = resp.get_json()
    assert data["name"] == "New Item"


def test_create_item_no_name(client):
    """Should reject item without name."""
    resp = client.post("/items", json={"description": "No name"})
    assert resp.status_code == 400
