from fastapi import APIRouter

router = APIRouter()

# We will add more routers here later (e.g., detection, explainability)
# For now, we'll just register the health check which is in main.py
# Or we could move health check here. Let's keep it simple for now.