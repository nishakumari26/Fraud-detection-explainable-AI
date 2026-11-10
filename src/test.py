
import os
from google import genai

# Load API key securely from environment variable
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY not found. Please set it in your environment."
    )

# Initialize Gemini client
client = genai.Client(api_key=api_key)

# List available Gemini models
for model in client.models.list():
    print(model.name)
