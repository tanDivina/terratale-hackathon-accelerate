# test_live_api.py
import asyncio
import os
from dotenv import load_dotenv
import traceback

print("--- Starting Live API Test ---")

# Load environment variables
load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

try:
    print("\n--- Attempting to import and use both APIs ---")
    # Import the standard library for GenerativeModel
    import google.generativeai as standard_api_genai
    # Import the parent google module to try and access the Client object
    from google import genai as google_root_genai
    from google.genai import types

    print("Imports successful.")

    # 1. Configure and test the standard API
    standard_api_genai.configure(api_key=API_KEY)
    print("Standard API configured successfully.")
    
    print("Testing standard text model...")
    text_model = standard_api_genai.GenerativeModel("models/gemini-2.5-flash")
    text_response = asyncio.run(text_model.generate_content_async("Hello"))
    print(f"Standard model response: {text_response.text.strip()}")
    print("Standard API test successful.")

    # 2. Create the Live API client and test connection
    print("\nCreating Live API client...")
    live_client = google_root_genai.Client(api_key=API_KEY)
    print("Live API client created successfully.")

    async def test_live_connection():
        print("Testing Live API connection...")
        async with live_client.aio.live.connect(
            model="gemini-live-2.5-flash-preview",
        ) as session:
            print("Live API connection successful! Session established.")
            # Just connect and immediately close for this test.
            # No need to send/receive.

    asyncio.run(test_live_connection())
    print("Live API connection test successful.")

    print("\n--- SUCCESS: Both APIs can coexist using aliased imports! ---")

except Exception as e:
    print(f"\n--- FAILED: An error occurred: {e} ---")
    traceback.print_exc()
