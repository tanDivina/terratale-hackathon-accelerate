import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Make sure it is set in your .env file.")

genai.configure(api_key=api_key)

print("--- Available Models for generateContent ---")
for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)
print("----------------------------------------")
