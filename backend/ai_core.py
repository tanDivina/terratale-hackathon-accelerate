"""
Core AI logic for TerraTale, generating dual text and audio responses.
This version uses the successful aliased import pattern from our test script.
"""
import os
import json
import asyncio

# Correct, aliased imports to handle the library's namespace conflict
import google.generativeai as standard_api_genai
from google.generativeai import types
from google.cloud import texttospeech
import google.auth.credentials
import google.oauth2.service_account

# Import persona configurations
from . import config

# --- Configuration ---
API_KEY = os.environ.get("GEMINI_API_KEY")
GOOGLE_APPLICATION_CREDENTIALS_JSON = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")

live_client = None
tts_client = None

if GOOGLE_APPLICATION_CREDENTIALS_JSON:
    info = json.loads(GOOGLE_APPLICATION_CREDENTIALS_JSON)
    credentials = google.oauth2.service_account.Credentials.from_service_account_info(info)
    standard_api_genai.configure(credentials=credentials)
    live_client = standard_api_genai.Client(credentials=credentials)
    tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
elif API_KEY:
    standard_api_genai.configure(api_key=API_KEY)
    live_client = standard_api_genai.Client(api_key=API_KEY)
    # This client uses standard Google Cloud authentication (not the API key)
    tts_client = texttospeech.TextToSpeechClient()

# --- Knowledge Base (CUSTOMIZE THIS) ---
def search_knowledge_base(query: str) -> list:
    """Placeholder for a real knowledge base search (e.g., Elasticsearch)."""
    print(f"Searching knowledge base for: {query}")
    return [
        "Manatees are large, fully aquatic, mostly herbivorous marine mammals. Their diet consists of seagrasses and other aquatic vegetation.",
        "The San San Pond Sak wetlands are a Ramsar site of international importance, located in the Bocas del Toro province of Panama.",
        "Local folklore speaks of the 'tulivieja', a spirit that protects the rivers and is said to appear as a woman with a monstrous face.",
        "The red mangrove, or 'Mangle Rojo', has distinctive prop roots that help stabilize coastlines and provide critical nursery habitat for fish and invertebrates."
    ]

# --- Core AI Functions ---
async def generate_text_response(query: str, context: list) -> str:
    print("Generating text response (Papito)...")
    model = standard_api_genai.GenerativeModel(config.TEXT_MODEL_NAME)
    model.system_instruction = config.TEXT_PERSONA_PROMPT
    context_str = "\n".join(context)
    full_prompt = f"Context:\n---\n{context_str}\n---\n\nQuestion: {query}"
    response = await model.generate_content_async(full_prompt)
    return response.text

def synthesize_papito_speech(text: str) -> bytes:
    """Synthesizes speech for Papito's text using a standard voice."""
    if not tts_client:
        raise ConnectionError("Text-to-Speech client not initialized.")

    # Load pronunciation dictionary
    pronunciation_file = os.path.join(os.path.dirname(__file__), "pronunciations.json")
    with open(pronunciation_file, 'r') as f:
        pronunciations = json.load(f)

    # Apply SSML for pronunciations
    for word, pronunciation in pronunciations.items():
        text = text.replace(word, f'<phoneme alphabet="x-ipa" ph="{pronunciation}">{word}</phoneme>')

    ssml_text = f"<speak>{text}</speak>"
    
    synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)
    voice = texttospeech.VoiceSelectionParams(language_code="en-US", name="en-US-Standard-D")
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    
    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    return response.audio_content


# --- Live API Session Class for Papito ---
class MateoAudioSession:
    def __init__(self, client_websocket):
        self.client_websocket = client_websocket

    async def generate_and_stream_audio(self, query: str, context: list):
        print("Generating audio response (Mateo)...")

        # Combine prompts
        system_prompt = f"{config.AUDIO_PERSONA_PROMPT}"
        context_str = "\n".join(context)
        full_prompt = f"System Prompt: {system_prompt}\n\nContext:\n---\n{context_str}\n---\n\nQuestion: {query}"

        try:
            async with live_client.aio.live.connect(
                model=config.AUDIO_MODEL_NAME,
                config=types.LiveConnectConfig(response_modalities=["AUDIO"])
            ) as google_session:
                await google_session.send_client_content(
                    turns=[{"role": "user", "parts": [{"text": full_prompt}]}], turn_complete=True
                )

                google_turn = google_session.receive()
                async for response_chunk in google_turn:
                    if response_chunk.data:
                        await self.client_websocket.send_bytes(response_chunk.data)
                
                print("Audio stream finished.")
                await self.client_websocket.send_text(json.dumps({"type": "audio_end"}))

        except Exception as e:
            print(f"Error during audio generation: {e}")
            await self.client_websocket.send_text(json.dumps({"type": "error", "content": str(e)}))