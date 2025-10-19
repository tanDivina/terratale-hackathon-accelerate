from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import asyncio
import io
import os
import json
from dotenv import load_dotenv

load_dotenv()

from . import ai_core
from . import qa_system
from . import image_search

import google.generativeai as genai

# --- App Setup ---
app = FastAPI(title="Dual Response AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    # Load and index documents on startup
    # qa_system.load_and_index_docs() # Disabled for now
    # image_search.index_images() # Disabled for now
    pass

# --- API Models ---
class SynthesizeRequest(BaseModel):
    text: str

class QARequest(BaseModel):
    question: str

class ImageSearchRequest(BaseModel):
    query: str

# --- Endpoints ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Websocket connected")
    try:
        while True:
            query = await websocket.receive_text()
            print(f"Received query: {query}")
            
            # The text and audio generation are now two separate, parallel tasks
            async def text_task():
                text_response = await ai_core.generate_text_response(query, ai_core.search_knowledge_base(query))
                await websocket.send_text(json.dumps({"type": "text", "content": text_response}))

            audio_session = ai_core.MateoAudioSession(websocket)
            await asyncio.gather(
                text_task(),
                audio_session.generate_and_stream_audio(query, ai_core.search_knowledge_base(query))
            )

    except WebSocketDisconnect:
        print("Client disconnected.")

@app.post("/synthesize")
async def synthesize_text(request: SynthesizeRequest):
    try:
        audio_bytes = ai_core.synthesize_papito_speech(request.text)
        return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/qa")
async def qa_endpoint(request: QARequest):
    try:
        qa_chain = qa_system.create_qa_chain()
        answer = qa_chain.invoke(request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/image-search")
async def image_search_endpoint(request: ImageSearchRequest):
    try:
        results = image_search.search_images(request.query)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Static Files ---
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")