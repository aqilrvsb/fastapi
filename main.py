# main.py
import os
from typing import Optional, Dict, Any
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx

app = FastAPI(title="GPT-OSS FastAPI Gateway")

UPSTREAM = os.getenv("UPSTREAM", "gpt-oss-model.railway.internal")
UPSTREAM_PORT = os.getenv("UPSTREAM_PORT", "11434")
BASE = f"http://{UPSTREAM}:{UPSTREAM_PORT}"

API_KEY = os.getenv("API_KEY")  # optional

def require_auth(auth: Optional[str]):
    if not API_KEY:
        return
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(401, "Missing Authorization header")
    if auth.split(" ", 1)[1] != API_KEY:
        raise HTTPException(403, "Invalid API key")

@app.get("/")
def root():
    return {"greeting": "Hello, World!", "message": "Welcome to FastAPI!"}

@app.get("/health")
async def health():
    async with httpx.AsyncClient(timeout=10) as client:
        # Try Ollama first
        try:
            r = await client.get(f"{BASE}/api/tags")
            if r.status_code == 200:
                models = [m.get("name") for m in r.json().get("models", []) if isinstance(m, dict)]
                return {"ok": True, "backend": "ollama", "base": BASE, "models": models}
        except Exception:
            pass
        # Try OpenAI-compatible
        try:
            r = await client.get(f"{BASE}/v1/models")
            if r.status_code == 200:
                models = [m.get("id") for m in r.json().get("data", [])]
                return {"ok": True, "backend": "openai-compatible", "base": BASE, "models": models}
        except Exception:
            pass
    return {"ok": False, "base": BASE, "error": "Could not reach upstream"}

@app.post("/v1/chat/completions")
async def chat(request: Request, authorization: Optional[str] = Header(None)):
    require_auth(authorization)
    body: Dict[str, Any] = await request.json()
    model = body.get("model", "llama3.1")
    messages = body.get("messages", [])
    stream = bool(body.get("stream", False))

    # Detect backend per-request
    async with httpx.AsyncClient(timeout=httpx.Timeout(60, read=300)) as client:
        # Is it Ollama?
        try:
            r = await client.get(f"{BASE}/api/tags")
            is_ollama = (r.status_code == 200)
        except Exception:
            is_ollama = False

        if is_ollama:
            # ---- OpenAI -> Ollama translate ----
            payload = {"model": model, "messages": messages, "stream": stream}
            if not stream:
                r = await client.post(f"{BASE}/api/chat", json=payload)
                r.raise_for_status()
                data = r.json()
                content = data.get("message", {}).get("content", "")
                return JSONResponse({
                    "id": "chatcmpl-ollama",
                    "object": "chat.completion",
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": data.get("done_reason", "stop")
                    }]
                })
            async def streamer():
                async with client.stream("POST", f"{BASE}/api/chat", json=payload) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if line:
                            yield f"data: {line}\n\n"
                    yield "data: [DONE]\n\n"
            return StreamingResponse(streamer(), media_type="text/event-stream")

        # ---- OpenAI-compatible upstream: simple pass-through ----
        if not stream:
            r = await client.post(f"{BASE}/v1/chat/completions", json=body)
            return JSONResponse(status_code=r.status_code, content=r.json())

        async def streamer2():
            async with client.stream("POST", f"{BASE}/v1/chat/completions", json=body) as r:
                r.raise_for_status()
                async for chunk in r.aiter_bytes():
                    yield chunk
        return StreamingResponse(streamer2(), media_type="text/event-stream")
