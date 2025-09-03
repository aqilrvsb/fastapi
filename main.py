# app/main.py
import os
from typing import Optional, Dict, Any
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx

app = FastAPI(title="GPT-OSS FastAPI ↔ Ollama Gateway")

# ---- Railway vars ----
UPSTREAM = os.getenv("UPSTREAM", "gpt-oss-model.railway.internal")
UPSTREAM_PORT = os.getenv("UPSTREAM_PORT", "11434")
OLLAMA_BASE = f"http://{UPSTREAM}:{UPSTREAM_PORT}"

# Optional gateway key (leave unset to disable auth)
GATEWAY_API_KEY = os.getenv("API_KEY")

def require_auth(authorization: Optional[str]):
    if not GATEWAY_API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if authorization.split(" ", 1)[1] != GATEWAY_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

@app.get("/")
def root():
    return {"greeting": "Hello, World!", "message": "Welcome to FastAPI!"}

@app.get("/health")
async def health():
    # Ping Ollama to confirm connectivity
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{OLLAMA_BASE}/api/tags")
        return {"upstream_status": r.status_code, "upstream_ok": r.status_code == 200}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: Optional[str] = Header(None)):
    require_auth(authorization)

    payload: Dict[str, Any] = await request.json()
    # OpenAI-style fields
    model = payload.get("model", "llama3.1")
    messages = payload.get("messages", [])
    stream = bool(payload.get("stream", False))
    temperature = payload.get("temperature", 0.7)
    top_p = payload.get("top_p", 0.95)
    max_tokens = payload.get("max_tokens")  # optional

    # Convert OpenAI → Ollama
    ollama_body: Dict[str, Any] = {
        "model": model,
        "messages": messages,          # Ollama supports messages with {role, content}
        "stream": stream,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
        },
    }
    if max_tokens is not None:
        ollama_body["options"]["num_predict"] = max_tokens

    async with httpx.AsyncClient(timeout=httpx.Timeout(60, read=300)) as client:
        if not stream:
            r = await client.post(f"{OLLAMA_BASE}/api/chat", json=ollama_body)
            r.raise_for_status()
            data = r.json()
            # Convert Ollama → OpenAI-ish response
            content = "".join((m.get("content", "") for m in data.get("message", {}) and [data["message"]]))
            return JSONResponse({
                "id": "chatcmpl-ollama",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": data.get("message", {"role": "assistant", "content": content}),
                    "finish_reason": data.get("done_reason", "stop")
                }],
                "model": model
            })

        async def streamer():
            async with client.stream("POST", f"{OLLAMA_BASE}/api/chat", json=ollama_body) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    # Ollama streams JSON lines; forward as OpenAI SSE chunks
                    # Minimal transform for compatibility
                    yield f"data: {line}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(streamer(), media_type="text/event-stream")
