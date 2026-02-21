#!/usr/bin/env python3
"""Mock server that returns transient errors to test retry logic.

Run the server:
    python test_retry_server.py

Then in another terminal:
    python test_retry_client.py
"""

from fastapi import FastAPI, Response

app = FastAPI()
call_count = 0


@app.post("/v1/chat/completions")
def chat(response: Response) -> dict:
    """Return 429 for first 2 calls, then success."""
    global call_count
    call_count += 1
    print(f"Request #{call_count}")

    if call_count <= 2:
        print("  -> Returning 429 (rate limited)")
        response.status_code = 429
        return {"error": {"message": "Rate limited", "type": "rate_limit_error"}}

    print("  -> Returning 200 (success)")
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Success after retry!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@app.post("/v1/reset")
def reset() -> dict:
    """Reset the call counter."""
    global call_count
    call_count = 0
    print("Counter reset")
    return {"status": "reset"}


if __name__ == "__main__":
    import uvicorn

    print("Starting mock server on http://localhost:8111")
    print("First 2 requests will return 429, then success")
    print("POST /v1/reset to reset the counter")
    uvicorn.run(app, host="0.0.0.0", port=8111)
