#!/usr/bin/env python3
"""Test client that demonstrates retry with backoff.

First, start the mock server:
    python test_retry_server.py

Then run this client:
    python test_retry_client.py
"""

import httpx
from appinfra.log import Logger

from llm_infer.client import Factory

# Reset the server counter
httpx.post("http://localhost:8111/v1/reset")

lg = Logger("retry-test")
factory = Factory(lg)

# Create router with retry enabled
router = factory.from_config(
    {
        "retry": {
            "enabled": True,
            "timeout": 30,  # Give up after 30 seconds total
            "backoff": {
                "base": 0.5,  # Start with 0.5s delay
                "max": 10.0,  # Max delay between retries
            },
        },
        "backends": {
            "mock": {
                "type": "openai_compatible",
                "base_url": "http://localhost:8111/v1",
            }
        },
    },
    discover_models=False,
)

# Verify retry is configured
client = router.get_client()
print(f"Retry configured: {client._backoff is not None}")
print("Sending request (server will return 429 twice, then success)...")
print()

try:
    response = router.chat([{"role": "user", "content": "Hello!"}])
    print(f"Response: {response.content}")
except Exception as e:
    print(f"Error: {e}")
finally:
    router.close()
