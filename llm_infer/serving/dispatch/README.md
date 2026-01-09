# Dispatch Layer

Request dispatch and handler strategies for LLM inference.

## Architecture

```
inference/
├── api/              # HTTP layer (FastAPI, uvicorn subprocess)
├── dispatch/         # Request handlers, queueing, engine loop  <-- you are here
└── engine/           # Inference (model, KV cache, generation)
```

```
Main Process                          Uvicorn Subprocess
┌────────────────────────┐            ┌────────────────────────┐
│                        │            │                        │
│  InferenceEngine       │            │  FastAPI (api/)        │
│       │                │            │       │                │
│       ▼                │            │       ▼                │
│  RequestHandler        │◄──────────▶│  request_q / response_q│
│  (seq/bounded/batch)   │  mp.Queue  │                        │
│       │                │            │  POST /generate        │
│       ▼                │            │  GET /health           │
│  run_engine_loop       │            │                        │
│                        │            │  uvicorn (isolated)    │
└────────────────────────┘            └────────────────────────┘
```

## Files

```
inference/dispatch/
├── __init__.py          # Public exports
├── types.py             # Internal types (Request, Response, RequestStatus)
├── handler.py           # RequestHandler ABC
├── handlers/
│   ├── __init__.py
│   ├── sequential.py    # SequentialHandler (1 at a time)
│   ├── bounded.py       # BoundedQueueHandler (reject when full)
│   └── batching.py      # ContinuousBatchingHandler (stub)
├── loop.py              # run_engine_loop
└── main.py              # Entry point, CLI
```

## Usage

```bash
# Set required environment variable
export MODEL_PATH=/path/to/model

# Run with default settings (bounded handler, port 8000)
python -m inference.dispatch.main

# Run with options
python -m inference.dispatch.main --handler sequential --port 8080 --verbose

# With uvicorn log isolation
python -m inference.dispatch.main --log-file /var/log/uvicorn.log
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | (required) | Path to HuggingFace model |
| `NUM_BLOCKS` | 1024 | Number of KV cache blocks |
| `BLOCK_SIZE` | 16 | Tokens per KV cache block |
| `MAX_BATCH_SIZE` | 1 | Maximum batch size for engine |
| `MAX_PENDING` | 10 | Maximum pending requests (bounded handler) |

### CLI Options

```
--host          Host to bind to (default: 0.0.0.0)
--port          Port to bind to (default: 8000)
--handler       Handler type: sequential, bounded (default: bounded)
--log-file      Log file for uvicorn output (isolates uvicorn logs)
--verbose, -v   Verbose logging
```

## Request Handlers

Three execution strategies with a common interface:

| Handler | Description | Use Case |
|---------|-------------|----------|
| `SequentialHandler` | One request at a time, never rejects | Debugging, single-user |
| `BoundedQueueHandler` | Max N pending, rejects beyond | Production with latency SLOs |
| `ContinuousBatchingHandler` | Batch into single forward pass | High-throughput (stub) |

### Handler Interface

```python
from inference.dispatch import RequestHandler, Request, Response

class RequestHandler(ABC):
    def submit(self, request: Request) -> bool:
        """Submit request. Returns False if rejected."""
        ...

    def step(self) -> list[Response]:
        """Process one step, return completed responses."""
        ...

    @property
    def pending_count(self) -> int:
        """Number of pending requests."""
        ...

    @property
    def is_saturated(self) -> bool:
        """True if at capacity."""
        ...
```

## Internal Types

The dispatch layer uses internal types for queue messages (separate from HTTP schemas):

```python
from inference.dispatch import Request, Response, RequestStatus

# Internal request (queue message)
request = Request(
    id="uuid",
    prompt="Hello",
    max_tokens=100,
    temperature=1.0,
)

# Internal response (queue message)
response = Response(
    id="uuid",
    status=RequestStatus.COMPLETED,
    result="Generated text",
    prompt_tokens=4,
    completion_tokens=25,
)
```

## Future Work

1. **ContinuousBatchingHandler** - Full implementation with `engine.step_batch()`
2. **Streaming** - Token-by-token response streaming
3. **Metrics** - Queue depth, latency histograms, rejection rate
4. **Multi-GPU** - Multiple pipelines with routing
