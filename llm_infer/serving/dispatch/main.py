"""Main entry point for the inference server."""

import multiprocessing as mp
import signal
import threading
from pathlib import Path
from typing import Any

from appinfra.app.fastapi import ServerBuilder
from appinfra.app.fastapi.runtime.server import Server
from appinfra.time import ETA, Ticker, delta_str, since, start

from ..api.openai.router import create_openai_router
from ..api.routes import create_health_handler, create_routes
from .config import InferenceConfig
from .handler import RequestHandler
from .loop import run_engine_loop

# ---------------------------------------------------------------------------
# Progress tracking for model loading
# ---------------------------------------------------------------------------

_PHASE_LABELS = {
    "tokenizer": ("tokenizer", None),
    "weights:init": ("weights", "initialized"),
    "weights:alloc": ("weights", "allocated"),
    "weights:stream": ("weights", "loaded"),
    "kv_cache": ("kv_cache", None),
}

_PHASE_ACTIONS = {
    "weights:init": ("initializing", "initialized"),
    "weights:alloc": ("allocating", "allocated"),
    "weights:stream": ("loading", "loaded"),
}


class ProgressTracker:
    """Tracks loading progress with timing and ETA for each phase."""

    def __init__(self, lg: Any) -> None:
        self._lg = lg
        self._start_times: dict[str, float] = {}
        self._last_logged: dict[str, int] = {}
        self._etas: dict[str, ETA | None] = {}

    def __call__(self, phase: str, current: int, total: int) -> None:
        label, progress_field = _PHASE_LABELS.get(phase, (phase, None))
        action_ing, action_ed = _PHASE_ACTIONS.get(phase, ("loading", "loaded"))

        if current == 0:
            self._on_phase_start(phase, total, action_ing, label)
        elif current < total:
            self._on_phase_progress(
                phase, current, total, action_ing, label, progress_field
            )
        else:
            self._on_phase_complete(phase, action_ed, label)

    def _on_phase_start(
        self, phase: str, total: int, action_ing: str, label: str
    ) -> None:
        self._start_times[phase] = start()
        self._last_logged[phase] = 0
        self._etas[phase] = ETA(total=total) if total > 1 else None
        self._lg.debug(f"{action_ing} {label}...")

    def _build_progress_extra(
        self, phase: str, current: int, total: int, progress_field: str | None
    ) -> dict[str, Any]:
        """Build extra dict for progress logging."""
        extra: dict[str, Any] = {
            "after": since(self._start_times[phase]),
            "total": total,
            "progress": f"{(current * 100) // total}%",
        }
        if progress_field is not None:
            extra[progress_field] = current
        if eta_obj := self._etas.get(phase):
            eta_obj.update(current)
            if (remaining := eta_obj.remaining_secs()) is not None:
                extra["eta"] = delta_str(remaining)
        return extra

    def _on_phase_progress(
        self,
        phase: str,
        current: int,
        total: int,
        action_ing: str,
        label: str,
        progress_field: str | None,
    ) -> None:
        step = max(1, total // 10)
        if current - self._last_logged.get(phase, 0) < step:
            return
        self._lg.debug(
            f"{action_ing} {label}...",
            extra=self._build_progress_extra(phase, current, total, progress_field),
        )
        self._last_logged[phase] = current

    def _on_phase_complete(self, phase: str, action_ed: str, label: str) -> None:
        elapsed = since(self._start_times[phase])
        self._lg.info(f"{label} {action_ed}", extra={"after": elapsed})


# ---------------------------------------------------------------------------
# Engine creation (deferred imports to avoid loading torch at module level)
# ---------------------------------------------------------------------------


def create_engine(config: InferenceConfig, lg: Any = None) -> Any:
    """Create engine from configuration.

    Dispatches to native or vLLM engine based on backends.engine setting.
    Returns an engine implementing InferenceEngineProtocol.
    """
    if not config.model.path:
        raise ValueError(
            "model.path is required (set via config, --model-path, or MODEL_PATH)"
        )

    engine_type = config.backends.engine
    model_name = Path(config.model.path).name
    if lg:
        lg.info(
            "initializing engine & loading model...",
            extra={"engine": engine_type, "model": model_name},
        )
    t0 = start()

    if engine_type == "vllm":
        engine = _create_vllm_engine(config, lg)
    else:
        engine = _create_native_engine(config, lg)

    if lg:
        lg.info(
            "engine initialized & model loaded",
            extra={"after": since(t0), "engine": engine_type, "model": model_name},
        )

    return engine


def _create_native_engine(config: InferenceConfig, lg: Any = None) -> Any:
    """Create native inference engine."""
    from ...pipelines import EngineConfig, InferenceEngine, ModelConfig

    assert config.model.path is not None  # Validated in create_engine
    native_cfg = config.engines.native
    model_config = ModelConfig.from_hf_config(config.model.path)
    engine_config = EngineConfig(
        model=model_config,
        model_path=config.model.path,
        num_blocks=native_cfg.num_blocks,
        block_size=native_cfg.block_size,
        max_batch_size=native_cfg.max_batch_size,
        attention_backend=native_cfg.attention_backend,
        linear_backend=config.backends.linear,
        torch_compile=native_cfg.torch_compile,
        warmup=native_cfg.warmup,
    )

    on_progress = ProgressTracker(lg) if lg else None
    return InferenceEngine(lg, engine_config, on_progress=on_progress)


def _build_vllm_config(model_path: str, vllm_cfg) -> Any:
    """Build VLLMConfig from dispatch config."""
    from ...pipelines.engines.vllm_engine import VLLMConfig

    return VLLMConfig(
        model_path=model_path,
        gpu_memory_utilization=vllm_cfg.gpu_memory_utilization,
        cpu_offload_gb=vllm_cfg.cpu_offload_gb,
        swap_space=vllm_cfg.swap_space,
        max_model_len=vllm_cfg.max_model_len,
        tensor_parallel_size=vllm_cfg.tensor_parallel_size,
        pipeline_parallel_size=vllm_cfg.pipeline_parallel_size,
        max_num_seqs=vllm_cfg.max_num_seqs,
        max_num_batched_tokens=vllm_cfg.max_num_batched_tokens,
        scheduling_policy=vllm_cfg.scheduling_policy,
        enable_prefix_caching=vllm_cfg.enable_prefix_caching,
        kv_cache_dtype=vllm_cfg.kv_cache_dtype,
        enforce_eager=vllm_cfg.enforce_eager,
        disable_custom_all_reduce=vllm_cfg.disable_custom_all_reduce,
        quantization=vllm_cfg.quantization,
        speculative_model=vllm_cfg.speculative_model,
        num_speculative_tokens=vllm_cfg.num_speculative_tokens,
        dtype=vllm_cfg.dtype,
        trust_remote_code=vllm_cfg.trust_remote_code,
    )


def _create_vllm_engine(config: InferenceConfig, lg: Any = None) -> Any:
    """Create vLLM-backed inference engine."""
    try:
        from ...pipelines.engines.vllm_engine import VLLMEngine
    except ImportError as e:
        raise ImportError(
            "vLLM engine requested (backends.engine=vllm) but vLLM is not installed. "
            "Install with: pip install vllm\nOr use native engine: backends.engine=native"
        ) from e

    assert config.model.path is not None  # Validated in create_engine
    return VLLMEngine(_build_vllm_config(config.model.path, config.engines.vllm), lg)


def create_handler(engine: Any, config: InferenceConfig) -> RequestHandler:
    """Create a request handler for the engine."""
    from .handlers import BoundedQueueHandler, SequentialHandler

    handler_type = config.dispatch.handler

    if handler_type == "sequential":
        return SequentialHandler(engine)
    elif handler_type == "bounded":
        # vLLM handles batching internally, use max_batch_size=1
        max_batch_size = (
            1
            if config.backends.engine == "vllm"
            else config.engines.native.max_batch_size
        )
        return BoundedQueueHandler(
            engine,
            max_pending=config.dispatch.max_pending,
            max_batch_size=max_batch_size,
            batch_streaming=getattr(config.dispatch, "batch_streaming", False),
        )
    else:
        raise ValueError(f"Unknown handler type: {handler_type}")


# ---------------------------------------------------------------------------
# Boot sequence - orchestrates server startup phases
# ---------------------------------------------------------------------------


class BootSequence:
    """Manages server boot sequence with staged initialization.

    Phases:
        1. start_server() - Start HTTP server (health returns 'initializing')
        2. load_engine() - Load model weights (heavy, triggers torch import)
        3. create_handler() - Create request handler
        4. warmup() - Run warmup query if configured
        5. mark_ready() - Health now returns 'ok'
        6. run_loop() - Process requests until shutdown
    """

    def __init__(self, config: InferenceConfig, lg: Any) -> None:
        self._config = config
        self._lg = lg

        # IPC
        self._request_q: mp.Queue = mp.Queue()
        self._response_q: mp.Queue = mp.Queue()

        # Shared ready flag for health endpoint
        self._ready = mp.Value("b", False)

        # Components (initialized during boot phases)
        self._server: Server | None = None
        self._engine: Any = None
        self._handler: RequestHandler | None = None
        self._memory_ticker: Ticker | None = None

        # Shutdown coordination
        self._shutdown = threading.Event()

    # -----------------------------------------------------------------------
    # Boot phases
    # -----------------------------------------------------------------------

    def start_server(self) -> None:
        """Phase 1: Start HTTP server immediately.

        Health endpoint returns 'initializing' until mark_ready() is called.
        """
        model_name = (
            Path(self._config.model.path).name if self._config.model.path else "unknown"
        )

        self._server = self._build_server(model_name)
        proc = self._server.start_subprocess()

        self._lg.info(
            "server started",
            extra={
                "host": self._config.api.host,
                "port": self._config.api.port,
                "pid": proc.pid,
            },
        )

    def load_engine(self) -> None:
        """Phase 2: Load model and create engine.

        This is the heavy phase - triggers torch import and loads weights.
        """
        self._engine = create_engine(self._config, lg=self._lg)

        self._log_gpu_stats()
        if self._config.backends.engine == "native":
            self._log_kv_cache_info()

        # Start periodic GPU stats logging
        self._start_memory_ticker()

    def create_handler(self) -> None:
        """Phase 3: Create request handler."""
        self._handler = create_handler(self._engine, self._config)
        self._handler.set_logger(self._lg)
        self._lg.info("handler created", extra={"type": self._config.dispatch.handler})

    def warmup(self) -> None:
        """Phase 4: Run warmup query if configured."""
        cfg = self._config
        should_warmup = (
            cfg.backends.engine == "native" and cfg.engines.native.warmup
        ) or (cfg.backends.engine == "vllm" and cfg.engines.vllm.warmup)

        if should_warmup:
            self._lg.debug("running warmup query...")
            t0 = start()
            output = self._engine.generate("Say hello", max_tokens=8)
            self._lg.info(
                "warmup complete",
                extra={"after": since(t0), "tokens": len(output.split())},
            )

    def mark_ready(self) -> None:
        """Phase 5: Mark server as ready to accept requests."""
        self._ready.value = True
        self._lg.info("ready to serve requests")

    def run_loop(self) -> None:
        """Phase 6: Run main request loop until shutdown."""
        assert self._handler is not None, "create_handler() must be called first"
        run_engine_loop(
            self._handler,
            self._request_q,
            self._response_q,
            self._shutdown,
            lg=self._lg,
        )

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def run(self) -> None:
        """Execute full boot sequence and run server."""
        self._install_signal_handlers()

        try:
            self.start_server()
            self.load_engine()
            self.create_handler()
            self.warmup()
            self.mark_ready()
            self.run_loop()
        except KeyboardInterrupt:
            self._lg.info("interrupted")
        except Exception as e:
            self._lg.error("server error", extra={"exception": e})
            raise
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Clean shutdown of all components.

        Shutdown order (reverse of boot):
        1. Stop memory ticker
        2. Shutdown engine (releases GPU, destroys process groups)
        3. Stop HTTP server
        """
        if self._memory_ticker and self._memory_ticker.is_running():
            self._memory_ticker.stop()

        if self._engine is not None:
            self._lg.debug("shutting down engine...")
            try:
                self._engine.shutdown()
            except Exception as e:
                self._lg.warning("engine shutdown error", extra={"error": str(e)})

        if self._server and self._server.is_running:
            self._lg.debug("stopping server...")
            self._server.stop()
            self._lg.info("server stopped")

    def signal_shutdown(self) -> None:
        """Signal the main loop to stop."""
        self._shutdown.set()

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _build_server(self, model_name: str) -> Server:
        """Build the HTTP server."""
        cfg = self._config
        health_handler = create_health_handler(self._ready)

        return (
            ServerBuilder("inference")
            .with_host(cfg.api.host)
            .with_port(cfg.api.port)
            .with_title(cfg.api.title)
            .with_description(cfg.api.description)
            .with_version(cfg.api.version)
            .routes.with_route("/health", health_handler)
            .with_router(create_routes(model_name))
            .with_router(create_openai_router(model_name), prefix="/v1")
            .done()
            .subprocess.with_ipc(self._request_q, self._response_q)
            .with_log_file(cfg.api.log_file)
            .with_auto_restart(enabled=True)
            .with_response_timeout(cfg.api.response_timeout)
            .done()
            .uvicorn.with_workers(cfg.api.uvicorn.workers)
            .with_timeout_keep_alive(cfg.api.uvicorn.timeout_keep_alive)
            .with_log_level(cfg.api.uvicorn.log_level)
            .with_access_log(cfg.api.uvicorn.access_log)
            .done()
            .build()
        )

    def _install_signal_handlers(self) -> None:
        """Install signal handlers for graceful shutdown."""

        def handler(signum: int, frame: Any) -> None:
            self._lg.debug("received shutdown signal", extra={"signal": signum})
            self.signal_shutdown()

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _log_gpu_stats(self) -> None:
        """Log current GPU memory stats."""
        import torch

        if not torch.cuda.is_available():
            return

        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        peak = torch.cuda.max_memory_allocated() / (1024**3)

        self._lg.info(
            "GPU stats",
            extra={
                "allocated_gb": f"{allocated:.2f}",
                "reserved_gb": f"{reserved:.2f}",
                "peak_gb": f"{peak:.2f}",
            },
        )

    def _log_kv_cache_info(self) -> None:
        """Log KV cache allocation info (native engine only)."""
        cfg = self._config.engines.native
        kv_mem_gb = self._engine.block_pool.memory_usage_bytes() / (1024**3)
        kv_capacity = cfg.num_blocks * cfg.block_size

        self._lg.info(
            "KV cache allocated",
            extra={
                "size_gb": f"{kv_mem_gb:.2f}",
                "capacity_tokens": kv_capacity,
                "num_blocks": cfg.num_blocks,
                "block_size": cfg.block_size,
            },
        )

    def _start_memory_ticker(self) -> None:
        """Start periodic GPU memory logging."""
        import torch

        def log_stats() -> None:
            if not torch.cuda.is_available():
                return
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            self._lg.info(
                "GPU stats",
                extra={
                    "allocated_gb": f"{allocated:.2f}",
                    "reserved_gb": f"{reserved:.2f}",
                    "peak_gb": f"{peak:.2f}",
                },
            )

        self._memory_ticker = Ticker(self._lg, log_stats, secs=120.0, initial=False)
        threading.Thread(target=self._memory_ticker.run, daemon=True).start()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_server(lg: Any, config: InferenceConfig) -> None:
    """Run the inference server.

    Args:
        lg: Logger instance.
        config: Inference configuration (required).
    """
    boot = BootSequence(config, lg)
    boot.run()
