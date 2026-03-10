"""Main entry point for the inference server."""

import multiprocessing as mp
import signal
import threading
from pathlib import Path
from typing import Any, cast

from appinfra.app.fastapi import ServerBuilder
from appinfra.app.fastapi.runtime.server import Server
from appinfra.log import Logger
from appinfra.size import size_str
from appinfra.time import Ticker, since, start

from ..adapters import AdapterManager
from ..api.adapters import create_adapter_router
from ..api.openai.router import create_openai_router
from ..api.routes import create_health_handler, create_routes
from .config import InferenceConfig
from .errors import ExceptionHandler
from .factories import get_engine_factory, get_handler_factory
from .handler import RequestHandler
from .loop import run_engine_loop
from .progress import ProgressTracker
from .warmup import warmup_adapters, warmup_base_model

# ---------------------------------------------------------------------------
# Engine and handler creation via factories
# ---------------------------------------------------------------------------


def create_engine(lg: Logger, config: InferenceConfig) -> Any:
    """Create engine from configuration.

    Uses factory pattern to dispatch to native or vLLM engine based on
    backends.engine setting. Returns an engine implementing InferenceEngineProtocol.
    """
    if not config.models.path:
        raise ValueError(
            "models.path is required (set via config, --model-path, or MODEL_PATH)"
        )

    engine_type = config.backends.engine
    model_name = Path(config.models.path).name
    lg.info(
        "initializing engine & loading model...",
        extra={"engine": engine_type, "model": model_name},
    )
    t0 = start()

    factory = get_engine_factory(engine_type)
    on_progress = ProgressTracker(lg) if engine_type == "native" else None
    engine = factory.create(lg, config, on_progress=on_progress)

    lg.info(
        "engine initialized & model loaded",
        extra={"after": since(t0), "engine": engine_type, "model": model_name},
    )
    return engine


def _select_handler_type(config: InferenceConfig) -> str:
    """Select handler type based on engine.

    Uses primary handler for HTTP-based engines (vLLM server, Ollama),
    fallback handler for in-process engines (native, vllm).
    """
    if config.backends.engine in ("vllm-server", "ollama"):
        return config.dispatch.handler_primary
    return config.dispatch.handler_fallback


def create_handler(lg: Logger, engine: Any, config: InferenceConfig) -> Any:
    """Create a request handler for the engine using factory pattern."""
    handler_type = _select_handler_type(config)
    factory = get_handler_factory(handler_type)
    return factory.create(lg, engine, config)


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

    def __init__(self, config: InferenceConfig, lg: Logger) -> None:
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
        self._adapter_manager: AdapterManager | None = None

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
            Path(self._config.models.path).name
            if self._config.models.path
            else "unknown"
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
        Also creates PEFT engine if enabled (lazy-loaded on first use).
        """
        self._engine = create_engine(self._lg, self._config)

        self._log_gpu_stats()
        if self._config.backends.engine == "native":
            self._log_kv_cache_info()

        # Start periodic GPU stats logging
        self._start_memory_ticker()

    def _get_adapter_base_path(self) -> str | None:
        """Get adapter base path based on engine type."""
        engine_type = self._config.backends.engine
        if engine_type == "vllm":
            lora_cfg = self._config.engines.vllm.lora
            return lora_cfg.base_path if lora_cfg.enabled else None
        if engine_type == "vllm-server":
            lora_cfg = self._config.engines.vllm_server.lora
            return lora_cfg.base_path if lora_cfg.enabled else None
        if engine_type == "peft":
            return self._config.engines.peft.adapter_base_path
        return None

    def _configure_adapters(self, adapter_base_path: str) -> None:
        """Configure adapter manager for the handler."""
        assert self._handler is not None
        self._handler.set_lora_base_path(adapter_base_path)
        peft_type_filter = None
        if self._config.backends.engine == "peft":
            peft_type_filter = AdapterManager.PROMPT_LEARNING_TYPES
        elif self._config.backends.engine == "vllm-server":
            peft_type_filter = AdapterManager.LORA_TYPES
        self._adapter_manager = AdapterManager(
            self._lg,
            adapter_base_path,
            base_model_path=self._config.models.path,
            peft_type_filter=peft_type_filter,
        )
        count = self._adapter_manager.scan()
        self._handler.set_adapter_manager(self._adapter_manager)
        self._lg.info(
            "adapters enabled",
            extra={"base_path": adapter_base_path, "adapters_loaded": count},
        )

    def create_handler(self) -> None:
        """Phase 3: Create request handler."""
        handler_type = _select_handler_type(self._config)
        self._handler = create_handler(self._lg, self._engine, self._config)
        self._lg.info("handler created", extra={"type": handler_type})
        adapter_base_path = self._get_adapter_base_path()
        if adapter_base_path:
            self._configure_adapters(adapter_base_path)

    def warmup(self) -> None:
        """Phase 4: Run warmup query if configured."""
        factory = get_engine_factory(self._config.backends.engine)
        if not factory.warmup_enabled(self._config):
            return

        baseline = warmup_base_model(self._lg, self._engine)
        warmup_adapters(self._lg, self._engine, self._adapter_manager, baseline)

    def mark_ready(self) -> None:
        """Phase 5: Mark server as ready to accept requests."""
        self._ready.value = True
        self._lg.info("ready to serve requests")

    def run_loop(self) -> None:
        """Phase 6: Run main request loop until shutdown."""
        assert self._handler is not None, "create_handler() must be called first"
        run_engine_loop(
            self._lg,
            self._handler,
            self._request_q,
            self._response_q,
            self._shutdown,
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
        2. Shutdown handler (stops thread pool, fails pending requests)
        3. Shutdown engine (releases GPU, destroys process groups)
        4. Stop HTTP server
        """
        if self._memory_ticker and self._memory_ticker.is_running():
            self._memory_ticker.stop()

        if self._handler is not None and hasattr(self._handler, "shutdown"):
            self._lg.debug("shutting down handler...")
            try:
                self._handler.shutdown()
            except Exception as e:
                self._lg.warning("handler shutdown error", extra={"exception": e})

        if self._engine is not None:
            self._lg.debug("shutting down engine...")
            try:
                self._engine.shutdown()
            except Exception as e:
                self._lg.warning("engine shutdown error", extra={"exception": e})

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

    def _add_lora_routes(self, builder: Any) -> Any:
        """Add LoRA adapter routes if enabled (call in routes mode)."""
        lora_cfg = self._config.engines.vllm.lora
        if lora_cfg.enabled and lora_cfg.base_path:
            builder = builder.with_router(create_adapter_router(), prefix="/v1")
        return builder

    def _build_server_builder(self) -> Any:
        """Build initial ServerBuilder with host/port/metadata."""
        cfg = self._config.api
        return (
            ServerBuilder(self._lg, "inference")
            .with_host(cfg.host)
            .with_port(cfg.port)
            .with_title(cfg.title)
            .with_description(cfg.description)
            .with_version(cfg.version)
        )

    def _build_routes(self, model_name: str) -> Any:
        """Build routes configuration."""
        from ..api.trace import TraceMiddleware

        health_handler = create_health_handler(self._ready)
        model_config = self._config.models.get(model_name)
        routes = (
            self._build_server_builder()
            .routes.with_middleware(TraceMiddleware)
            .with_route("/health", health_handler)
            .with_router(create_routes(model_name))
            .with_router(create_openai_router(model_name, model_config), prefix="/v1")
            .with_exception_handler(Exception, ExceptionHandler(self._lg))
        )
        return self._add_lora_routes(routes)

    def _build_server(self, model_name: str) -> Server:
        """Build the HTTP server."""
        cfg = self._config.api
        routes_builder = self._build_routes(model_name)
        return cast(
            Server,
            routes_builder.done()
            .subprocess.with_ipc(self._request_q, self._response_q)
            .with_log_file(cfg.log_file)
            .with_auto_restart(enabled=True)
            .with_response_timeout(cfg.response_timeout)
            .done()
            .uvicorn.with_workers(cfg.uvicorn.workers)
            .with_timeout_keep_alive(cfg.uvicorn.timeout_keep_alive)
            .with_log_level(cfg.uvicorn.log_level)
            .with_access_log(cfg.uvicorn.access_log)
            .done()
            .build(),
        )

    def _install_signal_handlers(self) -> None:
        """Install signal handlers for graceful shutdown."""

        def handler(signum: int, frame: Any) -> None:
            self._lg.debug("received shutdown signal", extra={"signal": signum})
            self.signal_shutdown()

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _log_gpu_stats(self) -> None:
        """Log current GPU memory stats from engine."""
        stats = self._engine.memory_stats()

        # Prefer device-level stats (pynvml) for vLLM, fall back to torch stats
        if stats.get("device_used", 0) > 0:
            # Calculate KV cache used from usage percentage
            kv_total = stats.get("kv_cache_bytes", 0)
            kv_usage_perc = stats.get("kv_cache_usage_perc", 0.0)
            kv_used = int(kv_total * kv_usage_perc) if kv_total else 0

            self._lg.info(
                "GPU memory",
                extra={
                    "total": size_str(stats["device_total"]),
                    "used": size_str(stats["device_used"]),
                    "free": size_str(stats["device_free"]),
                    "model": size_str(stats.get("model_memory", 0)),
                    "kv": f"total[{size_str(kv_total)}] used[{size_str(kv_used)}]",
                },
            )
        elif stats.get("allocated", 0) > 0:
            self._lg.info(
                "GPU memory",
                extra={
                    "allocated": size_str(stats["allocated"]),
                    "reserved": size_str(stats["reserved"]),
                    "peak": size_str(stats["peak"]),
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

        def log_stats() -> None:
            self._log_gpu_stats()

        self._memory_ticker = Ticker(self._lg, log_stats, secs=120.0, initial=False)
        threading.Thread(target=self._memory_ticker.run, daemon=True).start()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_server(lg: Logger, config: InferenceConfig) -> None:
    """Run the inference server.

    Args:
        lg: Logger instance.
        config: Inference configuration (required).
    """
    boot = BootSequence(config, lg)
    boot.run()
