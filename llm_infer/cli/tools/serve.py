"""Serve tool - starts the inference server."""

import argparse
from pathlib import Path
from typing import Any

from appinfra.app.tools import Tool, ToolConfig

from ...models import ModelResolver
from ...serving.dispatch.config_overrides import parse_override_args


class ServeTool(Tool):
    """Start the inference server."""

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(name="serve", help_text="Start the inference server")
        super().__init__(parent, config)

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        self._add_model_args(parser)
        self._add_server_args(parser)

    def _add_model_args(self, parser: argparse.ArgumentParser) -> None:
        """Add model-related arguments."""
        parser.add_argument(
            "--list-models",
            action="store_true",
            help="List available models from config and exit",
        )
        parser.add_argument(
            "--models-dir", type=Path, help="Directory containing models"
        )
        parser.add_argument("--model", help="Model name (subdirectory in models-dir)")
        parser.add_argument(
            "--model-path",
            type=Path,
            help="Direct path to model weights (alternative to --model)",
        )
        parser.add_argument(
            "--embed",
            action="store_true",
            help="Use default embedding model (from selection.embed)",
        )

    def _add_server_args(self, parser: argparse.ArgumentParser) -> None:
        """Add server-related arguments."""
        parser.add_argument("--host", help="Host to bind to")
        parser.add_argument("--port", type=int, help="Port to bind to")
        parser.add_argument(
            "--handler", choices=["sequential", "bounded"], help="Request handler type"
        )
        parser.add_argument(
            "--engine",
            choices=["native", "vllm", "ollama"],
            help="Inference engine backend",
        )
        parser.add_argument(
            "-o",
            "--override",
            action="append",
            metavar="KEY=VALUE",
            dest="overrides",
            help="Override config value (e.g. -o engines.vllm.gpu_memory_utilization=0.1)",
        )

    def _get_raw_config(self) -> dict:
        """Get raw config dict from app (loaded by appinfra, respects --etc-dir)."""
        return dict(self.app.config) if self.app.config else {}

    def _get_model_locations(self) -> list[Path]:
        """Get model search locations from args or config.

        Returns list of directories to search for models, in priority order.
        """
        if self.args.models_dir:
            return [Path(self.args.models_dir)]
        raw_config = self._get_raw_config()
        models_cfg = raw_config.get("models", {})
        # Support both 'locations' (list) and legacy 'location' (single)
        locations = models_cfg.get("locations", [])
        if not locations:
            location = models_cfg.get("location", ".models")
            locations = [location]
        return [Path(loc) for loc in locations]

    def _configure_logging(self, raw_config: dict) -> None:
        """Configure third-party logging before slow imports."""
        from ...logging_setup import configure_third_party_logging

        log_cfg = raw_config.get("third_party_logging", {})
        configure_third_party_logging(
            torch_level=log_cfg.get("torch", "warning"),
            transformers_level=log_cfg.get("transformers", "error"),
        )

    def _get_model_name_early(self, raw_config: dict) -> str | None:
        """Get model name from CLI args, selection file, or config default.

        Uses lightweight resolver for selection file parsing.
        """
        if self.args.model:
            return str(self.args.model)
        if self.args.model_path:
            return str(self.args.model_path.name)

        # Get task-specific selection config
        task = "embed" if self.args.embed else "generate"
        selection_all: dict = raw_config.get("models", {}).get("selection", {})
        selection: dict = selection_all.get(task, {})

        # Use resolver for selection file parsing (lightweight, no heavy imports)
        if selection.get("path"):
            resolver = ModelResolver(
                lg=self.lg, locations=[]
            )  # Locations not needed for selection file
            sel_name, sel_path = resolver.load_selection_file(selection["path"])
            if sel_path:
                return sel_path.name
            if sel_name:
                return sel_name

        return selection.get("default")

    def _get_server_config(self, raw_config: dict) -> tuple[str, int, str]:
        """Get host, port, handler from config with CLI overrides."""
        api_cfg = raw_config.get("api", {})
        dispatch_cfg = raw_config.get("dispatch", {})
        host = self.args.host or api_cfg.get("host", "0.0.0.0")
        port = self.args.port or api_cfg.get("port", 8000)
        handler = self.args.handler or dispatch_cfg.get("handler", "bounded")
        return host, port, handler

    def _import_server_deps(self) -> tuple[Any, Any]:
        """Import server dependencies."""
        from ...serving.dispatch import InferenceConfig, run_server

        return InferenceConfig, run_server

    def _apply_model_overrides(self, config: Any, model_name: str) -> None:
        """Apply model-specific settings from unified config.

        Applies in order: task, max_model_len, then vllm overrides.
        Model-level vllm overrides allow models to specify required vLLM settings
        (e.g., embedding models need enable_prefix_caching=false).
        """
        model_cfg = config.models.get(model_name)
        if model_cfg is None:
            return
        if model_cfg.task:
            config.engines.vllm.task = model_cfg.task
            config.engines.ollama.task = model_cfg.task
            self.lg.debug("model override", extra={"task": model_cfg.task})
        if model_cfg._max_model_len_set:
            config.engines.vllm.max_model_len = model_cfg.max_model_len
            self.lg.debug(
                "model override", extra={"max_model_len": model_cfg.max_model_len}
            )
        # Apply model-specific vLLM overrides
        for key, value in model_cfg.vllm.items():
            if hasattr(config.engines.vllm, key):
                setattr(config.engines.vllm, key, value)
                self.lg.debug("model vllm override", extra={key: value})
            else:
                self.lg.warning(
                    "unknown vllm config key in model override",
                    extra={"key": key, "model": model_name},
                )

    def _print_model_group(
        self, header: str, models: list[str], default: str | None
    ) -> None:
        """Print a group of models with header and default marker."""
        if not models:
            return
        print(f"  {header}:")
        for name in sorted(models):
            marker = " (default)" if name == default else ""
            print(f"    - {name}{marker}")
        print()

    def _list_models(self) -> int:
        """List available models from config."""
        from ...models.config import ModelsConfig

        models_cfg = ModelsConfig.from_raw_config(self._get_raw_config())

        # Group by task
        generate_models = []
        embed_models = []
        for name, cfg in models_cfg.models.items():
            if cfg.task == "embed":
                embed_models.append(name)
            else:
                generate_models.append(name)

        print("Configured models:\n")
        gen_default = models_cfg.get_selection("generate").default
        embed_default = models_cfg.get_selection("embed").default
        self._print_model_group("Generation", generate_models, gen_default)
        self._print_model_group("Embedding", embed_models, embed_default)
        return 0

    def _apply_cli_overrides(self, config: Any, model_path: Path) -> None:
        """Apply overrides in precedence order: model < env < cli.

        Note: apply_cli_overrides internally applies env first, then cli.
        """
        self._apply_model_overrides(config, model_path.name)
        config.apply_cli_overrides(
            host=self.args.host,
            port=self.args.port,
            handler=self.args.handler,
            model_path=str(model_path),
            engine=self.args.engine,
            overrides=self._parse_overrides(),
        )

    def run(self, **kwargs: Any) -> int:
        if self.args.list_models:
            return self._list_models()

        raw_config = self._get_raw_config()
        self._configure_logging(raw_config)

        model_name = self._get_model_name_early(raw_config)
        host, port, handler = self._get_server_config(raw_config)
        self.lg.info(
            "starting server...",
            extra={"model": model_name, "host": host, "port": port, "handler": handler},
        )

        InferenceConfig, run_server = self._import_server_deps()  # noqa: N806
        config = InferenceConfig.from_dict(raw_config)
        model_path = self._resolve_model_path(config)
        if model_path is None:
            return 1

        self._apply_cli_overrides(config, model_path)
        run_server(self.lg, config)
        return 0

    def _parse_overrides(self) -> dict[str, str] | None:
        """Parse -o key=value arguments into a dict."""
        return parse_override_args(self.args.overrides)

    def _resolve_model_path(self, config: Any) -> Path | None:
        """Resolve model path from CLI, selection file, or config default.

        Uses ModelResolver for unified resolution logic.
        For Ollama engine, returns synthetic path from model name (no local validation).
        """
        # Determine effective engine (CLI override > config)
        engine = self.args.engine or config.backends.engine

        # For Ollama, we just need the model name - Ollama manages its own files
        if engine == "ollama":
            return self._resolve_ollama_model_name(config)

        locations = self._get_model_locations()
        resolver = ModelResolver(lg=self.lg, locations=locations)

        # Get selection config based on task type (--embed flag)
        task = "embed" if self.args.embed else "generate"
        selection = config.models.get_selection(task)

        return resolver.resolve(
            model_path=self.args.model_path,
            model_name=self.args.model,
            selection=selection,
        )

    def _resolve_ollama_model_name(self, config: Any) -> Path | None:
        """Resolve model name for Ollama (no local file validation).

        For Ollama, we return a synthetic Path from the model name.
        The OllamaEngineFactory will look up the 'ollama' field in models.yaml.
        """
        model_name = self.args.model
        if not model_name:
            # Try selection file/default
            task = "embed" if self.args.embed else "generate"
            selection = config.models.get_selection(task)
            if selection.default:
                model_name = selection.default

        if not model_name:
            self.lg.error("no model specified for Ollama engine")
            return None

        # Return synthetic path - just the model name
        # OllamaEngineFactory extracts .name and looks up ollama field
        return Path(model_name)
