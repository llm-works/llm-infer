"""Serve tool - starts the inference server."""

import argparse
from pathlib import Path

from appinfra.app.tools import Tool, ToolConfig
from appinfra.yaml import load as yaml_load


class ServeTool(Tool):
    """Start the inference server."""

    def __init__(self, parent=None):
        config = ToolConfig(name="serve", help_text="Start the inference server")
        super().__init__(parent, config)

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--config",
            type=Path,
            help="Path to config file (default: etc/inference.yaml)",
        )
        parser.add_argument("--host", help="Host to bind to")
        parser.add_argument("--port", type=int, help="Port to bind to")
        parser.add_argument(
            "--models-dir", type=Path, help="Directory containing models"
        )
        parser.add_argument(
            "--model",
            help="Model name (subdirectory in models-dir)",
        )
        parser.add_argument(
            "--model-path",
            type=Path,
            help="Direct path to model weights (alternative to --model)",
        )
        parser.add_argument(
            "--handler", choices=["sequential", "bounded"], help="Request handler type"
        )

    def _load_yaml_config(self) -> dict:
        """Load raw config dict from YAML file."""
        config_path = getattr(self.args, "config", None)
        if config_path is None:
            candidates = [
                Path("etc/inference.yaml"),
                Path(__file__).parent.parent.parent.parent / "etc" / "inference.yaml",
            ]
            for candidate in candidates:
                if candidate.exists():
                    config_path = candidate
                    break

        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                result: dict = yaml_load(
                    f, current_file=config_path, project_root=config_path.parent.parent
                )
                return result
        return {}

    def _get_models_dir(self) -> Path:
        """Get models directory from args or config."""
        if self.args.models_dir:
            return Path(self.args.models_dir)
        raw_config = self._load_yaml_config()
        location: str = raw_config.get("models", {}).get("location", ".models")
        return Path(location)

    def _configure_logging(self, raw_config: dict) -> None:
        """Configure third-party logging before slow imports."""
        from ...logging_setup import configure_third_party_logging

        log_cfg = raw_config.get("third_party_logging", {})
        configure_third_party_logging(
            torch_level=log_cfg.get("torch", "warning"),
            transformers_level=log_cfg.get("transformers", "error"),
        )

    def _get_model_name_early(self, raw_config: dict) -> str | None:
        """Get model name from CLI args, selection file, or config default."""
        if self.args.model:
            return str(self.args.model)
        if self.args.model_path:
            return str(self.args.model_path.name)

        # Check selection file
        selection: dict = raw_config.get("models", {}).get("selection", {})
        if selection.get("path"):
            sel_name, sel_path = self._load_selection_file(selection["path"])
            if sel_path:
                return sel_path.name
            if sel_name:
                return sel_name

        default: str | None = selection.get("default")
        return default

    def _get_server_config(self, raw_config: dict) -> tuple[str, int, str]:
        """Get host, port, handler from config with CLI overrides."""
        api_cfg = raw_config.get("api", {})
        dispatch_cfg = raw_config.get("dispatch", {})
        host = self.args.host or api_cfg.get("host", "0.0.0.0")
        port = self.args.port or api_cfg.get("port", 8000)
        handler = self.args.handler or dispatch_cfg.get("handler", "bounded")
        return host, port, handler

    def _import_server_deps(self):
        """Import server dependencies."""
        from ...serving.dispatch import InferenceConfig, run_server

        return InferenceConfig, run_server

    def run(self, **kwargs) -> int:
        raw_config = self._load_yaml_config()
        self._configure_logging(raw_config)

        model_name = self._get_model_name_early(raw_config)
        host, port, handler = self._get_server_config(raw_config)
        self.lg.info(
            "starting server...",
            extra={"model": model_name, "host": host, "port": port, "handler": handler},
        )

        InferenceConfig, run_server = self._import_server_deps()  # noqa: N806
        config = InferenceConfig.load(self.args.config)
        config.apply_env_overrides()
        model_path = self._resolve_model_path(config)
        if model_path is None:
            return 1

        config.apply_cli_overrides(
            host=self.args.host,
            port=self.args.port,
            handler=self.args.handler,
            model_path=str(model_path),
        )
        run_server(self.lg, config)
        return 0

    def _load_selection_file(self, path: str | Path) -> tuple[str | None, Path | None]:
        """Load model selection from external file.

        Args:
            path: Path to selection YAML file.

        Returns:
            (model_name, model_path) - at most one will be set.
        """
        import yaml

        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            if data is None:
                return None, None
            model_name = data.get("name")
            model_path = data.get("path")
            return model_name, Path(model_path) if model_path else None
        except FileNotFoundError:
            self.lg.debug("selection file not found", extra={"path": str(path)})
            return None, None
        except Exception as e:
            self.lg.warning(
                "failed to load selection file",
                extra={"path": str(path), "error": str(e)},
            )
            return None, None

    def _find_model(self, name: str, models_dir: Path) -> Path | None:
        """Find model by name in models directory.

        Args:
            name: Model name (subdirectory name).
            models_dir: Directory containing model subdirectories.

        Returns:
            Path to model directory if found, None otherwise.
        """
        model_path = models_dir / name
        if model_path.is_dir() and (model_path / "config.json").exists():
            return model_path
        return None

    def _resolve_from_cli(self, models_dir: Path) -> Path | None:
        """Resolve model from CLI arguments (--model-path or --model)."""
        if self.args.model_path:
            model_path_arg: Path = self.args.model_path
            if not model_path_arg.exists():
                self.lg.error(
                    "model path does not exist", extra={"path": str(model_path_arg)}
                )
                return None
            return model_path_arg

        if self.args.model:
            path = self._find_model(self.args.model, models_dir)
            if path:
                return path
            self.lg.error(
                "model not found",
                extra={"model": self.args.model, "dir": str(models_dir)},
            )
        return None

    def _resolve_from_selection_file(
        self, selection, models_dir: Path
    ) -> tuple[Path | None, bool]:
        """Resolve model from selection file. Returns (path, was_attempted)."""
        if not selection.path:
            return None, False

        sel_name, sel_path = self._load_selection_file(selection.path)
        if sel_path:
            self.lg.debug(
                "using selection file path",
                extra={"path": str(sel_path), "file": selection.path},
            )
            if not sel_path.exists():
                self.lg.error(
                    "selection model_path does not exist", extra={"path": str(sel_path)}
                )
                return None, True
            return sel_path, True
        if sel_name:
            self.lg.debug(
                "using selection file name",
                extra={"name": sel_name, "file": selection.path},
            )
            path = self._find_model(sel_name, models_dir)
            if path:
                return path, True
            self.lg.error(
                "selection model not found",
                extra={"model": sel_name, "dir": str(models_dir)},
            )
            return None, True
        return None, False

    def _resolve_model_path(self, config) -> Path | None:
        """Resolve model path from CLI, selection file, or config default."""
        models_dir = self._get_models_dir()

        # 1-2. CLI arguments
        if self.args.model_path or self.args.model:
            return self._resolve_from_cli(models_dir)

        # 3. Selection file
        selection = config.model.selection
        path, attempted = self._resolve_from_selection_file(selection, models_dir)
        if attempted:
            return path

        # 4. Selection default
        if selection.default:
            path = self._find_model(selection.default, models_dir)
            if path:
                return path
            self.lg.error(
                "default model not found",
                extra={"model": selection.default, "dir": str(models_dir)},
            )
            return None

        self.lg.error("no model specified, use --model or configure selection")
        return None
