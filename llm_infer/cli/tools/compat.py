"""Compat tool - generate and validate engine compatibility spec."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from appinfra.app.tools import Tool, ToolConfig
from appinfra.log import Logger

from ...compat import check_spec_accuracy, generate_compat_spec, get_spec_header


class CompatTool(Tool):
    """Generate and validate engine compatibility specification."""

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(
            name="compat",
            aliases=["c"],
            help_text="Engine compatibility specification",
        )
        super().__init__(parent, config)

    def _get_template_path(self) -> Path | None:
        """Get template path from app config (respects --etc-dir)."""
        if not self.app.config:
            return None
        compat_cfg = self.app.config.get("compat", {})
        template = compat_cfg.get("template") if compat_cfg else None
        return Path(template) if template else None

    def _add_generate_args(self, subparsers: Any) -> None:
        """Add generate subcommand arguments."""
        gen = subparsers.add_parser(
            "generate", aliases=["gen"], help="Generate compatibility spec"
        )
        gen.add_argument(
            "-o",
            "--output",
            metavar="FILE",
            help="Write spec to file (default: stdout)",
        )
        gen.add_argument(
            "--format",
            "-f",
            choices=["yaml", "json"],
            default="yaml",
            help="Output format (default: yaml)",
        )

    def _add_check_args(self, subparsers: Any) -> None:
        """Add check subcommand arguments."""
        check = subparsers.add_parser(
            "check", help="Verify spec file matches implementation"
        )
        check.add_argument(
            "file",
            nargs="?",
            metavar="FILE",
            help="Spec file to check (default: built-in template)",
        )

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        subparsers = parser.add_subparsers(dest="command", help="Commands")
        self._add_generate_args(subparsers)
        self._add_check_args(subparsers)

    @property
    def _lg(self) -> Logger:
        """Get logger with type narrowing (always set after setup)."""
        assert self.lg is not None
        return self.lg

    def run(self, **kwargs: Any) -> int:
        if self.args.command in ("generate", "gen"):
            return self._generate()
        elif self.args.command == "check":
            return self._check()
        else:
            # No subcommand - show help
            self._lg.error("no command specified, use 'generate' or 'check'")
            return 1

    def _serialize_spec(self, spec: dict) -> str:
        """Serialize spec to JSON or YAML."""
        if self.args.format == "json":
            return json.dumps(spec, indent=2)
        template_path = self._get_template_path()
        return get_spec_header(template_path) + str(
            yaml.dump(spec, default_flow_style=False, sort_keys=False)
        )

    def _generate(self) -> int:
        """Generate compatibility spec."""
        template_path = self._get_template_path()
        try:
            spec = generate_compat_spec(self._lg, template_path)
        except FileNotFoundError:
            self._lg.error("template file not found: compat_template.yaml")
            return 1
        except yaml.YAMLError as e:
            self._lg.error(f"template parse error: {e}")
            return 1

        output = self._serialize_spec(spec)
        if self.args.output:
            try:
                with open(self.args.output, "w") as f:
                    f.write(output)
                self._lg.info(f"spec written to {self.args.output}")
            except OSError as e:
                self._lg.error(f"failed to write file: {e}")
                return 1
        else:
            sys.stdout.write(output)
        return 0

    def _check(self) -> int:
        """Verify spec file matches implementation."""
        spec_file = getattr(self.args, "file", None)
        template_path = self._get_template_path()
        is_valid, issues = check_spec_accuracy(self._lg, spec_file, template_path)

        if is_valid:
            target = spec_file or "template"
            self._lg.info(f"spec {target} is accurate")
            return 0
        else:
            self._lg.error("spec has discrepancies")
            for issue in issues:
                self._lg.warning(f"  - {issue}")
            return 1
