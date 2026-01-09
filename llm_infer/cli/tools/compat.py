"""Compat tool - generate and validate engine compatibility spec."""

from __future__ import annotations

import argparse
import json
import sys

import yaml
from appinfra.app.tools import Tool, ToolConfig

from ...compat import check_spec_accuracy, generate_compat_spec, get_spec_header


class CompatTool(Tool):
    """Generate and validate engine compatibility specification."""

    def __init__(self, parent=None):
        config = ToolConfig(
            name="compat",
            aliases=["c"],
            help_text="Engine compatibility specification",
        )
        super().__init__(parent, config)

    def _add_generate_args(self, subparsers) -> None:
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

    def _add_check_args(self, subparsers) -> None:
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

    def run(self, **kwargs) -> int:
        if self.args.command in ("generate", "gen"):
            return self._generate()
        elif self.args.command == "check":
            return self._check()
        else:
            # No subcommand - show help
            self.lg.error("no command specified, use 'generate' or 'check'")
            return 1

    def _serialize_spec(self, spec: dict) -> str:
        """Serialize spec to JSON or YAML."""
        if self.args.format == "json":
            return json.dumps(spec, indent=2)
        return get_spec_header() + str(
            yaml.dump(spec, default_flow_style=False, sort_keys=False)
        )

    def _generate(self) -> int:
        """Generate compatibility spec."""
        try:
            spec = generate_compat_spec(self.lg)
        except FileNotFoundError:
            self.lg.error("template file not found: compat_template.yaml")
            return 1
        except yaml.YAMLError as e:
            self.lg.error(f"template parse error: {e}")
            return 1

        output = self._serialize_spec(spec)
        if self.args.output:
            try:
                with open(self.args.output, "w") as f:
                    f.write(output)
                self.lg.info(f"spec written to {self.args.output}")
            except OSError as e:
                self.lg.error(f"failed to write file: {e}")
                return 1
        else:
            sys.stdout.write(output)
        return 0

    def _check(self) -> int:
        """Verify spec file matches implementation."""
        spec_file = getattr(self.args, "file", None)
        is_valid, issues = check_spec_accuracy(self.lg, spec_file)

        if is_valid:
            target = spec_file or "template"
            self.lg.info(f"spec {target} is accurate")
            return 0
        else:
            self.lg.error("spec has discrepancies")
            for issue in issues:
                self.lg.warning(f"  - {issue}")
            return 1
