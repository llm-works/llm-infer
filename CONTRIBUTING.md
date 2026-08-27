# Contributing Guide

## Development Setup

```bash
git clone https://github.com/llm-works/llm-infer.git
cd llm-infer

# Install in development mode
pip install -e ".[dev,runtime]"

# Run checks
make check
```

## Code Quality

```bash
make fmt          # Format code with ruff
make lint         # Run linter
make typecheck    # Type check with mypy
make check        # Run all checks (fmt, lint, typecheck, tests)
```

## Test Suite

```bash
make test.unit    # Unit tests
make test.all     # All tests with coverage
```

## Sign-off (DCO)

We use the [Developer Certificate of Origin (DCO)](https://developercertificate.org). Every commit
must be signed off, asserting you have the right to submit it under the project's Apache-2.0
license.

Add sign-off automatically:

```bash
git commit -s -m "your commit message"
```

This appends `Signed-off-by: Your Name <your@email>` to the commit message. Forgot? Amend:

```bash
git commit --amend -s
```

## Pull Request Guidelines

1. Run `make check` and ensure all checks pass
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure every commit has `Signed-off-by:` (DCO check enforces this on all PRs)

All PRs are squash-merged to keep git history clean.

## Release Process

```bash
# 1. Create release PR to develop, squash-merge

# 2. Merge to main
git checkout main && git pull origin main
git merge origin/develop --no-ff -m "Release vX.Y.Z"
git push origin main

# 3. Tag (triggers PyPI publish)
git tag vX.Y.Z && git push origin vX.Y.Z

# 4. Sync develop
git checkout develop && git merge main && git push origin develop
```

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
