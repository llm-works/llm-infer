# Known Issues

Operational and runtime issues with workarounds. For resolved bugs, see `bugs-journal.md`.

---

## Qwen 3.5: `think: true` doesn't separate reasoning from content

**Status:** Open (vLLM limitation)
**Affected:** Qwen 3.5 models via vllm-server engine
**Since:** 2026-04-05 (vLLM 0.19.0)

### Problem

When `think: true` is requested, the model does produce reasoning (396 tokens vs 32 without
thinking), but vLLM returns `reasoning_content: ""` — the reasoning is not extracted into a
separate field. It either stays in `content` or is consumed internally.

### Root Cause

vLLM's `--reasoning-parser qwen3` only works when `enable_thinking: true` is set as the
**server-level default** via `--default-chat-template-kwargs`. When the server default is
`enable_thinking: false` and a per-request `chat_template_kwargs: {"enable_thinking": true}`
override is sent, the chat template enables thinking (model produces reasoning tokens), but the
reasoning parser doesn't extract them.

Related: https://github.com/vllm-project/vllm/issues/38894

### Impact

- **`think: false` (default):** Works correctly. No thinking, clean responses.
- **`think: true`:** Model thinks (uses more tokens) but the `thinking` field in the response is
  empty. The answer in `content` is still correct.

Prod workloads (tool calling, summarization, ratings) do not use `think: true`, so no prod impact.

### Workaround

None for per-request toggle. If thinking mode is needed for all requests, start the vLLM server
with `enable_thinking: true` as the default:

```yaml
# models.yaml — override for always-think mode
vllm:
  chat_template_kwargs:
    enable_thinking: true
```

This makes all requests think by default and the reasoning parser correctly separates content.
The downside is non-thinking requests also incur the thinking token overhead.

### Resolution Path

Wait for vLLM to fix per-request `chat_template_kwargs` interaction with the reasoning parser.
Once fixed, the existing llm-infer implementation should work as-is since the per-request
plumbing is already in place.
