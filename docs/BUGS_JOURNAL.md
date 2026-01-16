# Inference Implementation Bugs - December 2025

This document captures bugs found while debugging inference output that didn't match HuggingFace's
reference implementation.

## Symptom

Model outputs garbage/repetitive text instead of coherent responses. For example:
- Input: "The capital of France is"
- Expected: "Paris"
- Got: "nednednedned" or "of" or other nonsense

## Root Causes Found

### 1. Attention Bias Mismatch

**File:** `llm_infer/pipelines/model/config.py`, `llm_infer/pipelines/model/transformer.py`

**Problem:** Q/K/V projection layers were initialized with `bias=True`, but many models (LLaMA,
TinyLlama, Mistral) have `attention_bias: false` in their config. The random bias values were never
overwritten during weight loading, corrupting the forward pass.

**Fix:**
```python
# config.py - Load attention_bias from HF config
model_type = hf_config.get("model_type", "")
if "attention_bias" in hf_config:
    attention_bias = hf_config["attention_bias"]
elif model_type in ("qwen2", "qwen"):
    attention_bias = True  # Qwen2 hardcodes bias=True
else:
    attention_bias = False  # LLaMA-style default

# transformer.py - Use config value
nn.Linear(..., bias=cfg.attention_bias)
```

**Insight:** HuggingFace's Qwen2 implementation hardcodes `bias=True` without a config option.
Always check the actual model class source, not just the config schema.


### 2. RoPE Implementation Mismatch

**File:** `llm_infer/pipelines/model/attention.py`

**Problem:** Two common RoPE implementations exist:
1. **Interleaved:** Split by even/odd indices `q[..., ::2], q[..., 1::2]`
2. **Half-split:** Split first/second half `q[..., :half], q[..., half:]`

LLaMA and most HF models use half-split. Our implementation used interleaved.

**Fix:**
```python
def rotate_half(x):
    """LLaMA-style rotation."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin, positions):
    # Expand cos/sin to full head_dim
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    # Apply rotation
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out
```

**Insight:** These produce mathematically different rotations. The pattern determines how position
information is encoded. Always verify against the reference implementation.


### 3. KV Cache Position Bug (allocate_for_prompt timing)

**File:** `llm_infer/pipelines/generation.py`

**Problem:** During prefill, `allocate_for_prompt()` was called AFTER the forward pass. This meant
`kv_cache.num_tokens` was 0 during the forward pass, causing `update_kv_cache()` to write to
position 0 and `paged_attention()` to read with `kv_len=0`.

The bug was subtle: the model still produced output, but the KV cache positions were wrong,
causing the decode phase to attend to incorrectly positioned key/value pairs.

**Symptom:** Model produces coherent first token (from correct prefill attention) but garbage in
subsequent decode steps (from corrupted KV cache).

**Fix:**
```python
def run_prefill(request, model, block_pool, device):
    with torch.inference_mode():
        prompt_len = len(request.prompt_tokens)
        # MUST allocate BEFORE forward pass so num_tokens is set correctly
        request.kv_cache.allocate_for_prompt(block_pool, prompt_len)

        token_ids = torch.tensor([request.prompt_tokens], device=device)
        positions = torch.arange(prompt_len, device=device).unsqueeze(0)

        logits = model.forward(token_ids, positions, [request.kv_cache], block_pool)
        # ... rest of function
```

**Insight:** KV cache metadata (num_tokens, block allocation) must be set up BEFORE the forward
pass that writes to the cache. The forward pass uses this metadata to determine write positions.


### 4. Missing Weight Tying (tie_word_embeddings)

**File:** `llm_infer/pipelines/model/config.py`, `llm_infer/pipelines/model/transformer.py`

**Problem:** Many models (Qwen2, LLaMA-3, etc.) have `tie_word_embeddings: true` in their config.
This means `lm_head.weight` should be the SAME tensor as `embed_tokens.weight`. HuggingFace doesn't
save `lm_head.weight` separately for such models.

Our implementation:
1. Initialized `lm_head` with random weights in `_init_layers()`
2. No `lm_head.weight` existed in safetensors file (due to weight tying)
3. Random weights were never overwritten → completely wrong logits

**Symptom:** Hidden states match HuggingFace almost exactly, but final logits are completely wrong.
This is particularly confusing because layer-by-layer debugging shows everything is correct until
the very last step.

**Debugging trace:**
```
Layer 27 output: matches HF (max_diff ~0.5)
After final norm: matches HF (max_diff ~0.25)
Logits: WRONG (max_diff ~10+, top predictions completely different)
```

The divergence at `lm_head` despite correct hidden states was the key clue.

**Fix:**
```python
# config.py - Add field
@dataclass
class ModelConfig:
    # ...
    tie_word_embeddings: bool = False

    @classmethod
    def from_hf_config(cls, model_path):
        # ...
        return cls(
            # ...
            tie_word_embeddings=hf_config.get("tie_word_embeddings", False),
        )

# transformer.py - Tie weights after loading
def _load_weights(self, weights_path):
    # ... load weights from safetensors ...

    # Handle weight tying: lm_head shares weights with embed_tokens
    if self.config.tie_word_embeddings:
        self.lm_head.weight = self.embed_tokens.weight
```

**Verification:**
```python
# After fix, weights should be same tensor
assert model.lm_head.weight.data_ptr() == model.embed_tokens.weight.data_ptr()
```

**Insight:** When logits diverge but hidden states match, check the `lm_head` layer. Weight tying
is a common optimization that reduces model size but requires explicit handling during loading.


### 5. Missing Causal Mask in Prefill

**File:** `llm_infer/pipelines/model/attention.py`

**Problem:** During prefill (processing multiple prompt tokens at once), each query position should
only attend to previous positions. The naive paged attention implementation had no causal mask,
allowing positions to attend to future tokens.

**Fix:**
```python
def naive_paged_attention(...):
    ...
    # Apply causal mask for prefill (seq_len > 1)
    if seq_len > 1:
        query_positions = torch.arange(seq_len, device=device)
        kv_positions = torch.arange(kv_len, device=device)
        offset = kv_len - seq_len
        mask = kv_positions.unsqueeze(0) > (query_positions.unsqueeze(1) + offset)
        scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))
    ...
```

**Insight:** Decode (seq_len=1) doesn't need a mask since we naturally attend to all previous
tokens. Prefill requires explicit masking.


## Debugging Methodology

### Layer-by-Layer Comparison

The most effective debugging approach was comparing hidden states layer by layer:

```python
# Get HF hidden states
hf_out = hf_model(input_ids, output_hidden_states=True)
hf_hidden_states = hf_out.hidden_states

# Compare after each layer
for layer_idx in range(num_layers):
    our_hidden = run_our_layer(...)
    hf_hidden = hf_hidden_states[layer_idx + 1]
    max_diff = abs(our_hidden - hf_hidden).max()
    print(f'Layer {layer_idx}: max_diff={max_diff:.4f}')
```

This quickly revealed where divergence started (layer 0's attention output).


### Weight Comparison

Verify weights are loaded correctly:

```python
our_q = our_model.layers[0]['q_proj'].weight
hf_q = hf_model.model.layers[0].self_attn.q_proj.weight
print(f'Match: {torch.allclose(our_q, hf_q, atol=1e-3)}')
```


### Check Bias Presence

```python
print(f'Our q_proj has bias: {layer.q_proj.bias is not None}')
print(f'HF q_proj has bias: {hf_layer.self_attn.q_proj.bias is not None}')
```


## Key Learnings

### 1. Reference Implementation is Truth

Always compare against HuggingFace's implementation step-by-step. Don't assume the paper or docs
describe the actual implementation.

### 2. Small Errors Compound

A small error in layer 0 (max_diff=0.12) exploded to max_diff=140 by layer 2. Floating point
errors compound exponentially through layers.

### 3. Check Model-Specific Defaults

Different model families have different conventions:
- LLaMA: `attention_bias=False`, half-split RoPE
- Qwen2: `attention_bias=True` (hardcoded), half-split RoPE
- Some models: interleaved RoPE

### 4. Prefill vs Decode Differences

Prefill and decode have different requirements:
- Prefill: Multiple tokens, needs causal mask, positions are 0..n-1
- Decode: Single token, no mask needed, position is n

### 5. Sampling vs Greedy

Default `temperature=1.0` does random sampling (`torch.multinomial`). For deterministic output
matching HF's `generate(..., do_sample=False)`, use `temperature=0` (argmax).

### 6. Check Weight Tying

If hidden states match but logits don't, check `lm_head`. Many models tie `lm_head.weight` to
`embed_tokens.weight` to save parameters. HuggingFace doesn't save `lm_head.weight` separately
for such models, so you must explicitly handle this:

```python
if config.tie_word_embeddings:
    model.lm_head.weight = model.embed_tokens.weight
```

### 7. Initialization Order Matters

When using paged KV cache, the cache metadata (num_tokens, block allocation) must be set up
BEFORE the forward pass. The forward pass reads this metadata to determine where to write K/V
values. Calling `allocate_for_prompt()` after forward means K/V gets written to wrong positions.


## Files Changed

| File | Change |
|------|--------|
| `llm_infer/pipelines/model/config.py` | Added `attention_bias`, `tie_word_embeddings` fields |
| `llm_infer/pipelines/model/transformer.py` | Use `cfg.attention_bias` for Q/K/V bias; tie `lm_head` to `embed_tokens` |
| `llm_infer/pipelines/model/attention.py` | Fixed RoPE to half-split, added causal mask in prefill |
| `llm_infer/pipelines/generation.py` | Call `allocate_for_prompt()` BEFORE forward pass |


## Testing Verification

After fixes, verify with:

```python
# Should match exactly with temperature=0
our_token = our_model(...).argmax()
hf_token = hf_model.generate(..., max_new_tokens=1, do_sample=False)[0, -1]
assert our_token == hf_token
```

Test multiple prompts and multi-token generation to ensure decode loop is also correct.


### 6. Missing Chat Template for Instruct Models

**File:** `llm_infer/primitives/tokenizer/huggingface.py`, `llm_infer/pipelines/core.py`

**Problem:** Instruct-tuned models (e.g., `qwen2.5-1.5b-instruct`) expect prompts formatted with a
chat template that wraps user messages in special tokens. Without this, the model receives raw text
and doesn't know it should respond as an assistant.

**Symptom:** Model outputs are incoherent, repetitive, or continue the prompt instead of responding:
```
Input: "Kannst du deutsch?"
Got: "Ja, ich kann Deutsch. Wie kann ich dir helfen? Ja, ich kann Deutsch. Ich kann auf Deutsch:
sprechen, schreiben und ein wenig rechnen. Entschuldigung, aber ich bin ein Großvater..."
```

The model is completing text rather than responding because it doesn't see the assistant turn
marker.

**Fix:**
```python
# tokenizer/base.py - Add abstract methods
@property
@abstractmethod
def has_chat_template(self) -> bool:
    """Whether this tokenizer has a chat template."""
    ...

@abstractmethod
def encode_chat(self, message: str, add_generation_prompt: bool = True) -> list[int]:
    """Encode a user message using the chat template."""
    ...

# tokenizer/huggingface.py - Implement
@property
def has_chat_template(self) -> bool:
    return self._hf.chat_template is not None

def encode_chat(self, message: str, add_generation_prompt: bool = True) -> list[int]:
    messages = [{"role": "user", "content": message}]
    return self._hf.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
    )

# engine/core.py - Use chat template when available
def _create_request(self, prompt, ...):
    if self.tokenizer.has_chat_template:
        tokens = self.tokenizer.encode_chat(prompt)
    else:
        tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
    ...
```

**What the chat template does:** For Qwen2.5-Instruct, it transforms:
```
"Kannst du deutsch?"
```
Into:
```
<|im_start|>user
Kannst du deutsch?<|im_end|>
<|im_start|>assistant
```

This tells the model "the user said X, now respond as assistant."

**Insight:** Always check if a model has a chat template (`tokenizer.chat_template`). Base models
use raw encoding; instruct models need the chat template applied.


### 7. Missing Repetition Penalty

**File:** `llm_infer/pipelines/sampler.py`

**Problem:** Without repetition penalty, models tend to repeat tokens or phrases, especially for
longer generations. This is particularly noticeable with smaller models.

**Symptom:** Output contains repeated phrases:
```
"Ja, ich kann Deutsch. Wie kann ich dir helfen? Ja, ich kann Deutsch. Ich kann auf Deutsch..."
```

**Fix:**
```python
# sampler.py - Add repetition penalty
def _apply_repetition_penalty(
    logits: Tensor, past_tokens: list[int], penalty: float
) -> Tensor:
    """Apply repetition penalty to previously seen tokens.

    For tokens that appeared before:
    - If logit > 0: divide by penalty (reduce probability)
    - If logit < 0: multiply by penalty (reduce probability further)
    """
    logits = logits.clone()
    for token_id in set(past_tokens):
        if logits[0, token_id] > 0:
            logits[0, token_id] /= penalty
        else:
            logits[0, token_id] *= penalty
    return logits

def sample(logits, ..., repetition_penalty=1.0, past_tokens=None):
    if repetition_penalty != 1.0 and past_tokens:
        logits = _apply_repetition_penalty(logits, past_tokens, repetition_penalty)
    # ... rest of sampling
```

**Wiring:** The parameter flows through the entire stack:
- `scheduler.py`: Request dataclass
- `generation.py`: Passed to sample() with prompt + output tokens
- `core.py`: generate() accepts the parameter
- `dispatch/types.py`: Internal Request type
- `dispatch/handlers/`: All handlers pass it through
- `api/schemas.py`: HTTP request schema (default: 1.1)
- `cli/tools/query.py`: `--repetition-penalty/-r` flag

**Recommended values:**
- `1.0`: Disabled (no penalty)
- `1.1`: Light penalty (good default for most models)
- `1.2`: Stronger penalty (for very repetitive outputs)

**Insight:** Repetition penalty should be applied BEFORE temperature scaling, to the raw logits.
Include both prompt tokens and generated tokens in the penalty calculation.


## Files Changed (Chat Template & Repetition Penalty)

| File | Change |
|------|--------|
| `llm_infer/primitives/tokenizer/base.py` | Added `has_chat_template`, `encode_chat()` abstract methods |
| `llm_infer/primitives/tokenizer/huggingface.py` | Implemented chat template methods |
| `llm_infer/pipelines/core.py` | Auto-detect and use chat template; added `repetition_penalty` param |
| `llm_infer/pipelines/sampler.py` | Added `_apply_repetition_penalty()`, new params to `sample()` |
| `llm_infer/pipelines/generation.py` | Pass repetition penalty and past tokens to sampler |
| `llm_infer/pipelines/scheduler.py` | Added `repetition_penalty` to Request |
| `llm_infer/serving/dispatch/types.py` | Added `repetition_penalty` to Request |
| `llm_infer/serving/dispatch/handlers/bounded.py` | Pass repetition penalty to engine |
| `llm_infer/serving/dispatch/handlers/sequential.py` | Pass repetition penalty to engine |
| `llm_infer/serving/api/schemas.py` | Added `repetition_penalty` field (default: 1.1) |
| `llm_infer/cli/tools/query.py` | Added `--repetition-penalty/-r` argument |


## Verification

After applying both fixes:

```bash
# Before (raw encoding, no repetition penalty)
$ ./inference.py query "Kannst du deutsch?"
Ja, ich kann Deutsch. Wie kann ich dir helfen? Ja, ich kann Deutsch. Ich kann auf Deutsch:
sprechen, schreiben und ein wenig rechnen. Entschuldigung, aber ich bin ein Großvater...

# After (chat template + repetition penalty 1.1)
$ ./inference.py query "Kannst du deutsch?"
Ja, ich kenne Deutsch und kann darüber kommunizieren. Bitte zögern Sie nicht, mich zu fragen,
falls Sie Fragen zu deutscher Sprache haben. Ich stehe Ihnen gerne zur Verfügung!
```


### 8. Chat Template Applied to Base Models

**File:** `llm_infer/pipelines/engine.py`

**Problem:** Base models (e.g., `qwen2.5-0.5b`, `qwen2.5-1.5b`) ship with chat template files in
their tokenizer directories, so `tokenizer.chat_template is not None` returns `True`. However, base
models were NOT trained with chat templates - they're for text completion, not conversation.

The previous fix (Bug #6) auto-applied chat templates when available. This worked for instruct
models but broke base models because:
1. `has_chat_template` returned `True` (file exists)
2. Engine applied chat formatting, wrapping prompts like:
   ```
   <|im_start|>system
   You are a helpful assistant.<|im_end|>
   <|im_start|>user
   The capital of France is<|im_end|>
   <|im_start|>assistant
   ```
3. Base model doesn't understand these special tokens → garbage output

**Symptom:**
```
Input: "The capital of France is"
Expected: "Paris. It is the largest city..."
Got: "salute salute salute salute salute..." (or other nonsense)
```

The key confusion: model forward pass and generation logic were CORRECT (verified by comparing
with HuggingFace). The bug was purely in the tokenization/encoding step.

**Debugging approach:**
1. Created `debug_compare.py` - compared model logits with HuggingFace → **matched exactly**
2. Created `debug_generation.py` - compared step-by-step generation → **worked correctly**
3. Created `debug_api.py` - tested what the engine does → **found the bug**

The comparison tests used raw tokenizer encoding, while the engine used `encode_chat()`.

**Fix:** Auto-detect based on model name, not just chat template presence:

```python
# engine.py
def _should_use_chat_template(self) -> bool:
    """Auto-detect if chat template should be used based on model name.

    Returns True if:
    - Model has a chat template AND
    - Model name suggests it's an instruct/chat model

    Base models often ship with chat templates but shouldn't use them.
    """
    if not self.tokenizer.has_chat_template:
        return False

    model_name = self.model_name.lower()
    return "instruct" in model_name or "chat" in model_name

def generate(self, prompt, ..., use_chat_template: bool | None = None):
    if use_chat_template is None:
        use_chat_template = self._should_use_chat_template()
    # ...

def _create_request(self, prompt, ..., use_chat_template: bool):
    if use_chat_template and self.tokenizer.has_chat_template:
        tokens = self.tokenizer.encode_chat(prompt)
    else:
        tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
```

**API changes:** Added `use_chat_template` parameter throughout the stack:
- `engine.generate()` and `engine.generate_stream()` - accepts `bool | None`
- `api/schemas.py` - `GenerateRequest.use_chat_template` field
- `dispatch/types.py` - `Request.use_chat_template` field
- `dispatch/handlers/*.py` - pass parameter to engine

**Behavior:**
- `use_chat_template=None` (default): Auto-detect from model name
- `use_chat_template=True`: Force chat template (for instruct models with unusual names)
- `use_chat_template=False`: Force raw encoding (for base models or completion tasks)

**Verification:**
```
# Base model (qwen2.5-0.5b) - auto-detects False
Model name: qwen2.5-0.5b
Auto-detect use_chat_template: False
Generated: Paris. It is the largest city in Europe...

# Instruct model (qwen2.5-1.5b-instruct) - auto-detects True
Model name: qwen2.5-1.5b-instruct
Auto-detect use_chat_template: True
Generated: The capital of France is Paris.
```

**Insight:** The presence of a chat template file doesn't mean the model should use it. HuggingFace
base models often include chat templates for convenience (so you can use the same tokenizer config
with an instruct variant), but base models weren't trained to understand chat tokens.

**Key learning:** When debugging garbage output:
1. First verify the model forward pass matches HuggingFace (logits comparison)
2. If forward pass is correct, the bug is likely in tokenization/encoding
3. Check what tokens are actually being fed to the model


### 9. OpenAI Chat API Double "user" Prefix

**File:** `llm_infer/serving/api/openai/mappers.py`

**Problem:** When using the OpenAI-compatible `/v1/chat/completions` endpoint, the model received a
double "user" prefix in prompts, causing garbage output during MMLU benchmarks.

The flow was:
1. Client sends: `messages=[{"role": "user", "content": "Question: What is 2+2?..."}]`
2. `format_messages_as_prompt()` converts to: `"user: Question: What is 2+2?..."`
3. `use_chat_template=True` passes this to `tokenizer.encode_chat()`
4. `encode_chat()` wraps as: `[{"role": "user", "content": "user: Question..."}]`
5. `apply_chat_template()` produces: `<|im_start|>user\nuser: Question...<|im_end|>`

The model sees "user" twice, which confuses it and produces garbage like `"AsyncCallback"` repeated
or other nonsense instead of answering the question.

**Symptom:**
```
MMLU Benchmark Results:
- Expected accuracy: ~60% (reported for qwen2.5-1.5b)
- Actual accuracy: 15% (worse than random chance!)
- Output: "AsyncCallback AsyncCallback AsyncCallback..." or similar garbage
```

**Root Cause:** Two issues compounded:
1. `format_messages_as_prompt()` added role prefix (`"user: "`) but `encode_chat()` already handles
   roles via `apply_chat_template()`
2. `use_chat_template=True` was hardcoded, bypassing auto-detection that would disable chat template
   for base models

**Fix:**
```python
# mappers.py
def chat_request_to_internal(body: ChatCompletionRequest, request_id: str) -> InternalRequest:
    # For single user message, pass content directly - the tokenizer's encode_chat
    # will wrap it with proper chat template (adding role markers).
    # For multi-turn or system messages, fall back to simple format.
    if len(body.messages) == 1 and body.messages[0].role == Role.USER:
        prompt = body.messages[0].content or ""
    else:
        # Multi-turn conversations - use simple format as fallback
        prompt = format_messages_as_prompt(body.messages)

    return InternalRequest(
        ...
        use_chat_template=None,  # Let engine auto-detect based on model type
        ...
    )
```

**Key changes:**
1. Pass raw message content for single user messages (no role prefix)
2. Use `use_chat_template=None` to let engine auto-detect based on model name

**Verification:**
```bash
# Before fix
curl .../v1/chat/completions -d '{"messages":[{"role":"user","content":"2+2=?"}]}'
# Response: "AsyncCallback AsyncCallback..."

# After fix
curl .../v1/chat/completions -d '{"messages":[{"role":"user","content":"2+2=?"}]}'
# Response: "4" or "B" (correct answer)
```

**Insight:** When building an OpenAI-compatible API layer on top of a custom inference engine, be
careful about the boundary between "API formatting" and "model formatting". The chat template is
the model's responsibility, not the API layer's. The API should pass structured data, not
pre-formatted strings.


---

## Performance Optimizations

### 1. Tensor Pre-allocation for Decode Step

**Date:** 2025-12-08

**Files:** `llm_infer/pipelines/generation.py`, `llm_infer/pipelines/engine.py`

**Problem:** During the decode phase, every token generation created two new tensors:
```python
def run_decode(...):
    with torch.inference_mode():
        token_ids = torch.tensor([[request.output_tokens[-1]]], device=device)  # ~1.7ms
        positions = torch.tensor([[request.kv_cache.num_tokens - 1]], device=device)  # ~1.6ms
```

This added ~3.3ms of CPU overhead per token just for Python/PyTorch dispatch, even though:
- Tensor shapes are always (1, 1) during decode
- CUDA operations are async, so CPU becomes the bottleneck
- At 55 tok/s, this wastes ~180ms of CPU time per second of generation

**Symptom:** 100% CPU usage during inference even with GPU at 90%, variable throughput.

**Fix:** Pre-allocate tensors once in `InferenceEngine.__init__()` and reuse via in-place
assignment:

```python
# engine.py - Pre-allocate once
class InferenceEngine:
    def __init__(self, ...):
        ...
        self._decode_buffers = {
            "token_ids": torch.zeros((1, 1), dtype=torch.long, device=self.device),
            "positions": torch.zeros((1, 1), dtype=torch.long, device=self.device),
        }

# generation.py - Reuse buffers
def run_decode(..., buffers: dict[str, Tensor] | None = None):
    with torch.inference_mode():
        if buffers:
            buffers["token_ids"][0, 0] = request.output_tokens[-1]
            buffers["positions"][0, 0] = request.kv_cache.num_tokens - 1
            token_ids = buffers["token_ids"]
            positions = buffers["positions"]
        else:
            # Fallback for backwards compatibility
            token_ids = torch.tensor([[...]], device=device)
            positions = torch.tensor([[...]], device=device)
```

**Changes:**
- `generation.py`: Added optional `buffers` param to `run_decode()`
- `engine.py`: Added `_decode_buffers` initialization, passed to all `run_decode()` calls
- `StreamingResult`: Updated to accept and use decode buffers

**Expected impact:**
- Per-token CPU overhead: ~3.3ms → <0.1ms
- CPU usage during decode should drop significantly
- Backwards compatible (calls without buffers work as before)

**Insight:** PyTorch tensor creation has significant Python-side overhead even for small tensors.
For hot loops where tensor shapes are fixed, pre-allocate and reuse via in-place operations. The
GPU work is async anyway, so CPU overhead directly impacts throughput.


### 10. Double KV Cache Allocation

**Date:** 2025-12-09

**File:** `llm_infer/pipelines/generation.py`

**Problem:** The KV cache was being allocated twice for each prefill operation:
1. `engine.py` called `request.kv_cache.allocate_for_prompt()` before calling `run_prefill()`
2. `run_prefill()` in `generation.py` called `allocate_for_prompt()` again internally

The second allocation appended more blocks to the cache, corrupting the block indices. When
`update_kv_cache()` wrote K/V values, they went to the wrong positions. Subsequent decode steps
read garbage from incorrect cache locations.

**Symptom:** Model produces incoherent/garbage output that looks like random text:
```
Input: "What is 2+2?"
Got: "Ifyouknowtheansweremailittousandwewillletyouknow!"
```

**Debugging approach:**
1. Tested direct model forward pass → matched HuggingFace exactly
2. Tested `run_prefill()` + `run_decode()` directly → worked correctly
3. Tested `engine.generate()` → garbage output
4. Found: `engine.generate()` allocates, then calls `run_prefill()` which allocates again

**Fix:** Remove the duplicate allocation from `run_prefill()`:

```python
# Before (buggy)
def run_prefill(request, model, block_pool, device, guards=()):
    with torch.inference_mode():
        prompt_len = len(request.prompt_tokens)
        request.kv_cache.allocate_for_prompt(block_pool, prompt_len)  # DUPLICATE!
        token_ids = torch.tensor([request.prompt_tokens], device=device)
        ...

# After (fixed)
def run_prefill(request, model, block_pool, device, guards=()):
    with torch.inference_mode():
        # Note: KV cache must be allocated by caller before calling run_prefill.
        # The caller (InferenceEngine.generate) calls request.kv_cache.allocate_for_prompt()
        # which sets num_tokens so update_kv_cache() writes to correct positions.
        prompt_len = len(request.prompt_tokens)
        token_ids = torch.tensor([request.prompt_tokens], device=device)
        ...
```

**All callers that allocate:**
- `engine.py:generate()` - lines 149-151
- `engine.py:generate_stream()` - lines 197-199
- `engine.py:StreamingResult.__init__()` - lines 114-116

**Insight:** When a function requires setup state (like KV cache allocation), either:
1. The caller handles it (and document this), OR
2. The function handles it (and caller must NOT)

Never both. The comment in the fix documents the contract clearly.


### 11. CLI Streaming Used Wrong Endpoint

**Date:** 2025-12-09

**File:** `llm_infer/cli/tools/query.py`

**Problem:** The CLI's `query` command uses streaming by default. The streaming implementation
(`_stream_request()`) called `/v1/completions` endpoint, which sets `use_chat_template=False`
per OpenAI API spec (raw completion, no chat formatting).

For instruct/chat models like TinyLlama-Chat or Qwen2.5-Instruct, this produced garbage output
because chat models expect prompts wrapped in chat templates with role markers.

**Flow:**
```
CLI streaming → /v1/completions → use_chat_template=False → raw prompt → garbage
CLI --no-stream → /generate → use_chat_template=auto → chat template applied → correct
```

**Symptom:**
```bash
# Streaming (broken)
$ ./inference.py query "What is 2+2?"
AsyncCallbackAsyncCallbackAsyncCallback...

# Non-streaming (worked)
$ ./inference.py query --no-stream "What is 2+2?"
2+2 equals 4.
```

**Fix:** Change CLI streaming to use `/v1/chat/completions` instead:

```python
# Before
def _stream_request(self, prompt: str) -> int:
    url = f"http://{self.args.host}:{self.args.port}/v1/completions"
    payload = {
        "model": "default",
        "prompt": prompt,  # Raw prompt
        ...
    }
    ...
    text = chunk.get("choices", [{}])[0].get("text", "")  # Completion format

# After
def _stream_request(self, prompt: str) -> int:
    # Use chat completions endpoint - properly applies chat template for instruct models
    url = f"http://{self.args.host}:{self.args.port}/v1/chat/completions"
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],  # Chat format
        ...
    }
    ...
    # Chat completions uses delta.content instead of text
    delta = chunk.get("choices", [{}])[0].get("delta", {})
    text = delta.get("content", "")
```

**Key changes:**
1. Endpoint: `/v1/completions` → `/v1/chat/completions`
2. Payload: `prompt` string → `messages` array
3. Response parsing: `choices[0].text` → `choices[0].delta.content`

**OpenAI API semantics preserved:**
- `/v1/completions` - Raw text completion, no chat template (correct behavior per spec)
- `/v1/chat/completions` - Chat conversation with template (what CLI should use)

**Verification:**
```bash
$ ./inference.py query "What is 2+2?"
Yes, 2+2=4.

$ ./inference.py query --no-stream "What is 2+2?"
In math, 2 + 2 = 4...
```

**Insight:** When building CLI tools that wrap APIs, choose the endpoint that matches user
expectations. CLI users expect to type a question and get a coherent response, which requires
chat template handling. The raw `/v1/completions` endpoint is for programmatic use where callers
explicitly want raw completion behavior.


### 12. Batched Streaming Support

**Date:** 2025-12-09

**Files:** `llm_infer/serving/dispatch/handlers/bounded.py`, `etc/llm-infer.yaml`

**Problem:** Streaming requests ran sequentially, blocking other requests. When multiple agents hit
the server concurrently, each streaming request had to complete before the next could start. With
20 tokens at ~30ms/token, that's ~600ms blocking per request.

Non-streaming requests already batched correctly via `step_decode()`, but streaming requests were
handled in a separate single-request path.

**Symptom:** With 4 concurrent streaming requests:
- Sequential: ~2400ms wall time (600ms × 4)
- Expected batched: ~600ms wall time

**Fix:** Added `batch_streaming` config option to allow streaming requests to join the batched
decode loop:

```yaml
# etc/llm-infer.yaml
dispatch:
  handler: bounded
  max_pending: 10
  batch_streaming: true  # Allow streaming requests to batch with others
```

**Implementation changes in `bounded.py`:**

1. Track last streamed token index in `RunningRequest`:
```python
@dataclass
class RunningRequest:
    request: Request
    engine_request: "EngineRequest"
    output_tokens: list[int] = field(default_factory=list)
    last_streamed_idx: int = 0  # Track tokens already sent to stream
```

2. Allow streaming requests into batch when `batch_streaming=True`:
```python
if req.stream and not self.batch_streaming:
    # Old behavior: process streaming sequentially
    ...
# With batch_streaming=True, streaming requests join the batch
```

3. After each `step_decode()`, push new tokens to response queue:
```python
if self._response_q is not None:
    for req_id, running in self.running.items():
        if running.request.stream:
            new_tokens = running.engine_request.output_tokens[running.last_streamed_idx:]
            for token_id in new_tokens:
                token_text = self.engine.tokenizer.decode([token_id], skip_special_tokens=True)
                chunk = StreamChunk(id=req_id, token=token_text)
                self._response_q.put(chunk)
            running.last_streamed_idx = len(running.engine_request.output_tokens)
```

4. Send final chunk with metadata when stream completes:
```python
def _send_final_stream_chunk(self, running: RunningRequest) -> None:
    final_chunk = StreamChunk(
        id=running.request.id,
        token="",
        is_final=True,
        finish_reason=...,
        prompt_tokens=...,
        completion_tokens=...,
    )
    self._response_q.put(final_chunk)
```

**Verification:**
```python
# 4 parallel streaming requests
Wall time: 2420ms
Sum of individual: 6132ms
Speedup: 2.5x  # Batched!
```

**Insight:** The batched decode architecture was already correct - it just needed streaming
requests allowed into the batch. After each GPU forward pass, iterate through running requests
and push any new tokens to their respective streams.


### 13. FP8 Weight Dequantization Formula Inverted

**Date:** 2025-12-13

**File:** `llm_infer/pipelines/model/layers.py`

**Problem:** The FP8 quantization format stores weights with block-wise scaling. The scale tensor is
named `weight_scale_inv` in the safetensors file, indicating it's the **inverse scale** (i.e., the
value you multiply by to dequantize, not divide by).

The initial implementation incorrectly interpreted this as "the scale to invert":
```python
# WRONG: Taking reciprocal of already-inverted scale
scales = (1.0 / self.weight_scale_inv).view(out_blocks, 1, in_blocks, 1)
```

This caused the dequantized weights to overflow:
- FP8 weights range: [-448, 448]
- `weight_scale_inv` values: ~0.0003 (small)
- Wrong: `448 / 0.0003 = 1,493,333` (overflow → inf/nan)
- Correct: `448 * 0.0003 = 0.13` (reasonable weight value)

**Symptom:** Model produces garbage output (unrelated text) or nan/inf errors:
```
Input: "What is 2+2?"
Got: "Also, could you check if the following are correct: A) Number of moles..."
```

The output was coherent English (not random characters), indicating the model structure was correct
but weights were wrong. The nan/inf appeared during softmax when logits contained extreme values.

**Debugging approach:**
1. Isolated `Fp8Linear.forward()` - produced nan/inf
2. Printed raw values:
   ```
   scale_inv min: 0.000148, max: 0.001305, mean: 0.000276
   weight (as fp32) min: -448.0, max: 448.0

   Dequant (* scale_inv): min=-0.30, max=0.28  ← CORRECT
   Dequant (/ scale_inv): min=-663505, max=616112  ← WRONG (overflow)
   ```
3. The naming `weight_scale_inv` was the clue - it's already the inverse

**Fix:**
```python
# Before (wrong)
scales = (1.0 / self.weight_scale_inv).view(out_blocks, 1, in_blocks, 1)

# After (correct)
# weight_scale_inv IS the inverse scale (multiply by it to dequantize)
scales = self.weight_scale_inv.view(out_blocks, 1, in_blocks, 1)
```

**Verification:**
```python
# After fix
Dequantized weights:
  min: -0.584473
  max: 0.417480
  std: 0.022690
  Values outside [-1, 1]: 0 / 10485760  # All values in expected range
```

Model now generates coherent responses matching the prompt.

**Insight:** When working with quantized weight formats, pay close attention to naming conventions:
- `weight_scale` → divide weights by this to dequantize
- `weight_scale_inv` → multiply weights by this to dequantize (inverse already computed)

The "inv" suffix is a strong hint. When in doubt, check the actual value ranges:
- Scales are typically small (0.001-0.01) to map FP8's [-448, 448] range to typical weight ranges
- If dequantized values explode, you're probably applying the scale in the wrong direction


### 14. Streaming Weight Loading to Avoid OOM

**Date:** 2025-12-13

**File:** `llm_infer/pipelines/model/transformer.py`

**Problem:** Loading large models caused OOM even when the final model would fit in GPU memory. The
issue was PyTorch's default behavior: `model.to(device)` allocates all parameters on GPU first, then
loads weights. For a 4B parameter model:

1. `model = TransformerModel(config)` - Creates parameters on CPU (~8GB for FP16)
2. `model.to("cuda")` - Allocates GPU memory for all parameters (~8GB)
3. `model.load_state_dict(weights)` - Loads weights, requiring temporary CPU copy

Peak memory: ~16GB+ (GPU allocation + CPU weights + overhead), even though final model is only 4GB
(FP8) or 8GB (FP16).

**Symptom:**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 742.00 MiB.
GPU 0 has a total capacity of 7.53 GiB of which 151.25 MiB is free.
```

This occurred even for FP8 models that should only need ~4GB.

**Fix:** Stream weights directly to GPU one tensor at a time, bypassing bulk allocation:

```python
def _stream_weights(self, files: list[Path], total: int, on_progress) -> None:
    """Stream weights from safetensor files directly to GPU.

    Weights are loaded one tensor at a time, converted to target dtype,
    moved to GPU, and assigned to model parameters. This enables loading
    quantized models without OOM from bulk allocation.
    """
    loaded = 0
    for sf_path in files:
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)

                # Preserve dtype for quantized tensors
                if not self._is_awq_int_tensor(key) and not self._is_fp8_tensor(key):
                    tensor = tensor.to(dtype=self.dtype)

                # Move to GPU and assign (not copy) to parameter
                tensor = tensor.to(device=self.device)
                self._assign_weight(key, tensor)
                del tensor  # Free CPU memory immediately

                loaded += 1
                if on_progress:
                    on_progress("stream", loaded, total)
```

**Key implementation details:**

1. **No pre-allocation**: Model parameters are created with empty tensors, then replaced in-place
2. **One tensor at a time**: Load → convert dtype → move to GPU → assign → delete CPU copy
3. **Preserve quantized dtypes**: AWQ (int32) and FP8 (float8_e4m3fn) tensors skip dtype conversion
4. **Direct assignment**: Use `param.data = tensor` instead of copy to avoid duplication

**Helper for dtype preservation:**
```python
def _is_fp8_tensor(self, key: str) -> bool:
    """Check if tensor is FP8 and should not have dtype converted."""
    if self.config.quant_method != "fp8":
        return False
    # Only projection weights and their scales are FP8
    fp8_projs = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    for proj in fp8_projs:
        if key.endswith(f".{proj}.weight") or key.endswith(f".{proj}.weight_scale_inv"):
            return True
    return False
```

**Memory profile (Qwen3-4B-FP8 on 7.5GB GPU):**
```
Before streaming: OOM during load
After streaming:
  - Weights loaded: 955ms
  - GPU allocated: 6.45GB (4GB weights + 2.25GB KV cache)
  - Peak usage: ~6.5GB
```

**Insight:** The `safetensors` library supports lazy loading via `safe_open()`, which memory-maps
the file and loads tensors on demand. Combined with immediate deletion of CPU copies and direct
parameter assignment, this enables loading models that would otherwise OOM during the load phase.

For quantized models, this is especially important:
- FP8: 4GB model would previously need 8GB+ during load (FP8→FP16 conversion overhead)
- AWQ: 2GB model would need 4GB+ during load (unpacking overhead)

Streaming eliminates the "load peak" and keeps memory usage close to final model size


### 14. Emoji/Unicode Corruption in vLLM Streaming

**Date:** 2025-12-17

**File:** `llm_infer/pipelines/engines/vllm_engine.py`

**Problem:** Emojis and multi-byte Unicode characters displayed as mojibake (`�` or `���`) in CLI
streaming output when using vLLM backend.

**Symptom:**
```bash
$ ./cli.py q "Hello!"
Hello! How can I assist you today? ��
```

**Root cause:** The vLLM sync API doesn't support true streaming, so `generate_stream_sync()`
simulates streaming by:
1. Generating the complete output via `engine.generate()`
2. Re-tokenizing the output text
3. Decoding each token individually to simulate token-by-token streaming

The bug was in step 3. When an emoji (e.g., 😊) spans multiple tokens in the tokenizer's vocabulary,
decoding each token individually produces invalid UTF-8:

```python
# Buggy code
token_ids = self._tokenizer.encode(text, add_special_tokens=False)
for token_id in token_ids:
    token_text = self._tokenizer.decode([token_id])  # Each token decoded separately
    result._tokens.append(token_text)
```

For example, 😊 might be encoded as tokens `[<byte_F0>, <byte_9F>, <byte_98>, <byte_8A>]`. Decoding
each individually produces 4 replacement characters instead of one emoji.

**First fix attempt (failed):** Incremental decoding - decode all tokens up to position `i` and
yield the delta:

```python
for i in range(len(token_ids)):
    current_text = self._tokenizer.decode(token_ids[:i+1], skip_special_tokens=True)
    if len(current_text) > len(prev_text):
        result._tokens.append(current_text[len(prev_text):])
    prev_text = current_text
```

This failed because when an incomplete byte sequence is decoded, it produces a replacement character
`�`. When the full sequence is decoded, the emoji has the same character length as the replacement
character (both are 1 character). So `len(current_text) > len(prev_text)` was False and the emoji
was never yielded.

**Working fix:** Since vLLM's sync API generates the complete output first anyway, stream
character-by-character from the already-decoded text:

```python
# Fixed code - stream by character
text = output.text  # Already correctly decoded by vLLM
if text:
    for char in text:
        result._tokens.append(char)
```

This gives the same streaming UI effect while avoiding all token-boundary Unicode issues.

**Files changed:**
- `llm_infer/pipelines/engines/vllm_engine.py` - Fixed `generate_stream_sync()` method

**Insight:** When simulating streaming from a complete result, work with the already-decoded text
rather than re-tokenizing and re-decoding. The tokenizer's byte-level tokens don't align with
Unicode character boundaries, making per-token decoding fundamentally broken for multi-byte
characters.


### 15. appinfra `with_on_startup` Breaks IPC Response Queue

**Date:** 2026-01-16

**File:** `llm_infer/serving/dispatch/main.py`

**Problem:** When using appinfra's `with_on_startup()` callback with subprocess mode
(`.subprocess.with_ipc()`), the response queue stopped delivering messages. The main process
completed requests successfully, but responses never reached the API subprocess, causing client
timeouts.

**Symptom:**
```text
# Server logs show successful processing (81ms):
[D] requested   request_id[chatcmpl-22bbd67...] stream[False]
[D] decoded     [79ms]
[D] complete    [81ms] prompt_tokens[30] completion_tokens[34]
[T] queueing response  response_id[chatcmpl-22bbd67...]
[T] response queued    response_id[chatcmpl-22bbd67...]

# But client times out after 180s:
TimeoutError: Request chatcmpl-22bbd67... timed out after 180.0s
```

**Root cause:** A bug in appinfra's handling of startup callbacks when combined with subprocess IPC
mode. The startup callback execution interfered with the response queue listener thread in the API
subprocess. The exact mechanism was in appinfra's internal code.

**Debugging approach:**
1. Added trace logging around `response_q.put()` - confirmed responses WERE being queued
2. Confirmed engine completed requests in ~80ms but API timed out at 180s
3. Disabled `with_on_startup()` callback - IPC started working immediately
4. Re-enabled callback after appinfra fix - confirmed fix worked

**Code that triggered the bug:**
```python
# This startup callback broke IPC when combined with subprocess mode
def _add_lora_startup(self, builder: Any) -> Any:
    lora_cfg = self._config.engines.vllm.lora
    if lora_cfg.enabled and lora_cfg.base_path:
        builder = builder.with_on_startup(
            self._create_adapter_startup_callback(lora_cfg.base_path)
        )
    return builder

# Used with subprocess IPC:
builder.subprocess.with_ipc(self._request_q, self._response_q)
```

**Fix:** Bug was in appinfra - reported and fixed by appinfra team.

**Trace logging added for future debugging:**
```python
# In loop.py - trace-level logging for response queue operations
for response in handler.step():
    if lg:
        lg.trace("queueing response", extra={"response_id": response.id})
    response_q.put(response)
    if lg:
        lg.trace("response queued", extra={"response_id": response.id})
```

**Insight:** When debugging IPC issues between processes, add logging on BOTH sides of the queue.
Confirming that `put()` succeeds but the other side never receives narrows the problem to the
queue consumer or the queue itself. In this case, it pointed to appinfra's IPC listener being
broken by the startup callback.

