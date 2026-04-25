"""HuggingFace tokenizer implementation."""

from transformers import AutoTokenizer

from .config import TokenizerConfig


class HuggingFaceTokenizer:
    """HuggingFace tokenizer wrapper.

    Wraps HuggingFace's AutoTokenizer and applies configuration from
    TokenizerConfig, mapping abstract config options to HF internals.
    """

    def __init__(self, model_path: str, config: TokenizerConfig | None = None):
        """Initialize tokenizer from model path.

        Args:
            model_path: Path to HuggingFace model directory or model ID.
            config: Tokenizer configuration. If None, uses defaults.
        """
        config = config or TokenizerConfig()

        # Load base tokenizer
        self._hf = AutoTokenizer.from_pretrained(model_path)

        # Apply config - map abstract config to HF internals
        if config.pre_tokenizer_pattern:
            self._apply_pre_tokenizer_pattern(config.pre_tokenizer_pattern)

        # Ensure pad token (common requirement)
        if self._hf.pad_token_id is None:
            self._hf.pad_token_id = self._hf.eos_token_id

    def _apply_pre_tokenizer_pattern(self, pattern: str) -> None:
        """Replace pre-tokenizer regex pattern.

        This patches the pre-tokenizer's Split pattern with a corrected regex.
        Used to fix buggy patterns shipped with some models (e.g., Mistral).
        """
        try:
            import tokenizers

            # Access the backend tokenizer's pre-tokenizer sequence
            pre_tokenizer = self._hf.backend_tokenizer.pre_tokenizer
            if pre_tokenizer is None:
                return

            # Replace first element (Split pattern) with corrected regex
            # Only works if pre_tokenizer is a Sequence type
            pre_tokenizer[0] = tokenizers.pre_tokenizers.Split(
                pattern=tokenizers.Regex(pattern),
                behavior="isolated",
            )
        except (AttributeError, IndexError, TypeError, ImportError):
            # If structure doesn't match expected, skip silently
            # This handles tokenizers without Split pre-tokenizers or
            # non-Sequence pre-tokenizers (e.g., Metaspace)
            pass

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text to token IDs."""
        result: list[int] = self._hf.encode(text, add_special_tokens=add_special_tokens)
        return result

    def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        # HF decode() returns str for single sequence input (list[int])
        result = self._hf.decode(tokens, skip_special_tokens=skip_special_tokens)
        assert isinstance(result, str)
        return result

    @property
    def eos_token_id(self) -> int | None:
        """End-of-sequence token ID."""
        result: int | None = self._hf.eos_token_id
        return result

    @property
    def pad_token_id(self) -> int | None:
        """Padding token ID."""
        result: int | None = self._hf.pad_token_id
        return result

    @property
    def has_chat_template(self) -> bool:
        """Whether this tokenizer has a chat template."""
        return self._hf.chat_template is not None

    def encode_chat(
        self,
        message: str | list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> list[int]:
        """Encode messages using the chat template.

        Args:
            message: Either a user message string, or a list of message dicts
                with 'role' and 'content' keys.
            add_generation_prompt: Whether to add the assistant prompt prefix.

        Returns:
            List of token IDs with chat formatting applied.
        """
        if isinstance(message, str):
            messages = [{"role": "user", "content": message}]
        else:
            messages = message
        # With tokenize=True and single conversation, returns list[int]
        result = self._hf.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
        )
        assert isinstance(result, list)
        return result  # type: ignore[return-value]
