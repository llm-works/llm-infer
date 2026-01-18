"""UTF-8 stream buffer for handling incomplete multi-byte sequences.

Handles byte-level tokenizers that may split multi-byte UTF-8 characters
across multiple tokens.
"""

import codecs


class Utf8StreamBuffer:
    """Buffer for incomplete UTF-8 sequences in streaming output.

    Handles byte-level tokenizers (like Qwen) that may split multi-byte
    UTF-8 characters (e.g., emojis) across multiple tokens. Each partial
    byte sequence is buffered until complete characters can be output.

    Example:
        buffer = Utf8StreamBuffer()
        for chunk in stream:
            print(buffer.process(chunk), end="")
        print(buffer.flush())
    """

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def process(self, text: str) -> str:
        """Process text, buffering incomplete UTF-8 sequences.

        Re-encodes text to bytes using surrogateescape to preserve any
        partial UTF-8 bytes, then decodes incrementally to output only
        complete characters.

        Args:
            text: Text chunk that may contain incomplete UTF-8 sequences.

        Returns:
            Text with only complete UTF-8 characters. Incomplete sequences
            are buffered for the next call.
        """
        # surrogateescape preserves invalid bytes as surrogates
        data = text.encode("utf-8", errors="surrogateescape")
        return self._decoder.decode(data, final=False)

    def flush(self) -> str:
        """Flush remaining buffered bytes at end of stream.

        Call this after all chunks have been processed to get any
        remaining buffered content.
        """
        return self._decoder.decode(b"", final=True)
