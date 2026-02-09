"""Renderer for Microsoft Phi-4 chat format (ChatML variant)."""

import tinker

from tinker_cookbook.renderers.base import (
    Message,
    RenderContext,
    RenderedMessage,
    Renderer,
    ensure_text,
    parse_response_for_stop_token,
)


class Phi4Renderer(Renderer):
    """Renderer for Microsoft Phi-4 Instruct models.

    Format::

        <|system|>
        You are a helpful AI assistant.<|end|>
        <|user|>
        What can you help me with?<|end|>
        <|assistant|>
        I can help with many things!<|end|>

    This is the ChatML-style format used by Phi-4-mini-instruct.
    """

    @property
    def has_extension_property(self) -> bool:
        """Phi4 satisfies the extension property - no content is stripped from history."""
        return True

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        role = message["role"]
        header_str = f"<|{role}|>\n"
        output_str = ensure_text(message["content"]) + "<|end|>"

        header = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(header_str, add_special_tokens=False)
        )
        output: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(output_str, add_special_tokens=False)
            )
        ]
        return RenderedMessage(header=header, output=output)

    @property
    def _bos_tokens(self) -> list[int]:
        # Phi-4 uses the tokenizer's BOS token
        bos = self.tokenizer.encode("", add_special_tokens=True)
        return bos

    @property
    def _end_message_token(self) -> int:
        (token,) = self.tokenizer.encode("<|end|>", add_special_tokens=False)
        return token

    @property
    def _endoftext_token(self) -> int:
        (token,) = self.tokenizer.encode("<|endoftext|>", add_special_tokens=False)
        return token

    def get_stop_sequences(self) -> list[int]:
        return [self._end_message_token, self._endoftext_token]

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        return parse_response_for_stop_token(response, self.tokenizer, self._end_message_token)
