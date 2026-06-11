class InlineDirectiveStripper:
    """Stateful inline directive stripper for VoxCPM2 TTS.

    Strips all (...) parenthetical content from text, unconditionally.
    No content-based judgment — any (...) is stripped regardless of whether
    it contains English, CJK, or mixed content.

    VoxCPM2 uses inline directives like (happy), (angry), (开心地) etc.
    These are stripped from transcript text sent to the client,
    but preserved in the raw text sent to TTS.

    Usage examples:
        >>> stripper = InlineDirectiveStripper()
        >>> stripper.feed("(happy)你好")
        '你好'
        >>> stripper.reset()

        >>> stripper.feed("(angry and upset)我不同意")
        '我不同意'
        >>> stripper.reset()

        >>> stripper.feed("(开心地)你好")
        '你好'
        >>> stripper.reset()

        >>> stripper.feed("(中文内容)更多文字")
        '更多文字'
        >>> stripper.reset()

    Cross-chunk directive example:
        >>> stripper = InlineDirectiveStripper()
        >>> stripper.feed("(happ")
        ''
        >>> stripper.feed("y)你好")
        '你好'
        >>> stripper.flush()
        ''
        >>> stripper.reset()

    Incomplete directive (never closed) — discarded on flush:
        >>> stripper = InlineDirectiveStripper()
        >>> stripper.feed("(happy")
        ''
        >>> stripper.flush()
        ''
    """

    def __init__(self):
        self._in_directive: bool = False       # Whether currently inside parentheses
        self._directive_buffer: str = ""        # Content inside current parentheses (for cross-chunk tracking)

    def feed(self, text_delta: str) -> str:
        """Process a streaming text delta, stripping all (...) content.

        Iterates character-by-character through text_delta. When a '(' is
        encountered, the processor enters directive mode and buffers subsequent
        characters until the matching ')' arrives, at which point the entire
        (...) block is discarded.

        This method is stateful across calls so that a directive whose opening
        '(' and closing ')' arrive in separate chunks is handled correctly.

        Args:
            text_delta: A small piece of text from the LLM stream. May contain
                partial parenthetical directives that span chunk boundaries.

        Returns:
            Text with all complete (...) blocks stripped. Characters inside
            an unclosed parenthesis are buffered internally and not returned
            until the closing ')' is encountered (or discarded on flush).
        """
        output: list[str] = []

        for char in text_delta:
            if not self._in_directive:
                if char == '(':
                    # Enter directive mode — start tracking parenthetical content
                    self._in_directive = True
                    self._directive_buffer = ""
                else:
                    # Normal character outside parentheses — pass through
                    output.append(char)
            else:
                if char == ')':
                    # End of directive — discard entire (...) block
                    self._in_directive = False
                    self._directive_buffer = ""
                else:
                    # Inside directive — buffer but don't output
                    self._directive_buffer += char

        return "".join(output)

    def flush(self) -> str:
        """Flush any remaining state after all text deltas have been fed.

        If a '(' was opened but never closed (incomplete directive), the
        buffered content is discarded — incomplete directives should not
        leak into the output.

        Resets all internal state after flushing.

        Returns:
            Empty string. Any incomplete directive is silently dropped.
        """
        self._in_directive = False
        self._directive_buffer = ""
        return ""

    def reset(self) -> None:
        """Reset all internal state to initial values.

        Call this when starting a new text processing session or after
        flush() if reusing the same instance.
        """
        self._in_directive = False
        self._directive_buffer = ""
