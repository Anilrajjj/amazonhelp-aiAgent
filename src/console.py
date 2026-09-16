"""Force UTF-8 on stdout.

The dataset is multilingual and emoji-heavy; on Windows the default console
codec is cp1252 and any print() of a real message raises UnicodeEncodeError.
Every CLI entry point calls this first.
"""
import sys


def init():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
