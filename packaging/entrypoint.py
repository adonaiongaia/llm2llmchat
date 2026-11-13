"""PyInstaller entry point used for the AppImage build."""

import sys

from llm2llmchat.app import main

if __name__ == "__main__":
    sys.exit(main())
