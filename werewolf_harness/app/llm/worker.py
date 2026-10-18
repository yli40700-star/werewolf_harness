"""Entry point for a model worker process.

    python -m werewolf_harness.app.llm.worker

Scale by running more of them; they share the Redis consumer group.
"""
import asyncio

from .queue import main

if __name__ == "__main__":
    asyncio.run(main())
