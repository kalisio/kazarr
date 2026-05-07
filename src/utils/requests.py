import threading
from fastapi import Request


async def watch_disconnection(request: Request, cancel_event: threading.Event):
    """
    Asynchronous task that waits for the client to disconnect.
    When disconnected, it sets the thread-safe event.
    """
    try:
        if await request.is_disconnected():
            cancel_event.set()
    except Exception:
        cancel_event.set()
