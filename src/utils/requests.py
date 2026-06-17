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


def get_from_query(variable_name: str | None, current_value, request: Request):
    if variable_name is None:
        return current_value
    
    query_value = request.query_params.get(variable_name)
    if query_value is not None:
        # request.query_params is an immutable QueryParams object in Starlette/FastAPI.
        # We pop from its internal _dict to modify it by reference.
        request.query_params._dict.pop(variable_name, None)
        return query_value if current_value is None else current_value
    return current_value
