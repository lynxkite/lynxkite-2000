"""Joblib memory cache with coroutine deduplication.

If you call a Joblib-cached slow async function twice, Joblib will start it twice.
It only puts the result in the cache. We would like to only run it once, and
let the second call just wait for the result of the first and only execution.
"""

import asyncio
import functools
import inspect

import joblib


class Memory:
    def __init__(self, *args, **kwargs):
        self._memory = joblib.Memory(*args, **kwargs)

    def cache(self, func, **kwargs):
        return self._memory.cache(deduplicate(func), **kwargs)


def deduplicate(func):
    if inspect.iscoroutinefunction(func):
        # Joblib caching is added around this function.
        # The result is only written to the cache after completion.
        @functools.wraps(func)
        async def run_func(*args, **kwargs):
            return await get_task(*args, **kwargs)

        # If the function is called before completion, the call
        # comes to `get_task()`. A second line of caching stores
        # the task itself here, so it is shared across calls.
        @functools.cache
        def get_task(*args, **kwargs):
            return asyncio.create_task(func(*args, **kwargs))

        return run_func
    else:
        # We can't help with synchronous functions.
        return func
