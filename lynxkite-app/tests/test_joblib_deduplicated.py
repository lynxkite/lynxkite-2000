import asyncio

from lynxkite_app import joblib_deduplicated
import joblib


async def run_test(memory, expected_calls):
    started = asyncio.Event()
    finish = asyncio.Event()
    call_count = 0

    @memory.cache
    async def expensive_computation(value):
        nonlocal call_count
        call_count += 1
        started.set()
        await finish.wait()
        return value * 55

    task1 = asyncio.create_task(expensive_computation(10))
    task2 = asyncio.create_task(expensive_computation(10))
    await started.wait()
    assert call_count == expected_calls

    finish.set()
    assert await task1 == 550
    assert await task2 == 550
    assert await expensive_computation(10) == 550
    assert call_count == expected_calls


def test_joblib_does_not_deduplicate(tmp_path):
    asyncio.run(run_test(joblib.Memory(tmp_path), 2))


def test_joblib_deduplicated_deduplicates(tmp_path):
    asyncio.run(run_test(joblib_deduplicated.Memory(tmp_path), 1))
