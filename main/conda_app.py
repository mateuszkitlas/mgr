import concurrent.futures
import json
import logging
import os
import subprocess
from asyncio import get_event_loop, sleep
from http.client import RemoteDisconnected
from typing import Any, Generic, Optional, Tuple, TypeVar
from urllib import request
from urllib.error import URLError

from shared import Timer

logger = logging.Logger(__name__)
conda_dir = os.environ["CONDA_PREFIX"]


class AppKilled(Exception):
    pass


class AppError(Exception):
    pass


def _is_errno(e: URLError, errno: int):
    return isinstance(e.reason, OSError) and e.reason.errno == errno


def _fetch(port: int, data: Any, remaining_retries: int = 5) -> Tuple[bool, Any]:
    def f(e: Exception):
        if remaining_retries > 0:
            return _fetch(port, data, remaining_retries - 1)
        else:
            return (True, e)

    try:
        req = request.Request(
            f"http://localhost:{port}",
            method="POST",
            data=json.dumps(data).encode(),
        )
        res = request.urlopen(req)
        return (False, json.loads(res.read()))
    except URLError as e:
        if _is_errno(e, 110):
            return f(e)
        else:
            raise e
    except ConnectionResetError as e:
        return f(e)
    except Exception as e:
        return (True, e)


T = TypeVar("T")
R = TypeVar("R")


class CondaApp(Generic[T, R]):
    def __init__(self, port: int, subdir: str, env: str):
        self.port = port
        self.subdir = subdir
        self.env = env
        self.p: Optional[subprocess.Popen[Any]] = None
        self.executor: Optional[concurrent.futures.ProcessPoolExecutor] = None

    def __str__(self):
        return f'CondaApp(port={self.port}, subdir="{self.subdir}", env="{self.env}")'

    def running(self):
        return self.p and (self.p.poll() is None)

    async def fetch(self, data: Optional[T]) -> R:
        if self.running():
            is_internal_error, v = await get_event_loop().run_in_executor(
                self.executor, _fetch, self.port, data
            )
            if is_internal_error:
                logger.exception(v)
                raise v
            else:
                is_app_error, value = v
                if is_app_error:
                    raise AppError(value)
                else:
                    return value
        else:
            raise AppKilled()

    async def _start(self):
        assert not self.executor
        assert not self.p
        self.executor = concurrent.futures.ProcessPoolExecutor()
        self.p = subprocess.Popen(
            [
                os.path.join(conda_dir, "envs", self.env, "bin/python"),
                "-m",
                f"{self.subdir}.main",
                str(self.port),
            ]
        )
        while self.running():
            await sleep(1)
            try:
                await self.fetch(None)
                break
            except RemoteDisconnected:
                pass
            except URLError as e:
                if not _is_errno(e, 111):
                    raise e

    async def start(self):
        time, _ = await Timer.acalc(self._start())
        print(f"{self} starting time: {time}")

    async def __aenter__(self):
        await self.start()
        return self.fetch

    def stop(self):
        assert self.executor
        self.executor.shutdown()
        self.executor = None
        if self.running():
            assert self.p
            self.p.kill()
            try:
                self.p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                print(f"{self} did not exited gracefully")
            self.p = None

    async def __aexit__(self, *_: Any):
        self.stop()
