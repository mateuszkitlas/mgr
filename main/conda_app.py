import subprocess
import json
from typing import Any, Generic, Optional, Tuple, TypeVar
from urllib import request
from urllib.error import URLError
from asyncio import sleep, get_event_loop
import logging
import concurrent.futures
import os
from http.client import RemoteDisconnected

logger = logging.Logger(__name__)
conda_dir = os.environ["CONDA_PREFIX"]


class AppKilled(Exception):
    pass


class AppError(Exception):
    pass


def _fetch(port: int, data: Any, remaining_retries: int = 5) -> Tuple[bool, Any]:
    print("_fetch started")
    try:
        req = request.Request(
            f"http://localhost:{port}", method="POST", data=json.dumps(data).encode(),
        )
        res = request.urlopen(req)
        return (False, json.loads(res.read()))
    except ConnectionResetError as e:
        if remaining_retries > 0:
            return _fetch(port, data, remaining_retries - 1)
        else:
            return (True, e)
    except Exception as e:
        return (True, e)
    finally:
        print("_fetch done")


T = TypeVar("T")
R = TypeVar("R")


class CondaApp(Generic[T, R]):
    """
    https://github.com/python/mypy/issues/6073

    @staticmethod
    @asynccontextmanager
    async def many(first_port: int, pairs: List[Tuple[str, str]]):
        apps = [CondaApp(first_port + i, subdir, env)
                for (i, (subdir, env)) in enumerate(pairs)]
        try:
            await gather(*(app.start() for app in apps))
            yield [app.fetch for app in apps]
        finally:
            for app in apps:
                app.stop()
    """

    def __init__(self, port: int, subdir: str, env: str):
        self.port = port
        self.subdir = subdir
        self.env = env
        self.p: Optional[subprocess.Popen[Any]] = None
        self.executor: Optional[concurrent.futures.ProcessPoolExecutor] = None

    def __str__(self):
        return f'CondaApp(port={self.port}, subdir="{self.subdir}", env="{self.env}", running={self.running()})'

    def running(self):
        return self.p and (self.p.poll() is None)

    async def fetch(self, data: Optional[T]) -> R:
        if self.running():
            is_internal_error, v = await get_event_loop().run_in_executor(
                self.executor, _fetch, self.port, data
            )
            if is_internal_error:
                raise v
            else:
                is_app_error, value = v
                if is_app_error:
                    raise AppError(value)
                else:
                    return value
        else:
            raise AppKilled()

    async def start(self):
        logger.info(self)
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
                if not (isinstance(e.reason, OSError) and e.reason.errno == 111):
                    raise e
        logger.info(self)

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
            self.p = None

    async def __aexit__(self, _t: Any, _e: Any, _tb: Any):
        self.stop()
