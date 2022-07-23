import datetime
import json
import logging
import os
import signal
import subprocess
import sys
import traceback
from asyncio import Future, get_event_loop, sleep
from concurrent.futures import ThreadPoolExecutor
from http.client import RemoteDisconnected
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import (
    Any,
    Awaitable,
    Callable,
    Generic,
    Iterable,
    Optional,
    Tuple,
    Type,
    TypeVar,
    cast,
)
from urllib import request
from urllib.error import URLError

T = TypeVar("T")
R = TypeVar("R")
Fn = Callable[[T], R]
_U = TypeVar("_U")

paracetamol_smiles = "CC(=O)Nc1ccc(O)cc1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # disable tensorflow warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
project_dir = os.path.dirname(os.path.realpath(__file__))
logger = logging.Logger(__name__)
logger.setLevel(logging.DEBUG)
conda_dir = os.environ["CONDA_PREFIX"]


class Timer:
    @staticmethod
    def calc(fn: Callable[[], T]) -> Tuple[float, T]:
        timer = Timer()
        result = fn()
        return timer.done(), result

    @staticmethod
    async def acalc(awaitable: Awaitable[T]) -> Tuple[float, T]:
        timer = Timer()
        result = await awaitable
        return timer.done(), result

    def __init__(self):
        self.start = datetime.datetime.now()

    def done(self):
        self.delta = (datetime.datetime.now() - self.start).total_seconds()
        return self.delta


def _serve(port: int, callback: Fn[Any, Any]):
    class S(BaseHTTPRequestHandler):
        def log_message(self, *args: Any):
            pass

        def do_POST(self):
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length).decode("utf-8")
            try:
                result = (
                    False,
                    None if post_data == "status" else callback(json.loads(post_data)),
                )
            except Exception as e:
                result = (
                    True,
                    "".join(
                        traceback.format_exception(type(e), value=e, tb=e.__traceback__)
                    ),
                )
            try:
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except BrokenPipeError:
                pass

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        pass

    httpd = ThreadedHTTPServer(("localhost", port), S)
    httpd.serve_forever()


def serve(callback: Fn[Any, Any]):
    _serve(int(sys.argv[1]), callback)


class AppKilled(Exception):
    pass


class AppError(Exception):
    pass


def _is_errno(e: URLError, errno: int):
    return isinstance(e.reason, OSError) and e.reason.errno == errno


_running = True


def _fetch(port: int, data: bytes, remaining_retries: int = 5) -> Tuple[bool, Any]:
    def f(e: Exception):
        if remaining_retries > 0 and _running:
            return _fetch(port, data, remaining_retries - 1)
        else:
            return (True, e)

    try:
        req = request.Request(
            f"http://localhost:{port}",
            method="POST",
            data=data,
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


class CondaApp(Generic[T, R]):
    def __init__(self, port: int, module: str, env: str):
        self.port = port
        self.module = module
        self.env = env
        self.p: Optional[subprocess.Popen[Any]] = None
        self.executor: Optional[ThreadPoolExecutor] = None
        self._futures: set[Future[Any]] = set()

    def __str__(self):
        return f'CondaApp(port={self.port}, module="{self.module}", env="{self.env}")'

    def running(self):
        return self.p and (self.p.poll() is None)

    def _process_fetch(self, x: Tuple[bool, Any]):
        is_internal_error, v = x
        if is_internal_error:
            if isinstance(v, KeyboardInterrupt):
                logger.error(KeyboardInterrupt)
            else:
                logger.exception(v)
            raise v
        else:
            is_app_error, value = v
            if is_app_error:
                raise AppError(value)
            else:
                return value

    def fetch_sync(self, data: T) -> R:
        if self.running():
            return self._process_fetch(_fetch(self.port, json.dumps(data).encode()))
        else:
            raise AppKilled()

    def fetch(self, data: T):
        return self._fetch(json.dumps(data))

    async def _wrap_future(self, f: "Future[_U]") -> _U:
        self._futures.add(f)
        try:
            return await f
        finally:
            self._futures.remove(f)

    async def _fetch(self, raw_data: str) -> R:
        if self.running():
            return self._process_fetch(
                await self._wrap_future(
                    get_event_loop().run_in_executor(
                        self.executor, _fetch, self.port, raw_data.encode()
                    )
                )
            )
        else:
            raise AppKilled()

    async def _start(self):
        assert not self.executor
        assert not self.p
        self.executor = ThreadPoolExecutor()
        self.p = subprocess.Popen(
            [
                os.path.join(conda_dir, "envs", self.env, "bin/python"),
                "-m",
                f"{self.module}.main",
                str(self.port),
            ],
            env={
                **os.environ,
                "PYTHONPATH": f"{os.environ.get('PYTHONPATH', '')}:{project_dir}/askcos-core/",
            },
        )
        while self.running():
            await sleep(1)
            try:
                await self._fetch("status")
                break
            except RemoteDisconnected:
                pass
            except URLError as e:
                if not _is_errno(e, 111):
                    raise e

    async def __aenter__(self):
        time, _ = await Timer.acalc(self._start())
        print(f"{self} starting time: {time}")
        return self.fetch, self.fetch_sync

    async def __aexit__(self, *_: Any):
        global _running
        _running = False
        if self.executor:
            self.executor.shutdown(wait=False, cancel_futures=True)
        for f in self._futures:
            if not f.cancelled():
                f.cancel()
        if self.running():
            assert self.p
            self.p.send_signal(signal.SIGINT)


class Db:
    def __init__(self, name: str, readonly: bool):
        from sqlitedict import SqliteDict

        self.readonly = readonly
        self._write_counter = 0
        self.db = SqliteDict(
            f"{project_dir}/results/{name}.sqlite",
            outer_stack=False,
            autocommit=False,
            flag="r" if readonly else "c",
            journal_mode="OFF",
        )

    def _write(self, raw_key: str, value: T) -> T:
        self.db[raw_key] = json.dumps(value, sort_keys=True)
        self._write_counter += 1
        if self._write_counter % 1000 == 0:
            self._commit()
        return value

    async def maybe_create(self, key: Any, f: Callable[[], Awaitable[Any]]):
        raw_key = json.dumps(key, sort_keys=True)
        if raw_key not in self.db:
            self._write(raw_key, await f())

    def read_or_create_sync(self, key: Any, f: Callable[[], T]) -> T:
        raw_key = json.dumps(key, sort_keys=True)
        return (
            cast(T, self._read(raw_key))
            if raw_key in self.db
            else self._write(raw_key, f())
        )

    def has(self, key: Any):
        return json.dumps(key, sort_keys=True) in self.db

    async def read_or_create(
        self, key: Any, f: Callable[[], Awaitable[T]]
    ) -> Awaitable[T]:

        raw_key = json.dumps(key, sort_keys=True)
        return (
            self.db[raw_key] if raw_key in self.db else self._write(raw_key, await f())
        )

    def _read(self, raw_key: Any, type: Type[T] = Any, fallback: bool = True) -> T:
        return cast(
            type,
            json.loads(self.db.get(raw_key, "null") if fallback else self.db[raw_key]),
        )

    def read(self, key: Any, type: Type[T], fallback: bool = True) -> T:
        return self._read(json.dumps(key, sort_keys=True), type, fallback)

    def write(self, key: Any, value: Any):
        self._write(json.dumps(key, sort_keys=True), value)

    def items(
        self, _ktype: Type[T] = Any, _vtype: Type[R] = Any
    ) -> Iterable[Tuple[T, R]]:
        return ((json.loads(k), json.loads(v)) for k, v in self.db.iteritems())

    def as_json(self):
        return json.dumps({k: v for k, v in self.items()}, indent=2)

    def __enter__(self):
        self.db.__enter__()
        return self

    def _commit(self):
        if not self.readonly:
            print(
                f"committing {self._write_counter} write ops to {self.db.filename}..."
            )
            self.db.commit(True)
            self._write_counter = 0
            print(f"commited")

    def __exit__(self, *exc_info):
        self._commit()
        self.db.close()
