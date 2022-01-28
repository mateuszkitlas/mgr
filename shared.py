import datetime
import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from inspect import getframeinfo, stack
from socketserver import ThreadingMixIn
from typing import Any, Awaitable, Callable, Tuple, TypeVar

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # disable tensorflow warnings

project_dir = os.path.dirname(os.path.realpath(__file__))

T = TypeVar("T")


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


def _serve(port: int, callback: Callable[[Any], Any]):
    class S(BaseHTTPRequestHandler):
        def log_message(self, *args: Any):
            pass

        def do_POST(self):
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length).decode("utf-8")
            try:
                in_data = json.loads(post_data)
                out_data = callback(in_data)
                result = (False, out_data)
            except Exception as e:
                result = (
                    True,
                    "".join(
                        traceback.format_exception(type(e), value=e, tb=e.__traceback__)
                    ),
                )
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        """Handle requests in a separate thread."""

    httpd = ThreadedHTTPServer(("localhost", port), S)
    httpd.serve_forever()


def serve(callback: Callable[[Any], Any]):
    _serve(int(sys.argv[1]), callback)


def disable_syba():
    if os.environ.get("DISABLE_SYBA"):
        caller = getframeinfo(stack()[1][0])
        print(f"DISABLE_SYBA: {caller.filename}:{caller.lineno}")
        return True
    else:
        return False
