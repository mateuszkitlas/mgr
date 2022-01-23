from http.server import SimpleHTTPRequestHandler, HTTPServer
from typing import Callable, Any
import json
import os
import sys
import traceback

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # disable tensorflow warnings


project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def _serve(port: int, callback: Callable[[Any], Any]):
    class S(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length).decode("utf-8")
            try:
                in_data = json.loads(post_data)
                # print(f"CondaApp[port={port}, in] {in_data}")
                out_data = callback(in_data)
                # print(f"CondaApp[port={port}, out]")
                result = (False, out_data)
            except Exception as e:
                # print(f"CondaApp[port={port}, err] {e}")
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

    httpd = HTTPServer(("localhost", port), S)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()


def serve(callback: Callable[[Any], Any]):
    _serve(int(sys.argv[1]), callback)
