# Static server that logs how many bytes were actually written to each response.
# argv: port logfile root [megabytes_per_second]
# megabytes_per_second 0 (default) means no throttle. The limit is per connection.
import os, sys, time, json, threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

port = int(sys.argv[1])
log_path = sys.argv[2]
root = sys.argv[3]
mbps = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
bw = mbps * 1e6
lock = threading.Lock()


# H: Local HTTP handler that throttles bandwidth to simulate slow networks.
# Why these APIs: json: reads/writes structured data files; open(): reads/writes a local file; os / os.path:
#   handles local paths; ONNX / onnxruntime: exports, quantizes or test-runs the model in the same format the
#   browser uses; http.server: serves files locally for testing; time: measures or paces work.
class H(SimpleHTTPRequestHandler):
    extensions_map = {
        '': 'application/octet-stream',
        '.html': 'text/html',
        '.js': 'application/javascript',
        '.wasm': 'application/wasm',
        '.onnx': 'application/octet-stream',
        '.txt': 'text/plain',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
    }

    # guess_type: Correct MIME types (wasm/onnx) for local serving.
    # Why these APIs: os / os.path: handles local paths.
    def guess_type(self, path):
        ext = os.path.splitext(path)[1].lower()
        return self.extensions_map.get(ext, 'application/octet-stream')

    # copyfile: Sends a file at a limited rate and logs timing.
    # Why these APIs: json: reads/writes structured data files; open(): reads/writes a local file; time: measures or
    #   paces work.
    def copyfile(self, src, dst):
        t0 = time.time()
        n = 0
        while True:
            chunk = src.read(65536)
            if not chunk:
                break
            dst.write(chunk)
            n += len(chunk)
            if bw:
                delay = n / bw - (time.time() - t0)
                if delay > 0:
                    time.sleep(delay)
        rec = {'t': time.time(), 'path': self.path.split('?')[0], 'bytes': n, 'status': getattr(self, '_status', 0)}
        with lock:
            with open(log_path, 'a') as f:
                f.write(json.dumps(rec) + '\n')

    # send_response: Remembers the status code for logging.
    def send_response(self, code, message=None):
        self._status = code
        return super().send_response(code, message)

    # log_message: Silences default request logging.
    def log_message(self, fmt, *args):
        pass


os.chdir(root)
ThreadingHTTPServer.allow_reuse_address = True
ThreadingHTTPServer(('127.0.0.1', port), H).serve_forever()
