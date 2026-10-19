# Simulate iSH python http.server: no wasm/onnx/mjs MIME types, single-threaded, throttled bandwidth.
import http.server, sys, time, os
port = int(sys.argv[1]); bw = float(sys.argv[2]) * 1e6 if len(sys.argv) > 2 else 0; threaded = len(sys.argv) > 3 and sys.argv[3] == 't'
class H(http.server.SimpleHTTPRequestHandler):
    extensions_map = {'': 'application/octet-stream', '.html': 'text/html', '.js': 'application/javascript', '.txt': 'text/plain', '.png': 'image/png', '.jpg': 'image/jpeg'}
    def guess_type(self, path):
        ext = os.path.splitext(path)[1].lower(); return self.extensions_map.get(ext, 'application/octet-stream')
    def copyfile(self, src, dst):
        if not bw: return super().copyfile(src, dst)
        t0 = time.time(); n = 0
        while True:
            b = src.read(65536)
            if not b: break
            dst.write(b); n += len(b)
            d = n / bw - (time.time() - t0)
            if d > 0: time.sleep(d)
    def log_message(self, *a): pass
os.chdir(os.environ.get('SITE', '.'))  # folder that contains crowd/ (set SITE=...)
S = http.server.ThreadingHTTPServer if threaded else http.server.HTTPServer
S(('127.0.0.1', port), H).serve_forever()
