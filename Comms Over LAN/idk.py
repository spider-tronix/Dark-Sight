from os import curdir
from os.path import join as pjoin

from http.server import BaseHTTPRequestHandler, HTTPServer

class StoreHandler(BaseHTTPRequestHandler):
    store_path = pjoin(curdir, 'store.json')

    def do_GET(self):
        if self.path == '/store.json':
            with open(self.store_path) as fh:
                self.send_response(200)
                self.send_header('Content-type', 'text/json')
                self.end_headers()
                self.wfile.write(fh.read().encode())

    def do_POST(self):
        if self.path == '/store.json':
            length = self.headers['content-length']
            data = self.rfile.read(int(length))

            with open(self.store_path, 'w') as fh:
                fh.write(data.decode())

            self.send_response(200)


server = HTTPServer(('0.0.0.0', 8080), StoreHandler)
server.serve_forever()