# !/usr/bin/env python3
import http.server as SimpleHTTPServer
import socketserver as SocketServer
import urllib.parse as urlparse

import pandas as pd

PORT = 8000


# http://localhost:8000/NeuLogAPI?start=0&end=60
def get_next_data(data, start=0, end=None):
    print("Data Start: ", start)
    print("Data End: ", end)
    result = data.iloc[start:end]
    return result.to_json()


class GetHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

    def do_GET(self):
        self._set_headers()
        print("Get Request Received...")
        print("Command: ", self.command)
        print("Path: ", self.path)
        par = urlparse.parse_qs(urlparse.urlparse(self.path).query)
        start = par['start'][0]
        end = par['end'][0]
        content = get_next_data(data=data, start=int(start), end=int(end))
        self.wfile.write(content.encode("utf8"))


if __name__ == '__main__':
    data = pd.read_csv("test_data_set.csv", delimiter=',')  # TODO: REPLACE WITH TEST FILE
    Handler = GetHandler
    httpd = SocketServer.TCPServer(("", PORT), Handler)
    print("Data Simulator Started...")
    httpd.serve_forever()
