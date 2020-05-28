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
        print("Command: ", self.command)
        print("Path: ", self.path)
        par = urlparse.parse_qs(urlparse.urlparse(self.path).query)
        start = par['start'][0]
        end = par['end'][0]
        content = get_next_data(data=data, start=int(start), end=int(end))
        self.wfile.write(content.encode("utf8"))


data = pd.read_csv("raw_data_feature_only.csv", delimiter=',')  # TODO: REPLACE WITH TEST FILE
Handler = GetHandler
httpd = SocketServer.TCPServer(("", PORT), Handler)
httpd.serve_forever()
