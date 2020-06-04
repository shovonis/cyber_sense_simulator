# !/usr/bin/env python3
import http.server as SimpleHTTPServer
import socketserver as SocketServer
import pandas as pd


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
        content = get_next_data(data=data)
        self.wfile.write(content.encode("utf8"))


def get_next_data(data):
    global start_index
    start, end = get_start_end_index()
    start_index += DATA_OFFSET
    result = data.iloc[start:end]
    return result.to_json()


def get_start_end_index():
    start = start_index
    end = start_index + DATA_OFFSET
    # Check the outer bound
    if end > number_of_rows:
        end = number_of_rows
    if start > number_of_rows:
        start = number_of_rows

    return start, end


if __name__ == '__main__':
    PORT = 8800
    DATA_OFFSET = 61  # In order to look back 60 data we need 61 examples
    data = pd.read_csv("test_data_set.csv", delimiter=',')  # TODO: REPLACE WITH TEST FILE
    start_index = 0
    number_of_rows = len(data)

    # Run Server
    Handler = GetHandler
    httpd = SocketServer.TCPServer(("", PORT), Handler)
    print("Data Simulator Started...")
    httpd.serve_forever()
