import ibapi
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
import threading
import time

class IBApi(EWrapper, EClient):

    def __init__(self):
        EClient.__init__(self, self)

    def error(self, reqId, errorCode, errorString):
        print("Error:", reqId, errorCode, errorString)

    def nextValidId(self, orderId: int):
        print("Connected with order ID:", orderId)

class Bot:
    def __init__(self):
        self.ib = IBApi()
        self.ib.connect("127.0.0.1", 7497, 1)  # Host, port, client ID
        self.thread = threading.Thread(target=self.run_loop, daemon=True)
        self.thread.start()
        time.sleep(1)  # Wait a moment for connection to establish

    def run_loop(self):
        self.ib.run()

print("hello")
bot = Bot()
print("hello")

# Keep the script running to maintain the connection
time.sleep(5)  # Extend as needed, or replace with other main program logic
