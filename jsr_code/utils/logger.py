import sys
import os
from pathlib import Path

class Logger(object):
    def __init__(self, out_dir="./", filename="console.log"):
        os.makedirs(out_dir, exist_ok=True)
        self.terminal = sys.stdout
        self.log = (Path(out_dir) / filename).open(mode='a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  
        self.log.flush()

    def flush(self):
        self.log.flush()
        self.terminal.flush()
