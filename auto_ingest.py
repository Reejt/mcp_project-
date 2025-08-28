import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from mcp_server import MCPServer
import os

WATCH_DIR = './mcp_storage'

class IngestHandler(FileSystemEventHandler):
    def __init__(self, mcp_server):
        self.mcp_server = mcp_server
    def on_created(self, event):
        if not event.is_directory:
            ext = os.path.splitext(event.src_path)[1].lower()
            if ext in ['.txt', '.pdf', '.md', '.json', '.xml', '.html', '.csv']:
                print(f'[AUTO-INGEST] New file detected: {event.src_path}')
                result = self.mcp_server.ingest_file(event.src_path)
                print(f'[AUTO-INGEST] Ingest result: {result}')

def start_auto_ingest():
    mcp_server = MCPServer()
    event_handler = IngestHandler(mcp_server)
    observer = Observer()
    observer.schedule(event_handler, WATCH_DIR, recursive=False)
    observer.start()
    print(f'[AUTO-INGEST] Watching {WATCH_DIR} for new files...')
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == '__main__':
    start_auto_ingest()
