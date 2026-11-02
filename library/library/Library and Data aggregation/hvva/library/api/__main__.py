import argparse

import uvicorn

p = argparse.ArgumentParser(description="Material library API server")
p.add_argument("--host", default="127.0.0.1")
p.add_argument("--port", type=int, default=8000)
p.add_argument("--reload", action="store_true")
a = p.parse_args()
uvicorn.run("hvva.library.api.main:app", host=a.host, port=a.port, reload=a.reload)
