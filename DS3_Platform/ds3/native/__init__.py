"""Native PyQt5 versions of the DS3 web modules (Library (main) and Image).

The pages talk to the same FastAPI backend (``material_library/backend``) as the web pages did, running in its own
process on 127.0.0.1 (see ``ds3/web_pages.py``), but they are drawn with Qt widgets: no QtWebEngine, no JavaScript.
"""
