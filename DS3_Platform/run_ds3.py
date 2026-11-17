"""DS3 - Deep Spectral Signature System.  Run:  python run_ds3.py   (see README.md)"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _preload_tensorflow():
    """Windows: load TensorFlow BEFORE PyQt5.

    PyQt5 ships its own (older) Microsoft C++ runtime DLLs. Once Qt is loaded, Windows reuses them, and
    TensorFlow's DLLs then fail with "A dynamic link library (DLL) initialization routine failed" when the
    Segmentation tab imports TensorFlow later. Importing it first lets it load the runtime it needs.
    Skipped when TensorFlow is not installed or DS3_NO_TF_PRELOAD=1; any error here is ignored
    (the Segmentation tab then shows its own message)."""
    if os.name != "nt" or os.environ.get("DS3_NO_TF_PRELOAD") == "1":
        return
    import importlib.util
    if importlib.util.find_spec("tensorflow") is None:
        return
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if importlib.util.find_spec("tf_keras") is not None:
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")      # same choice as the viewer's seg_tf_ready()
    try:
        import tensorflow  # noqa: F401
        print("[DS3] TensorFlow", tensorflow.__version__, "loaded", flush=True)
    except Exception as e:                                      # noqa: BLE001
        print("[DS3] TensorFlow could not be loaded (Segmentation tab models will not run):",
              str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__, flush=True)


_preload_tensorflow()

from ds3.app import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
