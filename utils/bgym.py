import os
from pathlib import Path

def setup_bgym_env_variables():
    # Prefer MINIWOB_URL if already set; otherwise look under $HOME.
    potential_paths = [
        Path.home() / "miniwob-plusplus/miniwob/html/miniwob",
        Path.home() / "miniwob/html/miniwob",
        Path("/usr/local/miniwob-plusplus/miniwob/html/miniwob"),
    ]

    if "MINIWOB_URL" in os.environ:
        return

    for path in potential_paths:
        if path.exists():
            os.environ["MINIWOB_URL"] = f"file://{path}/"
            print(f"Found MiniWoB at: {path}")
            return

    fallback_path = Path.home() / "miniwob-plusplus/miniwob/html/miniwob"
    os.environ["MINIWOB_URL"] = f"file://{fallback_path}/"
    print(f"Using fallback MiniWoB path: {fallback_path}")
    print("If this does not work, install MiniWoB++ or set MINIWOB_URL.")
