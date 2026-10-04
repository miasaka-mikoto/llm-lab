"""Convenience launcher for the LLM Lab desktop application."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llmlab.gui import GuiAdapter, run  # noqa: E402


def main() -> None:
    # The SQLite store is optional at import time, but the normal desktop
    # launcher persists experiments in the project directory automatically.
    try:
        from llmlab.storage import SQLiteStore

        store = SQLiteStore(ROOT / "llmlab.sqlite3")
        run(adapter=GuiAdapter(store=store, project_dir=ROOT), project_dir=ROOT)
    except Exception:
        # A minimal Python installation can still open the standalone local
        # demo if storage initialization is unavailable.
        run(project_dir=ROOT)


if __name__ == "__main__":
    main()
