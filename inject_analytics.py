"""
Injects the GoatCounter analytics script tag into every built HTML page.

Run this AFTER `myst build --html` and BEFORE uploading/deploying the
_build/html directory. Safe to run multiple times (skips files that
already contain the tag).
"""

import pathlib

GOATCOUNTER_SNIPPET = (
    '<script data-goatcounter="https://cache-rowan.goatcounter.com/count" '
    'async src="//gc.zgo.at/count.js"></script>'
)

BUILD_DIR = pathlib.Path("_build/html")


def inject_into_file(path: pathlib.Path) -> bool:
    text = path.read_text(encoding="utf-8")

    if "data-goatcounter=" in text:
        return False  # already injected, skip

    if "</body>" not in text:
        return False  # not a full HTML doc, skip safely

    new_text = text.replace("</body>", f"{GOATCOUNTER_SNIPPET}</body>", 1)
    path.write_text(new_text, encoding="utf-8")
    return True


def main():
    if not BUILD_DIR.exists():
        raise SystemExit(f"Build directory not found: {BUILD_DIR.resolve()}")

    html_files = list(BUILD_DIR.rglob("*.html"))
    if not html_files:
        raise SystemExit(f"No .html files found under {BUILD_DIR.resolve()}")

    injected = 0
    for f in html_files:
        if inject_into_file(f):
            injected += 1

    print(f"Scanned {len(html_files)} HTML files, injected into {injected}.")


if __name__ == "__main__":
    main()
