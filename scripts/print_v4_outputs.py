import json, sys

with open("notebooks/backtest_historical_wc.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    cid = cell.get("id", "")
    if cid in {"v4-diagnostic", "v4-backtest", "v4-compare"}:
        print(f"=== {cid} ===")
        for out in cell.get("outputs", []):
            otype = out.get("output_type", "")
            text = (out.get("text", []) if otype == "stream"
                    else out.get("data", {}).get("text/plain", []))
            if isinstance(text, list):
                text = "".join(text)
            sys.stdout.buffer.write(text.encode("utf-8", "replace"))
            sys.stdout.buffer.write(b"\n")
        print()
