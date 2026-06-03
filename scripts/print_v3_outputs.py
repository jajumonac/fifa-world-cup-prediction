import json, sys

with open("notebooks/backtest_historical_wc.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

target_ids = {"v3-diagnostic", "v3-backtest", "v3-compare"}
for cell in nb["cells"]:
    cid = cell.get("id", "")
    if cid in target_ids:
        print(f"=== {cid} ===")
        for out in cell.get("outputs", []):
            otype = out.get("output_type", "")
            if otype == "stream":
                text = out.get("text", [])
            elif otype in ("execute_result", "display_data"):
                text = out.get("data", {}).get("text/plain", [])
            else:
                continue
            if isinstance(text, list):
                text = "".join(text)
            sys.stdout.buffer.write(text.encode("utf-8", "replace"))
            sys.stdout.buffer.write(b"\n")
        print()
