"""Patch the WC Psychology description in world_cup_prediction.ipynb summary cell."""
import json

NOTEBOOK = r"C:\Users\jadi_\Documents\github-repos\fifa-world-cup-prediction\notebooks\world_cup_prediction.ipynb"

OLD = "each team's WC win rate vs overall international win rate since 1990. Teams that historically over-perform in World Cups (e.g., Germany, France) get a positive adjustment. Teams that under-perform relative to their Elo (e.g., some high-rated but inexperienced nations) get a penalty."

NEW = "each team's WC win rate vs overall win rate across the **last 3 World Cups (2014, 2018, 2022)**. Using a recent sliding window instead of all history since 1990 prevents old titles (e.g., Brazil 2002) from dominating. Teams with strong recent WC records receive a boost: France (winner 2018, finalist 2022) and Argentina (winner 2022, finalist 2014) benefit most."

with open(NOTEBOOK, "r", encoding="utf-8") as f:
    nb = json.load(f)

patched = 0
for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if OLD in src:
        new_src = src.replace(OLD, NEW)
        cell["source"] = new_src.splitlines(keepends=True)
        patched += 1

print(f"Patched {patched} cell(s).")

with open(NOTEBOOK, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
