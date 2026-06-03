import json, sys

with open("notebooks/world_cup_prediction.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

# Print outputs from key cells: wc_uplift cell + champion results cell
grab_next = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell.get("source", []))

    if "window_start" in src and "wc_adj" in src:
        print("=== WC Uplift cell ===")
        grab_next = 1

    elif grab_next > 0 and "team_elos" in src and "form_adj" in src:
        print("=== Enriched Elo cell ===")
        grab_next = 2

    elif "Champion" in src and "N_SIMS" in src and "positions" in src:
        print("=== Results cell ===")
        grab_next = 3

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
        if grab_next > 0:
            sys.stdout.buffer.write(text.encode("utf-8", "replace"))
            sys.stdout.buffer.write(b"\n")

    if grab_next > 0:
        grab_next -= 1
