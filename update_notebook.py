import json

with open("notebooks/02_CleanData.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "source" in cell:
        source_str = "".join(cell["source"])
        if "images_to_delete =" in source_str:
            cell["source"] = [
                "import os\n",
                "images_to_delete = [f for f in os.listdir(check_dir) if f.endswith('.png')]"
            ]

with open("notebooks/02_CleanData.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

