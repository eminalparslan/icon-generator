file_path = "./outlined_rasterized/train/metadata.csv"

with open(file_path, "r") as f:
    lines = f.readlines()

with open(file_path, "w") as f:
    for line in lines:
        parts = line.split(",")
        new_parts = [parts[0]] + [p.replace("_", " ").strip() for p in parts[1:]]
        f.write(",".join(new_parts))
        f.write("\n")
