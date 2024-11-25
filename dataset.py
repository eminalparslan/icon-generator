from datasets import load_dataset

dataset = load_dataset("imagefolder", data_dir="outlined_rasterized/", split="train")
print(dataset[100]["text"])
