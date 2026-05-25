import kagglehub

# Download dataset
path = kagglehub.dataset_download(
    "anilsandhii/starfruit-disease"
)

print("Dataset path:", path)