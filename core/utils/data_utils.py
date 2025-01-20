import json
from pathlib import Path

def load_dataset(dataset_path):
    """
    Charge un dataset structuré depuis un fichier JSON.
    """
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset introuvable : {dataset_path}")

    with open(path, "r") as f:
        dataset = json.load(f)
    
    # Retourne le dataset sous forme de liste de dictionnaires
    return [{"input": item["text"]} for item in dataset]

def preprocess_data(raw_data_dir, output_file):
    """
    Prétraite les données brutes depuis raw_data/ et les transforme en dataset JSON structuré.
    """
    raw_data_dir = Path(raw_data_dir)
    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Le dossier {raw_data_dir} est introuvable.")

    dataset = []

    # Charger les fichiers JSON
    for file in raw_data_dir.glob("*.json"):
        with open(file, "r") as f:
            data = json.load(f)
            dataset.extend(data)

    # Charger les fichiers TXT
    for file in raw_data_dir.glob("*.txt"):
        with open(file, "r") as f:
            lines = f.readlines()
            dataset.extend({"input": line.strip()} for line in lines)

    # Sauvegarder dans le fichier de sortie
    output_path = Path(output_file)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=4)

    print(f"Données prétraitées et sauvegardées dans {output_path}.")

def load_splits(split_dir):
    """
    Charge les splits (train/test/val) depuis un répertoire donné.
    """
    split_dir = Path(split_dir)
    if not split_dir.exists():
        raise FileNotFoundError(f"Le dossier {split_dir} est introuvable.")

    splits = {}
    for split in ["train", "val", "test"]:
        split_file = split_dir / f"{split}.json"
        if split_file.exists():
            with open(split_file, "r") as f:
                splits[split] = json.load(f)
        else:
            print(f"Attention : Le fichier {split_file} est manquant.")

    return splits
