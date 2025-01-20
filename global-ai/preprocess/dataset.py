import json

def preprocess_raw_to_processed(raw_file, processed_file, structured_data):
    # Charger les données brutes depuis un fichier
    with open(raw_file, "r", encoding="utf-8") as f:
        raw_data = [line.strip() for line in f if line.strip()]  # Supprimer les lignes vides

    # Prétraitement des données brutes
    processed_data = []
    for i in range(0, len(raw_data), 2):
        input_line = raw_data[i]
        output_line = raw_data[i + 1] if i + 1 < len(raw_data) else ""
        processed_data.append({"input": input_line, "output": output_line})

    # Ajouter les données structurées
    processed_data.extend(structured_data)

    # Sauvegarde dans un fichier
    with open(processed_file, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    # Définir les chemins des fichiers
    raw_file = "data/raw/code_snippets.txt"
    processed_file = "data/processed/training_dataset.json"

    # Définir les données structurées
    structured_data = [
        {
            "input": "Créer une fonction Python pour additionner deux nombres.",
            "output": "def add(a, b):\n    return a + b"
        },
        {
            "input": "Écrire une fonction Python pour calculer la factorielle.",
            "output": "def factorial(n):\n    return 1 if n == 0 else n * factorial(n-1)"
        }
    ]

    # Exécuter le prétraitement
    preprocess_raw_to_processed(raw_file, processed_file, structured_data)
