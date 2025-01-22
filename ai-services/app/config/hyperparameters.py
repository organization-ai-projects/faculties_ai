HYPERPARAMETERS = {
    "teachers": {
        "learning_rate": 5e-5,
        "batch_size": 4,
        "epochs": 3,
        "save_dir": "app/models/teachers/"
    },
    "students": {
        "learning_rate": 3e-5,
        "batch_size": 8,
        "epochs": 5,
        "save_dir": "app/models/students/"
    },
    "dataset_path": "data/processed/training_dataset.json",
    "logging_dir": "logs/",
    "models": {
        "general_text_generation": ["gpt2-medium", "EleutherAI/gpt-neo-125M"],
        "text_analysis": ["bert-base-uncased", "distilbert-base-uncased"],
        "multilingual_tasks": ["facebook/m2m100_418M"],
        "code_generation": ["Salesforce/codegen-350M-multi"]
    }
}
