import torch
from transformers import TrainingArguments

def configure_training_args(model_name, hyperparameters):
    """Configure les arguments d'entraînement pour un modèle spécifique."""
    fp16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7
    return TrainingArguments(
        output_dir=f"models/{model_name}",
        num_train_epochs=hyperparameters.get("epochs", 3),
        per_device_train_batch_size=hyperparameters.get("batch_size", 4),
        gradient_accumulation_steps=hyperparameters.get("gradient_accumulation_steps", 4),
        fp16=fp16,
        learning_rate=hyperparameters.get("learning_rate", 5e-5),
        logging_dir=f"logs/{model_name}",
        save_steps=500,
        save_total_limit=2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        logging_steps=50,
    )
