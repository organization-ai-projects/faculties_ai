import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

def create_student_model(student_name="custom-gpt2", vocab_size=50257, hidden_size=768, num_layers=6, num_heads=12):
    """
    Crée un modèle étudiant à partir de zéro avec des optimisations modernes.
    """
    print(f"Création du modèle étudiant : {student_name}")
    
    # Configuration du modèle
    config = GPT2Config(
        vocab_size=vocab_size,
        n_embd=hidden_size,
        n_layer=num_layers,
        n_head=num_heads,
        use_cache=True,
    )

    # Création du modèle
    model = GPT2LMHeadModel(config)
    
    print(f"Modèle étudiant '{student_name}' créé avec :")
    print(f"- Taille des embeddings : {hidden_size}")
    print(f"- Nombre de couches : {num_layers}")
    print(f"- Nombre de têtes : {num_heads}")
    print(f"- Taille du vocabulaire : {vocab_size}")
    
    return model
