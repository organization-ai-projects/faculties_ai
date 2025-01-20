import torch
import torch.nn.functional as F

def distill_knowledge(teacher, student, train_dataset, val_dataset, output_dir, alpha=0.5, temperature=2.0):
    """
    Implémente la distillation de connaissances entre un professeur et un étudiant.
    """
    print("Distillation des connaissances en cours...")
    
    teacher.eval()  # Le professeur reste en mode évaluation
    optimizer = torch.optim.AdamW(student.parameters(), lr=5e-5)

    for epoch in range(3):  # Boucle simple pour l'exemple
        for batch in train_dataset:
            inputs = batch["input_ids"].to(student.device)
            labels = batch["labels"].to(student.device)

            # Sortie du professeur (logits)
            with torch.no_grad():
                teacher_logits = teacher(inputs).logits / temperature

            # Sortie de l'étudiant
            student_logits = student(inputs).logits / temperature

            # Perte de distillation
            loss = alpha * F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(teacher_logits, dim=-1),
                reduction="batchmean"
            ) + (1 - alpha) * F.cross_entropy(student_logits, labels)

            # Optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1} terminé, perte : {loss.item()}")

    # Sauvegarder l'étudiant
    student.save_pretrained(output_dir)
    print(f"Étudiant entraîné et sauvegardé dans : {output_dir}")
