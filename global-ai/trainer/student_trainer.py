import torch
from torch.optim import AdamW
from transformers import get_scheduler

def train_student_model(
    model, train_dataset, val_dataset, output_dir,
    epochs=3, learning_rate=5e-5, batch_size=4, gradient_accumulation_steps=4, device="cuda"
):
    """
    Entraîne un modèle étudiant avec des optimisations avancées.
    """
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=epochs * len(train_loader))

    scaler = torch.cuda.amp.GradScaler()  # Précision mixte

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels = batch["input_ids"].to(device), batch["labels"].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))

            scaler.scale(loss).backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, perte moyenne : {total_loss / len(train_loader)}")

        # Validation après chaque époque
        validate_model(model, val_loader, device)

    # Sauvegarde finale
    torch.save(model.state_dict(), f"{output_dir}/student_model.pth")
    print(f"Modèle étudiant sauvegardé dans {output_dir}")

def validate_model(model, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch["input_ids"].to(device), batch["labels"].to(device)
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_loss += loss.item()
    print(f"Validation Loss : {total_loss / len(val_loader)}")
