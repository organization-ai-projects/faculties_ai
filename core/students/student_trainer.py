from core.utils.base_trainer import BaseTrainer

class StudentTrainer(BaseTrainer):
    """
    Classe responsable de l'entraînement des étudiants IA.
    """
    def __init__(self, student_model, train_dataset, val_dataset, learning_rate=5e-5, epochs=3, batch_size=4):
        super().__init__(student_model, learning_rate)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self):
        """
        Lance l'entraînement des étudiants IA.
        """
        from torch.utils.data import DataLoader

        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size)
        self.configure_scheduler(train_dataloader, self.epochs)

        for epoch in range(self.epochs):
            avg_loss = self.train_one_epoch(train_dataloader, self.compute_loss)
            print(f"Epoch {epoch + 1}, Training Loss: {avg_loss}")
            self.evaluate(val_dataloader, epoch)

    def evaluate(self, dataloader, epoch):
        """
        Évalue le modèle étudiant.
        """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids']
                labels = batch['labels']

                outputs = self.model(input_ids)
                loss = self.compute_loss(outputs, labels)
                total_loss += loss.item()

        print(f"Validation Loss (Epoch {epoch + 1}): {total_loss / len(dataloader)}")
