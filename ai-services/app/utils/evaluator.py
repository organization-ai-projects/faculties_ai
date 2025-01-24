import torch
from app.utils.logger import Logger


class Evaluator:
    """
    Handles evaluation logic for AI models.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.logger = Logger.get_logger(__name__)

    def evaluate(self, dataloader):
        """
        Evaluates the model on the given dataloader and returns metrics.
        """
        self.logger.info("Starting evaluation.")
        total_loss = 0.0
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                if "input_ids" not in batch or "labels" not in batch:
                    self.logger.warning(
                        "Batch is missing 'input_ids' or 'labels'. Skipping."
                    )
                    continue
                inputs, labels = batch["input_ids"], batch["labels"]
                outputs = self.model(inputs)
                loss = self.compute_loss(outputs, labels)
                total_loss += loss.item()

                # Metrics calculation (e.g., accuracy)
                preds = torch.argmax(outputs, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0.0
        self.logger.info(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}")
        return avg_loss, {"accuracy": accuracy}

    def compute_loss(self, outputs, labels):
        """
        Computes the loss between model outputs and labels.
        """
        loss_fn = torch.nn.CrossEntropyLoss()
        return loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
