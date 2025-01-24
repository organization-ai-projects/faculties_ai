import json
import os
import time


class AlertSystem:
    """
    Gère l'envoi et le suivi des alertes aux superviseurs humains.
    Les alertes peuvent être sous forme de prompt interactif ou de fichiers de données.
    """

    def __init__(self, alert_folder="alerts"):
        """
        Initialise le système d'alerte.
        :param alert_folder: Répertoire pour stocker les alertes.
        """
        self.alert_folder = alert_folder
        if not os.path.exists(alert_folder):
            os.makedirs(alert_folder)

    def send_alert(self, teacher, responses, alert_type="prompt"):
        """
        Envoie une alerte au superviseur.
        :param teacher: Le professeur concerné par l'alerte.
        :param responses: Les réponses incohérentes des professeurs.
        :param alert_type: Le type d'alerte ('prompt' ou 'file').
        """
        alert_id = self.generate_alert_id()
        alert_data = {
            "teacher": teacher.__class__.__name__,
            "responses": responses,
            "alert_type": alert_type,
            "alert_id": alert_id,
            "status": "pending",
            "timestamp": time.time(),  # Enregistrer l'heure de l'alerte
        }

        if alert_type == "prompt":
            self.create_prompt(
                alert_data
            )  # Créer un prompt interactif pour le superviseur
        elif alert_type == "file":
            self.create_alert_file(
                alert_data
            )  # Sauvegarder les alertes dans un fichier
        else:
            print(f"Type d'alerte inconnu : {alert_type}")

    def generate_alert_id(self):
        """Génère un identifiant unique pour chaque alerte."""
        return f"alert_{int(time.time())}"

    def create_prompt(self, alert_data):
        """
        Crée un prompt interactif pour l'alerte.
        """
        print(f"[ALERTE] Professeur {alert_data['teacher']} - Incohérence détectée.")
        print(f"Réponses incohérentes : {alert_data['responses']}")
        print(f"Statut actuel : {alert_data['status']}")
        print("Veuillez résoudre le problème immédiatement.")
        print("Choisissez l'action à entreprendre :")
        print("1. Résoudre cette alerte")
        print("2. Ignorer cette alerte")

    def create_alert_file(self, alert_data):
        """
        Sauvegarde les détails de l'alerte dans un fichier JSON.
        """
        alert_path = os.path.join(self.alert_folder, f"{alert_data['alert_id']}.json")
        with open(alert_path, "w") as file:
            json.dump(alert_data, file, indent=4)
        print(f"Alerte sauvegardée dans {alert_path}.")

    def archive_alert(self, alert_id):
        """
        Déplace l'alerte résolue dans un sous-dossier 'resolved'.
        :param alert_id: L'identifiant de l'alerte à archiver.
        """
        resolved_folder = os.path.join(self.alert_folder, "resolved")
        if not os.path.exists(resolved_folder):
            os.makedirs(resolved_folder)

        alert_path = os.path.join(self.alert_folder, f"{alert_id}.json")
        if os.path.exists(alert_path):
            os.rename(alert_path, os.path.join(resolved_folder, f"{alert_id}.json"))
            print(f"Alerte {alert_id} archivée dans {resolved_folder}.")
        else:
            print(f"Alerte {alert_id} non trouvée.")
