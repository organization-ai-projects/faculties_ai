# core/teachers/teacher_interaction.py

class TeacherInteraction:
    """
    Classe responsable de l'interaction entre les professeurs IA. 
    Les professeurs peuvent s'échanger des connaissances, s'évaluer mutuellement, et améliorer les étudiants ensemble.
    """

    def __init__(self, teachers, alert_system):
        """
        Initialise l'interaction entre un ensemble de professeurs.
        :param teachers: liste d'instances des professeurs (de type TeacherBase).
        :param alert_system: système d'alerte pour détecter les incohérences ou incertitudes
        """
        self.teachers = teachers
        self.alert_system = alert_system

    def share_knowledge(self):
        """
        Simule un échange de connaissances entre les professeurs et les étudiants.
        Chaque professeur apprend de ses pairs et des retours des étudiants, et ajuste ses paramètres ou sa stratégie en conséquence.
        """
        print("Échange de connaissances entre professeurs et retour des étudiants...")

        for i, teacher in enumerate(self.teachers):
            # Échange de connaissances entre professeurs
            teacher.receive_knowledge(self.teachers[(i + 1) % len(self.teachers)])

            # Interaction avec les étudiants pour ajuster les stratégies
            student_feedback = teacher.receive_student_feedback()  # Récupère les feedbacks des étudiants
            teacher.learn_from_students(student_feedback)  # Ajuste ses paramètres en fonction des retours

    def evaluate_teachers(self):
        """
        Évalue l'efficacité de chaque professeur, soit en fonction de leurs performances sur des jeux de données,
        soit en fonction des performances des étudiants sous leur tutelle.
        """
        print("Évaluation des professeurs...")

        for teacher in self.teachers:
            performance = teacher.evaluate_performance()
            print(f"Performance du professeur {teacher.__class__.__name__} : {performance}")

    def co_train_students(self, student_model):
        """
        Permet à plusieurs professeurs d'entraîner un étudiant en utilisant leurs connaissances combinées.
        Cela peut aussi inclure des mécanismes où les professeurs se synchronisent pour co-apprendre.
        """
        print("Co-entraînement des étudiants par les professeurs...")

        for teacher in self.teachers:
            teacher.interact_with_students(student_model)

    def detect_incoherence_and_alert(self, student_model):
        """
        Détecte les incohérences dans les réponses des professeurs basées sur les retours des étudiants,
        puis déclenche une alerte au superviseur humain si nécessaire.
        """
        print("Vérification des incohérences dans les réponses des professeurs...")

        # Comparer les réponses des professeurs et détecter les incohérences
        for teacher in self.teachers:
            responses = teacher.get_responses(student_model)  # Récupère les réponses de chaque professeur
            if self.is_incoherent(responses):
                print(f"Incohérence détectée dans les réponses du professeur {teacher.__class__.__name__}. Alerte envoyée...")
                self.alert_system.send_alert(teacher, responses)  # Envoi de l'alerte au superviseur

    def is_incoherent(self, responses):
        """
        Vérifie si les réponses des professeurs sont incohérentes.
        Cela peut être basé sur des critères comme des divergences dans les conseils ou des incohérences dans les résultats.
        """
        # Exemple simplifié : si les réponses ne sont pas identiques, c'est une incohérence
        return len(set(responses)) > 1
