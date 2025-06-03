import spacy
from spacy.training import Example
import random
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from datetime import datetime

class DataGenerator:
    """Générateur de données d'entraînement à partir d'annuaire et de contextes"""
    
    def __init__(self):
        self.person_names = []
        self.org_names = []
        self.location_names = []
        self.context_templates = []
        self.generated_data = []
    
    def load_directory_file(self, file_path: str, entity_type: str = "PERSON"):
        """Charge un fichier annuaire"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                names = [line.strip() for line in f if line.strip()]
            
            if entity_type == "PERSON":
                self.person_names.extend(names)
            elif entity_type == "ORG":
                self.org_names.extend(names)
            elif entity_type == "LOC":
                self.location_names.extend(names)
            
            return len(names)
        except Exception as e:
            raise Exception(f"Erreur lors du chargement de l'annuaire : {str(e)}")
    
    def load_context_templates(self, file_path: str):
        """Charge des phrases contextuelles avec des placeholders"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                templates = [line.strip() for line in f if line.strip()]
            
            self.context_templates = templates
            return len(templates)
        except Exception as e:
            raise Exception(f"Erreur lors du chargement des contextes : {str(e)}")
    
    def generate_training_data(self, num_samples: int = 1000) -> List[Tuple[str, Dict]]:
        """Génère des données d'entraînement annotées"""
        if not self.context_templates:
            raise Exception("Aucun template de contexte chargé")
        
        training_data = []
        
        for _ in range(num_samples):
            template = random.choice(self.context_templates)
            text = template
            entities = []
            
            # Traitement séquentiel des placeholders
            current_text = text
            current_entities = []
            
            # Traitement des PERSON
            while "{PERSON}" in current_text and self.person_names:
                person = random.choice(self.person_names)
                pos = current_text.find("{PERSON}")
                if pos != -1:
                    # Ajout de l'entité avec la position actuelle
                    current_entities.append((pos, pos + len(person), "PERSON"))
                    # Remplacement dans le texte
                    current_text = current_text.replace("{PERSON}", person, 1)
            
            # Traitement des ORG
            while "{ORG}" in current_text and self.org_names:
                org = random.choice(self.org_names)
                pos = current_text.find("{ORG}")
                if pos != -1:
                    current_entities.append((pos, pos + len(org), "ORG"))
                    current_text = current_text.replace("{ORG}", org, 1)
            
            # Traitement des LOC
            while "{LOC}" in current_text and self.location_names:
                loc = random.choice(self.location_names)
                pos = current_text.find("{LOC}")
                if pos != -1:
                    current_entities.append((pos, pos + len(loc), "LOC"))
                    current_text = current_text.replace("{LOC}", loc, 1)
            
            # Validation : pas de placeholders restants
            if not any(placeholder in current_text for placeholder in ["{PERSON}", "{ORG}", "{LOC}"]):
                if current_entities:
                    # Tri des entités par position
                    current_entities.sort(key=lambda x: x[0])
                    annotation = {"entities": current_entities}
                    training_data.append((current_text, annotation))
        
        self.generated_data = training_data
        return training_data
    
    def save_training_data(self, output_path: str):
        """Sauvegarde les données d'entraînement au format JSON"""
        data_for_export = []
        for text, annotation in self.generated_data:
            data_for_export.append({
                "text": text,
                "entities": annotation["entities"]
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_for_export, f, ensure_ascii=False, indent=2)

class SpacyFineTuner:
    """Fine-tuner pour modèles spaCy simplifié"""
    
    def __init__(self, base_model: str = "fr_core_news_md"):
        self.base_model = base_model
        self.nlp = None
        self.training_data = []
    
    def load_base_model(self):
        """Charge le modèle de base"""
        try:
            self.nlp = spacy.load(self.base_model)
            
            if "ner" not in self.nlp.pipe_names:
                ner = self.nlp.add_pipe("ner")
            else:
                ner = self.nlp.get_pipe("ner")
            
            return True
        except OSError:
            raise Exception(f"Modèle {self.base_model} non trouvé. Installez-le avec : python -m spacy download {self.base_model}")
    
    def prepare_training_data(self, training_data: List[Tuple[str, Dict]]):
        """Prépare les données pour l'entraînement spaCy"""
        if not self.nlp:
            self.load_base_model()
        
        examples = []
        ner = self.nlp.get_pipe("ner")
        
        # Ajout des nouvelles labels
        for text, annotation in training_data:
            for start, end, label in annotation["entities"]:
                ner.add_label(label)
        
        # Création des exemples d'entraînement
        for text, annotation in training_data:
            doc = self.nlp.make_doc(text)
            example = Example.from_dict(doc, annotation)
            examples.append(example)
        
        self.training_data = examples
        return len(examples)
    
    def fine_tune(self, n_iter: int = 30, drop_rate: float = 0.2, progress_callback=None):
        """Fine-tune le modèle simplifié"""
        if not self.training_data:
            raise Exception("Aucune donnée d'entraînement préparée")
        
        # Désactivation des autres pipes
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
        
        with self.nlp.disable_pipes(*other_pipes):
            optimizer = self.nlp.resume_training()
            
            # Entraînement simple par batches
            for iteration in range(n_iter):
                if progress_callback:
                    progress_callback(iteration, n_iter)
                
                random.shuffle(self.training_data)
                losses = {}
                
                # Traitement par petits batches
                batch_size = min(8, len(self.training_data))
                for i in range(0, len(self.training_data), batch_size):
                    batch = self.training_data[i:i + batch_size]
                    self.nlp.update(batch, drop=drop_rate, losses=losses, sgd=optimizer)
                
                # Log périodique
                if iteration % 5 == 0:
                    print(f"Iteration {iteration}, Losses: {losses}")
        
        return True
    
    def save_model(self, output_path: str):
        """Sauvegarde le modèle fine-tuné"""
        if not self.nlp:
            raise Exception("Aucun modèle à sauvegarder")
        
        Path(output_path).mkdir(parents=True, exist_ok=True)
        self.nlp.to_disk(output_path)
        
        # Métadonnées
        metadata = {
            "base_model": self.base_model,
            "training_samples": len(self.training_data),
            "fine_tuned_date": datetime.now().isoformat(),
            "entities_trained": list(self.nlp.get_pipe("ner").labels)
        }
        
        with open(Path(output_path) / "fine_tune_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

class FineTuningGUI:
    """Interface graphique pour le fine-tuning - Version thread-safe simplifiée"""
    
    def __init__(self, parent_window=None):
        self.window = tk.Toplevel(parent_window) if parent_window else tk.Tk()
        self.window.title("Fine-tuning du modèle spaCy")
        self.window.geometry("700x800")
        
        self.data_generator = DataGenerator()
        self.fine_tuner = SpacyFineTuner()
        
        # Variables
        self.directory_file = tk.StringVar()
        self.context_file = tk.StringVar()
        self.output_model_path = tk.StringVar()
        self.status_text = tk.StringVar(value="Prêt pour le fine-tuning")
        self.entity_type = tk.StringVar(value="PERSON")
        
        # Variables pour les paramètres
        self.num_samples = tk.IntVar(value=100)  # Réduit pour les tests
        self.n_iterations = tk.IntVar(value=10)  # Réduit pour les tests
        
        self.create_widgets()
    
    def create_widgets(self):
        """Création de l'interface"""
        
        # Titre
        ttk.Label(self.window, text="Fine-tuning du modèle spaCy", 
                 font=('Arial', 16, 'bold')).pack(pady=15)
        
        # Section 1: Fichier annuaire
        directory_frame = ttk.LabelFrame(self.window, text="1. Fichier annuaire", padding=10)
        directory_frame.pack(fill='x', padx=20, pady=10)
        
        ttk.Label(directory_frame, text="Sélectionnez votre fichier annuaire (un nom par ligne) :").pack(anchor='w')
        
        dir_select_frame = ttk.Frame(directory_frame)
        dir_select_frame.pack(fill='x', pady=5)
        
        self.directory_entry = ttk.Entry(dir_select_frame, textvariable=self.directory_file, 
                 state='readonly', width=50)
        self.directory_entry.pack(side='left', fill='x', expand=True)
        ttk.Button(dir_select_frame, text="Parcourir...", 
                  command=self.select_directory_file).pack(side='right', padx=(5,0))
        
        # Type d'entité
        ttk.Label(directory_frame, text="Type d'entités dans ce fichier :").pack(anchor='w', pady=(10,0))
        entity_frame = ttk.Frame(directory_frame)
        entity_frame.pack(fill='x')
        
        ttk.Radiobutton(entity_frame, text="Personnes", variable=self.entity_type, 
                       value="PERSON").pack(side='left')
        ttk.Radiobutton(entity_frame, text="Organisations", variable=self.entity_type, 
                       value="ORG").pack(side='left', padx=(20,0))
        ttk.Radiobutton(entity_frame, text="Lieux", variable=self.entity_type, 
                       value="LOC").pack(side='left', padx=(20,0))
        
        ttk.Button(directory_frame, text="Charger l'annuaire", 
                  command=self.load_directory).pack(pady=10)
        
        # Section 2: Phrases contextuelles
        context_frame = ttk.LabelFrame(self.window, text="2. Phrases contextuelles", padding=10)
        context_frame.pack(fill='x', padx=20, pady=10)
        
        context_help = """Créez des phrases avec des placeholders :
{PERSON} travaille chez {ORG} depuis janvier.
L'entreprise {ORG} est basée à {LOC}.
{PERSON} de {LOC} contacte {PERSON} par email."""
        
        ttk.Label(context_frame, text=context_help, justify='left').pack(anchor='w')
        
        ctx_select_frame = ttk.Frame(context_frame)
        ctx_select_frame.pack(fill='x', pady=5)
        
        self.context_entry = ttk.Entry(ctx_select_frame, textvariable=self.context_file, 
                 state='readonly', width=50)
        self.context_entry.pack(side='left', fill='x', expand=True)
        ttk.Button(ctx_select_frame, text="Parcourir...", 
                  command=self.select_context_file).pack(side='right', padx=(5,0))
        
        ttk.Button(context_frame, text="Charger les contextes", 
                  command=self.load_contexts).pack(pady=10)
        
        # Section 3: Génération et entraînement
        training_frame = ttk.LabelFrame(self.window, text="3. Entraînement", padding=10)
        training_frame.pack(fill='x', padx=20, pady=10)
        
        # Paramètres
        params_frame = ttk.Frame(training_frame)
        params_frame.pack(fill='x', pady=5)
        
        ttk.Label(params_frame, text="Nombre d'exemples à générer :").pack(side='left')
        ttk.Entry(params_frame, textvariable=self.num_samples, width=10).pack(side='left', padx=(5,20))
        
        ttk.Label(params_frame, text="Itérations d'entraînement :").pack(side='left')
        ttk.Entry(params_frame, textvariable=self.n_iterations, width=10).pack(side='left', padx=5)
        
        # Chemin de sortie du modèle
        output_frame = ttk.Frame(training_frame)
        output_frame.pack(fill='x', pady=10)
        
        ttk.Label(output_frame, text="Dossier de sauvegarde du modèle :").pack(anchor='w')
        
        output_select_frame = ttk.Frame(output_frame)
        output_select_frame.pack(fill='x', pady=5)
        
        self.output_entry = ttk.Entry(output_select_frame, textvariable=self.output_model_path, 
                 state='readonly', width=50)
        self.output_entry.pack(side='left', fill='x', expand=True)
        ttk.Button(output_select_frame, text="Choisir dossier", 
                  command=self.select_output_path).pack(side='right', padx=(5,0))
        
        # Boutons d'action
        buttons_frame = ttk.Frame(training_frame)
        buttons_frame.pack(fill='x', pady=10)
        
        self.generate_button = ttk.Button(buttons_frame, text="1. Générer données", 
                  command=self.generate_data)
        self.generate_button.pack(side='left', padx=(0,10))
        
        self.train_button = ttk.Button(buttons_frame, text="2. Lancer fine-tuning", 
                  command=self.start_training)
        self.train_button.pack(side='left')
        
        # Barre de progression
        self.progress = ttk.Progressbar(self.window, mode='determinate')
        self.progress.pack(fill='x', padx=20, pady=5)
        
        # Statut
        status_frame = ttk.Frame(self.window)
        status_frame.pack(fill='x', padx=20, pady=10)
        
        ttk.Label(status_frame, text="Statut :").pack(side='left')
        ttk.Label(status_frame, textvariable=self.status_text, 
                 foreground='blue').pack(side='left', padx=(5,0))
        
        # Instructions
        help_frame = ttk.LabelFrame(self.window, text="Instructions", padding=10)
        help_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        final_help = """1. Chargez votre fichier annuaire (noms, organisations, ou lieux)
2. Chargez vos phrases contextuelles avec placeholders
3. Générez les données d'entraînement (commencez petit : 100 exemples)
4. Lancez le fine-tuning (commencez avec 10 itérations)
5. Le modèle sera sauvegardé dans le dossier choisi"""
        
        ttk.Label(help_frame, text=final_help, justify='left').pack()
    
    def select_directory_file(self):
        file_path = filedialog.askopenfilename(
            title="Fichier annuaire",
            filetypes=[("Fichiers texte", "*.txt"), ("Tous", "*.*")]
        )
        if file_path:
            self.directory_file.set(file_path)
    
    def select_context_file(self):
        file_path = filedialog.askopenfilename(
            title="Fichier phrases contextuelles",
            filetypes=[("Fichiers texte", "*.txt"), ("Tous", "*.*")]
        )
        if file_path:
            self.context_file.set(file_path)
    
    def select_output_path(self):
        folder_path = filedialog.askdirectory(title="Dossier pour le modèle fine-tuné")
        if folder_path:
            model_path = Path(folder_path) / "modele_finetuned"
            self.output_model_path.set(str(model_path))
    
    def load_directory(self):
        if not self.directory_file.get():
            messagebox.showerror("Erreur", "Sélectionnez d'abord un fichier annuaire")
            return
        
        try:
            count = self.data_generator.load_directory_file(
                self.directory_file.get(), 
                self.entity_type.get()
            )
            self.status_text.set(f"Annuaire chargé : {count} entrées ({self.entity_type.get()})")
            messagebox.showinfo("Succès", f"{count} entrées chargées")
        except Exception as e:
            messagebox.showerror("Erreur", str(e))
    
    def load_contexts(self):
        if not self.context_file.get():
            messagebox.showerror("Erreur", "Sélectionnez d'abord un fichier de contextes")
            return
        
        try:
            count = self.data_generator.load_context_templates(self.context_file.get())
            self.status_text.set(f"Contextes chargés : {count} templates")
            messagebox.showinfo("Succès", f"{count} templates chargés")
        except Exception as e:
            messagebox.showerror("Erreur", str(e))
    
    def generate_data(self):
        """Génération des données - version simplifiée sans threading"""
        try:
            if not self.data_generator.context_templates:
                messagebox.showerror("Erreur", "Chargez d'abord les contextes")
                return
            
            if not any([self.data_generator.person_names, 
                       self.data_generator.org_names, 
                       self.data_generator.location_names]):
                messagebox.showerror("Erreur", "Chargez d'abord un annuaire")
                return
            
            self.generate_button.config(state='disabled')
            self.status_text.set("Génération des données en cours...")
            self.window.update()
            
            # Génération directe (sans thread pour éviter les problèmes)
            training_data = self.data_generator.generate_training_data(self.num_samples.get())
            
            self.status_text.set(f"Données générées : {len(training_data)} exemples")
            self.generate_button.config(state='normal')
            
            messagebox.showinfo("Succès", f"{len(training_data)} exemples d'entraînement générés")
            
            # Sauvegarde optionnelle
            if self.output_model_path.get():
                try:
                    save_path = Path(self.output_model_path.get()).parent / "training_data.json"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    self.data_generator.save_training_data(str(save_path))
                    print(f"Données sauvegardées dans : {save_path}")
                except Exception as save_error:
                    print(f"Erreur sauvegarde optionnelle: {save_error}")
            
        except Exception as e:
            self.generate_button.config(state='normal')
            self.status_text.set("Erreur lors de la génération")
            messagebox.showerror("Erreur", f"Erreur lors de la génération : {str(e)}")
    
    def start_training(self):
        """Démarrage de l'entraînement - version simplifiée"""
        if not self.data_generator.generated_data:
            messagebox.showerror("Erreur", "Générez d'abord les données d'entraînement")
            return
        
        if not self.output_model_path.get():
            messagebox.showerror("Erreur", "Choisissez un dossier de sauvegarde")
            return
        
        # Désactivation des boutons
        self.generate_button.config(state='disabled')
        self.train_button.config(state='disabled')
        
        try:
            # Préparation
            self.status_text.set("Préparation du modèle...")
            self.progress.config(value=10)
            self.window.update()
            
            self.fine_tuner.load_base_model()
            
            # Préparation des données
            self.status_text.set("Préparation des données...")
            self.progress.config(value=20)
            self.window.update()
            
            self.fine_tuner.prepare_training_data(self.data_generator.generated_data)
            
            # Callback pour la progression
            def update_progress(iteration, total):
                progress = 20 + int((iteration / total) * 60)  # 20-80%
                self.progress.config(value=progress)
                self.status_text.set(f"Entraînement... {iteration+1}/{total}")
                self.window.update()
            
            # Entraînement
            self.status_text.set("Fine-tuning en cours...")
            self.progress.config(value=25)
            self.window.update()
            
            self.fine_tuner.fine_tune(
                n_iter=self.n_iterations.get(),
                progress_callback=update_progress
            )
            
            # Sauvegarde
            self.status_text.set("Sauvegarde du modèle...")
            self.progress.config(value=90)
            self.window.update()
            
            self.fine_tuner.save_model(self.output_model_path.get())
            
            # Succès
            self.status_text.set("Terminé avec succès !")
            self.progress.config(value=100)
            
            success_msg = f"""Fine-tuning terminé avec succès !

Modèle sauvegardé dans :
{self.output_model_path.get()}

Pour l'utiliser dans l'application principale :
1. Retournez à l'onglet "Pseudonymisation"
2. Sélectionnez "Modèle personnalisé"
3. Parcourez vers ce dossier"""
            
            messagebox.showinfo("Succès", success_msg)
            
        except Exception as e:
            self.status_text.set("Erreur durant l'entraînement")
            self.progress.config(value=0)
            error_msg = f"Erreur durant l'entraînement :\n{str(e)}\n\nVérifiez que :\n- Le modèle de base est installé\n- Vous avez assez d'espace disque\n- Les fichiers d'entrée sont corrects"
            messagebox.showerror("Erreur", error_msg)
        
        finally:
            # Réactivation des boutons
            self.generate_button.config(state='normal')
            self.train_button.config(state='normal')

def main():
    """Test de l'interface de fine-tuning"""
    root = tk.Tk()
    root.withdraw()  # Cache la fenêtre principale
    app = FineTuningGUI()
    app.window.mainloop()

if __name__ == "__main__":
    main()