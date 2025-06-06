import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import spacy
import re
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

# Import du module de fine-tuning
try:
    from spacy_finetuning_fixed import FineTuningGUI
    FINETUNING_AVAILABLE = True
except ImportError:
    FINETUNING_AVAILABLE = False

@dataclass
class PseudonymizationConfig:
    """Configuration pour la pseudonymisation"""
    mask_persons: bool = True
    mask_orgs: bool = True
    mask_locations: bool = True
    mask_dates: bool = True
    mask_emails: bool = True
    mask_phones: bool = True
    replacement_char: str = "X"
    preserve_length: bool = True
    use_placeholders: bool = False

class TextPseudonymizer:
    """Classe principale pour la pseudonymisation de textes"""
    
    def __init__(self, model_name: str = "fr_core_news_md"):
        """
        Initialise le pseudonymiseur
        
        Args:
            model_name: Nom du modèle spaCy à utiliser
        """
        try:
            self.nlp = spacy.load(model_name)
            print(f"✅ Modèle {model_name} chargé avec succès")
        except OSError:
            try:
                # Fallback vers le modèle plus petit
                model_name = "fr_core_news_sm"
                self.nlp = spacy.load(model_name)
                print(f"✅ Modèle de fallback {model_name} chargé")
            except OSError:
                raise Exception(f"Aucun modèle français trouvé. Installez avec : python -m spacy download fr_core_news_md")
        
        # Patterns regex pour emails et téléphones
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(?:\+33|0)[1-9](?:[.\-\s]?\d{2}){4}')
        
        # Mapping des types d'entités spaCy vers nos catégories
        self.entity_mapping = {
            'PERSON': 'PERSONNE',
            'PER': 'PERSONNE',
            'ORG': 'ORGANISATION',
            'GPE': 'LIEU',
            'LOC': 'LIEU',
            'DATE': 'DATE',
            'TIME': 'DATE'
        }
        
        # Compteurs pour générer des pseudonymes cohérents
        self.person_counter = 1
        self.org_counter = 1
        self.location_counter = 1
        self.date_counter = 1
        self.email_counter = 1
        self.phone_counter = 1
        
        # Dictionnaire de correspondances pour cohérence
        self.correspondences = {}
    
    def _generate_consistent_replacement(self, original_text: str, entity_type: str, config: PseudonymizationConfig) -> str:
        """Génère un remplacement cohérent (même entité = même pseudo)"""
        
        # Clé unique pour cette entité
        key = f"{entity_type}_{original_text.lower()}"
        
        if key in self.correspondences:
            return self.correspondences[key]
        
        if config.use_placeholders:
            replacement = f"[{entity_type}]"
        else:
            # Génération de pseudonymes cohérents
            if entity_type == 'PERSONNE':
                replacement = f"PERSONNE_{self.person_counter}"
                self.person_counter += 1
            elif entity_type == 'ORGANISATION':
                replacement = f"ORGANISATION_{self.org_counter}"
                self.org_counter += 1
            elif entity_type == 'LIEU':
                replacement = f"LIEU_{self.location_counter}"
                self.location_counter += 1
            elif entity_type == 'DATE':
                replacement = f"DATE_{self.date_counter}"
                self.date_counter += 1
            elif entity_type == 'EMAIL':
                replacement = f"email{self.email_counter}@exemple.com"
                self.email_counter += 1
            elif entity_type == 'TELEPHONE':
                replacement = f"0{self.phone_counter}.XX.XX.XX.XX"
                self.phone_counter += 1
            else:
                if config.preserve_length:
                    replacement = config.replacement_char * len(original_text)
                else:
                    replacement = config.replacement_char * 3
        
        self.correspondences[key] = replacement
        return replacement
    
    def _pseudonymize_regex_patterns(self, text: str, config: PseudonymizationConfig) -> str:
        """Pseudonymise les emails et téléphones avec regex"""
        if config.mask_emails:
            def replace_email(match):
                return self._generate_consistent_replacement(match.group(), "EMAIL", config)
            text = self.email_pattern.sub(replace_email, text)
        
        if config.mask_phones:
            def replace_phone(match):
                return self._generate_consistent_replacement(match.group(), "TELEPHONE", config)
            text = self.phone_pattern.sub(replace_phone, text)
        
        return text
    
    def pseudonymize(self, text: str, config: Optional[PseudonymizationConfig] = None) -> Dict:
        """Pseudonymise un texte selon la configuration"""
        if config is None:
            config = PseudonymizationConfig()
        
        doc = self.nlp(text)
        entities_found = []
        pseudonymized_text = text
        offset = 0
        
        # Traitement des entités nommées
        for ent in doc.ents:
            entity_type = self.entity_mapping.get(ent.label_, ent.label_)
            
            should_mask = False
            if entity_type == 'PERSONNE' and config.mask_persons:
                should_mask = True
            elif entity_type == 'ORGANISATION' and config.mask_orgs:
                should_mask = True
            elif entity_type == 'LIEU' and config.mask_locations:
                should_mask = True
            elif entity_type == 'DATE' and config.mask_dates:
                should_mask = True
            
            if should_mask:
                start_pos = ent.start_char + offset
                end_pos = ent.end_char + offset
                
                replacement = self._generate_consistent_replacement(ent.text, entity_type, config)
                
                pseudonymized_text = (
                    pseudonymized_text[:start_pos] + 
                    replacement + 
                    pseudonymized_text[end_pos:]
                )
                
                offset += len(replacement) - len(ent.text)
                
                entities_found.append({
                    'original': ent.text,
                    'replacement': replacement,
                    'type': entity_type,
                    'position': f"{ent.start_char}-{ent.end_char}"
                })
        
        # Traitement des patterns regex
        pseudonymized_text = self._pseudonymize_regex_patterns(pseudonymized_text, config)
        
        return {
            'original_text': text,
            'pseudonymized_text': pseudonymized_text,
            'entities': entities_found,
            'correspondences': dict(self.correspondences)
        }

class PseudonymizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Pseudonymiseur de Textes - Version Stable")
        self.root.geometry("650x700")
        self.root.resizable(True, True)
        
        # Variables
        self.file_path = tk.StringVar()
        self.status_text = tk.StringVar(value="Initialisation...")
        self.custom_model_path = tk.StringVar()
        self.use_custom_model = tk.BooleanVar(value=False)
        
        # Variables pour les options
        self.mask_persons = tk.BooleanVar(value=True)
        self.mask_orgs = tk.BooleanVar(value=True)
        self.mask_locations = tk.BooleanVar(value=True)
        self.mask_dates = tk.BooleanVar(value=True)
        self.mask_emails = tk.BooleanVar(value=True)
        self.mask_phones = tk.BooleanVar(value=True)
        self.replacement_mode = tk.StringVar(value="consistent")
        
        self.pseudonymizer = None
        
        # Création de l'interface
        self.create_widgets()
        
        # Initialisation du modèle après création de l'interface
        self.root.after(500, self.init_model)
    
    def init_model(self):
        """Initialise le modèle de manière synchrone"""
        try:
            self.status_text.set("Chargement du modèle spaCy...")
            self.root.update()
            
            if self.use_custom_model.get() and self.custom_model_path.get():
                if os.path.exists(self.custom_model_path.get()):
                    self.pseudonymizer = TextPseudonymizer(self.custom_model_path.get())
                    self.status_text.set("Modèle personnalisé chargé - Prêt")
                else:
                    raise Exception("Chemin du modèle personnalisé introuvable")
            else:
                self.pseudonymizer = TextPseudonymizer()
                self.status_text.set("Modèle chargé - Prêt à traiter un fichier")
            
            self.process_button.config(state='normal')
            
        except Exception as e:
            self.status_text.set(f"Erreur : {str(e)}")
            messagebox.showerror("Erreur", 
                f"Impossible de charger le modèle :\n{str(e)}\n\n"
                "Vérifiez que spaCy et le modèle français sont installés.")
    
    def create_widgets(self):
        """Crée l'interface utilisateur"""
        
        # Titre principal
        title_label = ttk.Label(self.root, text="Pseudonymiseur de Textes", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=15)
        
        # Notebook pour les onglets
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Onglet 1: Pseudonymisation
        pseudo_frame = ttk.Frame(notebook)
        notebook.add(pseudo_frame, text="Pseudonymisation")
        
        # Section modèle
        model_frame = ttk.LabelFrame(pseudo_frame, text="Configuration du modèle", padding=10)
        model_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Radiobutton(model_frame, text="Modèle standard (fr_core_news_md)", 
                       variable=self.use_custom_model, value=False,
                       command=self.on_model_change).pack(anchor='w')
        
        custom_frame = ttk.Frame(model_frame)
        custom_frame.pack(fill='x', pady=5)
        
        ttk.Radiobutton(custom_frame, text="Modèle personnalisé :", 
                       variable=self.use_custom_model, value=True,
                       command=self.on_model_change).pack(anchor='w')
        
        custom_path_frame = ttk.Frame(model_frame)
        custom_path_frame.pack(fill='x', padx=20)
        
        self.custom_model_entry = ttk.Entry(custom_path_frame, textvariable=self.custom_model_path, 
                                           state='disabled', width=40)
        self.custom_model_entry.pack(side='left', fill='x', expand=True)
        
        self.browse_model_button = ttk.Button(custom_path_frame, text="Parcourir...", 
                                             command=self.select_custom_model, state='disabled')
        self.browse_model_button.pack(side='right', padx=(5,0))
        
        # Section fichier
        file_frame = ttk.LabelFrame(pseudo_frame, text="Fichier à traiter", padding=10)
        file_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(file_frame, text="Sélectionnez un fichier .txt :").pack(anchor='w')
        
        file_select_frame = ttk.Frame(file_frame)
        file_select_frame.pack(fill='x', pady=5)
        
        self.file_entry = ttk.Entry(file_select_frame, textvariable=self.file_path, 
                                   state='readonly', width=50)
        self.file_entry.pack(side='left', fill='x', expand=True)
        
        ttk.Button(file_select_frame, text="Parcourir...", 
                  command=self.select_file).pack(side='right', padx=(5,0))
        
        # Section options
        options_frame = ttk.LabelFrame(pseudo_frame, text="Options", padding=10)
        options_frame.pack(fill='x', padx=10, pady=5)
        
        # Checkboxes
        options_grid = ttk.Frame(options_frame)
        options_grid.pack(fill='x')
        
        ttk.Checkbutton(options_grid, text="Noms de personnes", 
                       variable=self.mask_persons).grid(row=0, column=0, sticky='w', padx=(0,20))
        ttk.Checkbutton(options_grid, text="Organisations", 
                       variable=self.mask_orgs).grid(row=0, column=1, sticky='w', padx=(0,20))
        ttk.Checkbutton(options_grid, text="Lieux", 
                       variable=self.mask_locations).grid(row=1, column=0, sticky='w', padx=(0,20))
        ttk.Checkbutton(options_grid, text="Dates", 
                       variable=self.mask_dates).grid(row=1, column=1, sticky='w', padx=(0,20))
        ttk.Checkbutton(options_grid, text="Emails", 
                       variable=self.mask_emails).grid(row=2, column=0, sticky='w', padx=(0,20))
        ttk.Checkbutton(options_grid, text="Téléphones", 
                       variable=self.mask_phones).grid(row=2, column=1, sticky='w', padx=(0,20))
        
        # Mode de remplacement
        ttk.Label(options_frame, text="Mode de remplacement :").pack(anchor='w', pady=(10,5))
        ttk.Radiobutton(options_frame, text="Pseudonymes cohérents (PERSONNE_1, LIEU_1, etc.)", 
                       variable=self.replacement_mode, value="consistent").pack(anchor='w')
        ttk.Radiobutton(options_frame, text="Placeholders génériques ([PERSONNE], [LIEU], etc.)", 
                       variable=self.replacement_mode, value="placeholder").pack(anchor='w')
        
        # Bouton de traitement
        self.process_button = ttk.Button(pseudo_frame, text="Traiter le fichier", 
                                        command=self.process_file, state='disabled')
        self.process_button.pack(pady=15)
        
        # Onglet 2: Fine-tuning
        if FINETUNING_AVAILABLE:
            finetune_frame = ttk.Frame(notebook)
            notebook.add(finetune_frame, text="Fine-tuning")
            
            ttk.Label(finetune_frame, text="Amélioration du modèle", 
                     font=('Arial', 14, 'bold')).pack(pady=20)
            
            info_text = """Le fine-tuning permet d'améliorer la précision du modèle 
en l'entraînant sur vos données spécifiques.

Vous aurez besoin de :
• Un fichier annuaire (noms, organisations, lieux)
• Des phrases contextuelles avec placeholders

Le processus crée un modèle personnalisé optimisé 
pour vos types de documents."""
            
            ttk.Label(finetune_frame, text=info_text, justify='center', 
                     wraplength=500).pack(pady=20)
            
            ttk.Button(finetune_frame, text="Lancer le Fine-tuning", 
                      command=self.open_finetuning_window).pack(pady=20)
        
        # Barre de statut
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(status_frame, text="Statut :").pack(side='left')
        status_label = ttk.Label(status_frame, textvariable=self.status_text, 
                                foreground='blue')
        status_label.pack(side='left', padx=(5,0))
    
    def on_model_change(self):
        """Gère le changement de modèle"""
        if self.use_custom_model.get():
            self.custom_model_entry.config(state='normal')
            self.browse_model_button.config(state='normal')
        else:
            self.custom_model_entry.config(state='disabled')
            self.browse_model_button.config(state='disabled')
        
        # Rechargement du modèle
        self.process_button.config(state='disabled')
        self.root.after(100, self.init_model)
    
    def select_custom_model(self):
        """Sélection du modèle personnalisé"""
        folder_path = filedialog.askdirectory(
            title="Sélectionner le dossier du modèle fine-tuné"
        )
        if folder_path:
            self.custom_model_path.set(folder_path)
            self.on_model_change()
    
    def select_file(self):
        """Ouvre la boîte de dialogue de sélection de fichier"""
        file_path = filedialog.askopenfilename(
            title="Sélectionner un fichier texte",
            filetypes=[("Fichiers texte", "*.txt"), ("Tous les fichiers", "*.*")]
        )
        if file_path:
            self.file_path.set(file_path)
    
    def open_finetuning_window(self):
        """Ouvre la fenêtre de fine-tuning"""
        if FINETUNING_AVAILABLE:
            try:
                FineTuningGUI(self.root)
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible d'ouvrir le fine-tuning : {str(e)}")
        else:
            messagebox.showerror("Erreur", "Module de fine-tuning non disponible")
    
    def process_file(self):
        """Traite le fichier sélectionné"""
        if not self.file_path.get():
            messagebox.showerror("Erreur", "Veuillez sélectionner un fichier.")
            return
        
        if not self.pseudonymizer:
            messagebox.showerror("Erreur", "Le modèle n'est pas chargé.")
            return
        
        try:
            # Lecture du fichier
            file_path = Path(self.file_path.get())
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Configuration
            config = PseudonymizationConfig(
                mask_persons=self.mask_persons.get(),
                mask_orgs=self.mask_orgs.get(),
                mask_locations=self.mask_locations.get(),
                mask_dates=self.mask_dates.get(),
                mask_emails=self.mask_emails.get(),
                mask_phones=self.mask_phones.get(),
                use_placeholders=(self.replacement_mode.get() == "placeholder")
            )
            
            # Traitement
            self.status_text.set("Traitement en cours...")
            self.root.update()
            
            result = self.pseudonymizer.pseudonymize(text, config)
            
            # Sauvegarde des fichiers
            base_name = file_path.stem
            output_dir = file_path.parent
            
            # Fichier pseudonymisé
            pseudo_file = output_dir / f"{base_name}_pseudo.txt"
            with open(pseudo_file, 'w', encoding='utf-8') as f:
                f.write(result['pseudonymized_text'])
            
            # Fichier de correspondances
            correspondances_file = output_dir / f"{base_name}_correspondances.json"
            correspondances_data = {
                'config_utilisee': config.__dict__,
                'statistiques': {
                    'entites_trouvees': len(result['entities']),
                    'correspondances_totales': len(result['correspondences'])
                },
                'entites_par_type': {},
                'correspondances': result['correspondences']
            }
            
            # Groupement par type
            for entity in result['entities']:
                entity_type = entity['type']
                if entity_type not in correspondances_data['entites_par_type']:
                    correspondances_data['entites_par_type'][entity_type] = []
                correspondances_data['entites_par_type'][entity_type].append({
                    'original': entity['original'],
                    'remplacement': entity['replacement'],
                    'position': entity['position']
                })
            
            with open(correspondances_file, 'w', encoding='utf-8') as f:
                json.dump(correspondances_data, f, ensure_ascii=False, indent=2)
            
            # Message de succès
            success_msg = f"""Traitement terminé avec succès !

Fichiers créés :
• {pseudo_file.name}
• {correspondances_file.name}

Entités pseudonymisées : {len(result['entities'])}"""
            
            self.status_text.set("Traitement terminé")
            messagebox.showinfo("Succès", success_msg)
            
        except Exception as e:
            error_msg = f"Erreur lors du traitement : {str(e)}"
            self.status_text.set("Erreur")
            messagebox.showerror("Erreur", error_msg)

def main():
    """Fonction principale"""
    root = tk.Tk()
    app = PseudonymizerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()