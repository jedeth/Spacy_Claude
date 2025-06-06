#!/usr/bin/env python3
"""
Version debug du fine-tuning pour diagnostiquer les problèmes
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import traceback
import sys
from pathlib import Path

# Redirection des prints vers la console
class DebugConsole:
    def __init__(self):
        self.console = None
    
    def create_console(self, parent):
        """Crée une console de debug"""
        console_frame = ttk.LabelFrame(parent, text="Console de Debug", padding=5)
        console_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Zone de texte avec scrollbar
        text_frame = ttk.Frame(console_frame)
        text_frame.pack(fill='both', expand=True)
        
        self.console = tk.Text(text_frame, height=10, width=80, wrap='word')
        scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=self.console.yview)
        self.console.configure(yscrollcommand=scrollbar.set)
        
        self.console.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Bouton pour vider la console
        ttk.Button(console_frame, text="Vider Console", 
                  command=self.clear).pack(pady=5)
    
    def log(self, message):
        """Ajoute un message à la console"""
        if self.console:
            self.console.insert('end', f"{message}\n")
            self.console.see('end')
        print(message)  # Aussi dans la console système
    
    def clear(self):
        """Vide la console"""
        if self.console:
            self.console.delete('1.0', 'end')

# Console globale
debug_console = DebugConsole()

def safe_execute(func, func_name, *args, **kwargs):
    """Exécute une fonction avec gestion d'erreur complète"""
    try:
        debug_console.log(f"🔄 Début: {func_name}")
        result = func(*args, **kwargs)
        debug_console.log(f"✅ Succès: {func_name}")
        return result
    except Exception as e:
        error_msg = f"❌ Erreur dans {func_name}: {str(e)}"
        debug_console.log(error_msg)
        debug_console.log(f"📋 Trace: {traceback.format_exc()}")
        messagebox.showerror("Erreur Debug", f"{func_name} a échoué:\n{str(e)}")
        return None

class DebugFineTuningGUI:
    """Version debug du fine-tuning"""
    
    def __init__(self, parent_window=None):
        debug_console.log("🚀 Initialisation de l'interface de debug")
        
        try:
            self.window = tk.Toplevel(parent_window) if parent_window else tk.Tk()
            self.window.title("Fine-tuning DEBUG")
            self.window.geometry("900x700")
            
            # Gestion de la fermeture
            self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            # Variables
            self.directory_file = tk.StringVar()
            self.context_file = tk.StringVar()
            self.status_text = tk.StringVar(value="Interface debug initialisée")
            
            debug_console.log("✅ Variables initialisées")
            
            # Création de l'interface
            self.create_debug_widgets()
            
            debug_console.log("✅ Interface créée avec succès")
            
        except Exception as e:
            debug_console.log(f"❌ Erreur lors de l'initialisation: {e}")
            debug_console.log(f"📋 Trace: {traceback.format_exc()}")
            raise
    
    def on_closing(self):
        """Gestion de la fermeture de fenêtre"""
        debug_console.log("👋 Fermeture de l'interface de debug")
        try:
            self.window.destroy()
        except:
            pass
    
    def create_debug_widgets(self):
        """Crée l'interface de debug"""
        debug_console.log("🔨 Création des widgets...")
        
        # Titre
        ttk.Label(self.window, text="Fine-tuning - Mode Debug", 
                 font=('Arial', 14, 'bold')).pack(pady=10)
        
        # Frame principal avec deux colonnes
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Colonne gauche : Contrôles
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side='left', fill='y', padx=(0,5))
        
        # Section test de base
        test_frame = ttk.LabelFrame(left_frame, text="Tests de Base", padding=10)
        test_frame.pack(fill='x', pady=5)
        
        ttk.Button(test_frame, text="Test Import spaCy", 
                  command=self.test_spacy_import).pack(fill='x', pady=2)
        ttk.Button(test_frame, text="Test Variables tkinter", 
                  command=self.test_tkinter_vars).pack(fill='x', pady=2)
        ttk.Button(test_frame, text="Test Dialog", 
                  command=self.test_file_dialog).pack(fill='x', pady=2)
        
        # Section sélection fichiers
        file_frame = ttk.LabelFrame(left_frame, text="Sélection Fichiers", padding=10)
        file_frame.pack(fill='x', pady=5)
        
        # Annuaire
        ttk.Label(file_frame, text="Annuaire:").pack(anchor='w')
        annuaire_frame = ttk.Frame(file_frame)
        annuaire_frame.pack(fill='x', pady=2)
        
        self.annuaire_entry = ttk.Entry(annuaire_frame, textvariable=self.directory_file, 
                                       state='readonly', width=30)
        self.annuaire_entry.pack(side='left', fill='x', expand=True)
        ttk.Button(annuaire_frame, text="...", width=3,
                  command=self.select_directory_debug).pack(side='right')
        
        ttk.Button(file_frame, text="Charger Annuaire", 
                  command=self.load_directory_debug).pack(fill='x', pady=5)
        
        # Contextes
        ttk.Label(file_frame, text="Contextes:").pack(anchor='w', pady=(10,0))
        context_frame = ttk.Frame(file_frame)
        context_frame.pack(fill='x', pady=2)
        
        self.context_entry = ttk.Entry(context_frame, textvariable=self.context_file, 
                                      state='readonly', width=30)
        self.context_entry.pack(side='left', fill='x', expand=True)
        ttk.Button(context_frame, text="...", width=3,
                  command=self.select_context_debug).pack(side='right')
        
        ttk.Button(file_frame, text="Charger Contextes", 
                  command=self.load_contexts_debug).pack(fill='x', pady=5)
        
        # Statut
        status_frame = ttk.LabelFrame(left_frame, text="Statut", padding=10)
        status_frame.pack(fill='x', pady=5)
        
        ttk.Label(status_frame, textvariable=self.status_text, 
                 wraplength=200).pack()
        
        # Colonne droite : Console
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side='right', fill='both', expand=True, padx=(5,0))
        
        debug_console.create_console(right_frame)
        
        debug_console.log("✅ Widgets créés avec succès")
    
    def test_spacy_import(self):
        """Test d'import spaCy"""
        def test():
            import spacy
            debug_console.log(f"spaCy version: {spacy.__version__}")
            
            # Test de chargement de modèle
            models = ["fr_core_news_md", "fr_core_news_sm"]
            for model in models:
                try:
                    nlp = spacy.load(model)
                    debug_console.log(f"✅ Modèle {model}: OK")
                    return True
                except OSError:
                    debug_console.log(f"❌ Modèle {model}: Non trouvé")
            
            return False
        
        safe_execute(test, "test_spacy_import")
    
    def test_tkinter_vars(self):
        """Test des variables tkinter"""
        def test():
            debug_console.log(f"directory_file: '{self.directory_file.get()}'")
            debug_console.log(f"context_file: '{self.context_file.get()}'")
            debug_console.log(f"status_text: '{self.status_text.get()}'")
            
            # Test de modification
            self.status_text.set("Test des variables réussi")
            debug_console.log("✅ Modification de status_text: OK")
        
        safe_execute(test, "test_tkinter_vars")
    
    def test_file_dialog(self):
        """Test des dialogs de fichiers"""
        def test():
            file_path = filedialog.askopenfilename(
                title="Test de sélection de fichier",
                filetypes=[("Fichiers texte", "*.txt"), ("Tous", "*.*")]
            )
            if file_path:
                debug_console.log(f"✅ Fichier sélectionné: {file_path}")
            else:
                debug_console.log("⚠️ Aucun fichier sélectionné")
        
        safe_execute(test, "test_file_dialog")
    
    def select_directory_debug(self):
        """Sélection du fichier annuaire avec debug"""
        def select():
            debug_console.log("📂 Ouverture dialog annuaire...")
            file_path = filedialog.askopenfilename(
                title="Fichier annuaire",
                filetypes=[("Fichiers texte", "*.txt"), ("Tous", "*.*")]
            )
            if file_path:
                self.directory_file.set(file_path)
                debug_console.log(f"✅ Annuaire sélectionné: {file_path}")
                self.status_text.set(f"Annuaire: {Path(file_path).name}")
            else:
                debug_console.log("⚠️ Sélection annuaire annulée")
        
        safe_execute(select, "select_directory_debug")
    
    def select_context_debug(self):
        """Sélection du fichier contextes avec debug"""
        def select():
            debug_console.log("📂 Ouverture dialog contextes...")
            file_path = filedialog.askopenfilename(
                title="Fichier contextes",
                filetypes=[("Fichiers texte", "*.txt"), ("Tous", "*.*")]
            )
            if file_path:
                self.context_file.set(file_path)
                debug_console.log(f"✅ Contextes sélectionnés: {file_path}")
                self.status_text.set(f"Contextes: {Path(file_path).name}")
            else:
                debug_console.log("⚠️ Sélection contextes annulée")
        
        safe_execute(select, "select_context_debug")
    
    def load_directory_debug(self):
        """Chargement de l'annuaire avec debug complet"""
        def load():
            if not self.directory_file.get():
                debug_console.log("❌ Aucun fichier annuaire sélectionné")
                messagebox.showwarning("Attention", "Sélectionnez d'abord un fichier annuaire")
                return
            
            file_path = self.directory_file.get()
            debug_console.log(f"📖 Lecture du fichier: {file_path}")
            
            # Vérification existence
            if not Path(file_path).exists():
                debug_console.log(f"❌ Fichier inexistant: {file_path}")
                messagebox.showerror("Erreur", f"Fichier introuvable: {file_path}")
                return
            
            # Lecture du contenu
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
                
                debug_console.log(f"✅ Fichier lu: {len(lines)} lignes")
                
                # Affichage des premières lignes
                preview = lines[:5]
                debug_console.log(f"📋 Aperçu: {preview}")
                
                self.status_text.set(f"Annuaire chargé: {len(lines)} entrées")
                messagebox.showinfo("Succès", f"Annuaire chargé avec succès!\n{len(lines)} entrées trouvées")
                
            except Exception as e:
                debug_console.log(f"❌ Erreur lecture fichier: {e}")
                messagebox.showerror("Erreur", f"Impossible de lire le fichier:\n{str(e)}")
        
        safe_execute(load, "load_directory_debug")
    
    def load_contexts_debug(self):
        """Chargement des contextes avec debug"""
        def load():
            if not self.context_file.get():
                debug_console.log("❌ Aucun fichier contextes sélectionné")
                messagebox.showwarning("Attention", "Sélectionnez d'abord un fichier de contextes")
                return
            
            file_path = self.context_file.get()
            debug_console.log(f"📖 Lecture des contextes: {file_path}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
                
                debug_console.log(f"✅ Contextes lus: {len(lines)} lignes")
                
                # Vérification des placeholders
                placeholders_found = set()
                for line in lines:
                    if "{PERSON}" in line:
                        placeholders_found.add("PERSON")
                    if "{ORG}" in line:
                        placeholders_found.add("ORG")
                    if "{LOC}" in line:
                        placeholders_found.add("LOC")
                
                debug_console.log(f"📋 Placeholders trouvés: {placeholders_found}")
                
                self.status_text.set(f"Contextes chargés: {len(lines)} templates")
                messagebox.showinfo("Succès", f"Contextes chargés!\n{len(lines)} templates\nPlaceholders: {placeholders_found}")
                
            except Exception as e:
                debug_console.log(f"❌ Erreur lecture contextes: {e}")
                messagebox.showerror("Erreur", f"Impossible de lire les contextes:\n{str(e)}")
        
        safe_execute(load, "load_contexts_debug")

def main():
    """Fonction principale de debug"""
    print("🐛 Lancement du mode debug fine-tuning")
    
    try:
        root = tk.Tk()
        root.withdraw()  # Cache la fenêtre principale
        
        app = DebugFineTuningGUI()
        debug_console.log("🎉 Interface de debug prête!")
        
        app.window.mainloop()
        
    except Exception as e:
        print(f"❌ Erreur critique: {e}")
        print(f"📋 Trace: {traceback.format_exc()}")
        input("Appuyez sur Entrée...")

if __name__ == "__main__":
    main()