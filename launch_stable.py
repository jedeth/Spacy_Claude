#!/usr/bin/env python3
"""
Lanceur pour la version stable du Pseudonymiseur
Évite tous les problèmes de threading
"""

import sys
import os
import subprocess
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

def check_spacy():
    """Vérifie spaCy rapidement"""
    try:
        import spacy
        
        # Test des modèles
        models = ["fr_core_news_md", "fr_core_news_sm"]
        for model in models:
            try:
                nlp = spacy.load(model)
                print(f"✅ Modèle {model} disponible")
                return True
            except OSError:
                continue
        
        print("⚠️  Aucun modèle français trouvé")
        return False
        
    except ImportError:
        print("❌ spaCy non installé")
        return False

def install_spacy_simple():
    """Installation simple de spaCy"""
    try:
        print("📦 Installation de spaCy...")
        subprocess.run([sys.executable, "-m", "pip", "install", "spacy"], check=True, capture_output=True)
        
        print("📥 Téléchargement du modèle...")
        subprocess.run([sys.executable, "-m", "spacy", "download", "fr_core_news_md"], check=True, capture_output=True)
        
        return True
    except subprocess.CalledProcessError:
        try:
            # Essai avec le modèle plus petit
            subprocess.run([sys.executable, "-m", "spacy", "download", "fr_core_news_sm"], check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError:
            return False

def launch_app():
    """Lance l'application stable"""
    try:
        print("🚀 Lancement de la version stable...")
        
        # Vérification des fichiers
        required_files = ["pseudonymizer_simple.py"]
        missing = [f for f in required_files if not Path(f).exists()]
        
        if missing:
            print(f"❌ Fichiers manquants : {missing}")
            return False
        
        # Lancement direct
        import pseudonymizer_simple
        pseudonymizer_simple.main()
        return True
        
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return False

def main():
    """Fonction principale"""
    print("=" * 50)
    print("🔒 PSEUDONYMISEUR - VERSION STABLE")
    print("=" * 50)
    
    # Vérification spaCy
    if not check_spacy():
        print("\n💡 Installation de spaCy requise...")
        
        if install_spacy_simple():
            print("✅ Installation réussie")
        else:
            print("❌ Installation échouée")
            print("\nInstallez manuellement :")
            print("pip install spacy")
            print("python -m spacy download fr_core_news_md")
            input("Appuyez sur Entrée...")
            return
    
    print("✅ spaCy configuré")
    
    # Lancement
    if launch_app():
        print("✅ Application fermée normalement")
    else:
        print("❌ Erreur de lancement")
        input("Appuyez sur Entrée...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Arrêt par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur critique : {e}")
        input("Appuyez sur Entrée...")