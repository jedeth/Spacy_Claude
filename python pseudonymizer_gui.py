#!/usr/bin/env python3
"""
Script d'installation automatique pour le Pseudonymiseur de Textes
Installe toutes les dépendances nécessaires
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Exécute une commande et affiche le résultat"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Succès")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Erreur: {e}")
        print(f"Sortie d'erreur: {e.stderr}")
        return False

def check_python_version():
    """Vérifie la version de Python"""
    version = sys.version_info
    print(f"🐍 Version Python détectée: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ requis. Veuillez mettre à jour Python.")
        return False
    
    print("✅ Version Python compatible")
    return True

def install_dependencies():
    """Installe les dépendances Python"""
    dependencies = [
        "spacy>=3.4.0",
        "spacy[lookups]"
    ]
    
    for dep in dependencies:
        if not run_command(f"{sys.executable} -m pip install {dep}", f"Installation de {dep}"):
            return False
    
    return True

def download_spacy_models():
    """Télécharge les modèles spaCy français"""
    models = [
        ("fr_core_news_sm", "Modèle français petit (rapide)"),
        ("fr_core_news_md", "Modèle français moyen (recommandé)"),
        ("fr_core_news_lg", "Modèle français large (plus précis)")
    ]
    
    success_count = 0
    
    for model, description in models:
        print(f"📥 Téléchargement {description}...")
        if run_command(f"{sys.executable} -m spacy download {model}", f"Téléchargement {model}"):
            success_count += 1
        else:
            print(f"⚠️  Échec du téléchargement de {model} (peut être ignoré)")
    
    if success_count > 0:
        print(f"✅ {success_count} modèle(s) installé(s) avec succès")
        return True
    else:
        print("❌ Aucun modèle installé")
        return False

def test_installation():
    """Teste l'installation"""
    print("🧪 Test de l'installation...")
    
    try:
        import spacy
        print("✅ spaCy importé avec succès")
        
        # Test des modèles
        models_to_test = ["fr_core_news_sm", "fr_core_news_md", "fr_core_news_lg"]
        working_models = []
        
        for model in models_to_test:
            try:
                nlp = spacy.load(model)
                working_models.append(model)
                print(f"✅ Modèle {model} fonctionnel")
            except OSError:
                print(f"⚠️  Modèle {model} non disponible")
        
        if working_models:
            print(f"✅ {len(working_models)} modèle(s) prêt(s) à l'utilisation")
            
            # Test rapide de pseudonymisation
            best_model = working_models[-1]  # Prendre le meilleur disponible
            nlp = spacy.load(best_model)
            doc = nlp("Jean Dupont travaille chez Microsoft à Paris.")
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            if entities:
                print(f"✅ Test de reconnaissance d'entités réussi: {entities}")
            else:
                print("⚠️  Test de reconnaissance d'entités: aucune entité détectée")
            
            return True
        else:
            print("❌ Aucun modèle fonctionnel")
            return False
            
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        return False

def create_example_files():
    """Crée des fichiers d'exemple pour le fine-tuning"""
    print("📄 Création des fichiers d'exemple...")
    
    # Exemple d'annuaire
    annuaire_example = """Jean Dupont
Marie Martin
Pierre Durand
Sophie Leclerc
Antoine Bernard
Camille Rousseau
Lucas Moreau
Emma Fournier
Hugo Girard
Léa Mercier"""
    
    # Exemple de contextes
    contextes_example = """{PERSON} travaille chez {ORG} depuis janvier 2020.
L'entreprise {ORG} est située à {LOC}.
{PERSON} de {LOC} contacte {PERSON} par email.
{PERSON} est responsable marketing chez {ORG}.
La société {ORG} basée à {LOC} recrute.
{PERSON} habite à {LOC} et travaille pour {ORG}.
{PERSON} a rejoint {ORG} en tant que consultant.
L'équipe de {ORG} à {LOC} se développe.
{PERSON} gère le bureau de {ORG} à {LOC}.
{PERSON} collabore avec {PERSON} sur le projet {ORG}."""
    
    # Exemple de texte à traiter
    texte_example = """Jean Dupont travaille chez Microsoft France depuis le 15 janvier 2020. 
Il habite à Paris dans le 15ème arrondissement et son email professionnel est jean.dupont@microsoft.com. 
Son téléphone portable est le 06.12.34.56.78.

Il collabore régulièrement avec Marie Martin de Google France, basée à Lyon. 
Marie peut être jointe au 04.78.90.12.34 ou par email à marie.martin@google.fr.

Leur prochaine réunion est prévue le 25 décembre 2024 dans les bureaux d'Amazon à Toulouse.
Le contact local est Pierre Durand (pierre.durand@amazon.fr)."""
    
    try:
        # Création du dossier exemples
        examples_dir = Path("exemples_finetuning")
        examples_dir.mkdir(exist_ok=True)
        
        # Sauvegarde des fichiers
        (examples_dir / "annuaire_personnes.txt").write_text(annuaire_example, encoding='utf-8')
        (examples_dir / "contextes_phrases.txt").write_text(contextes_example, encoding='utf-8')
        (examples_dir / "texte_test.txt").write_text(texte_example, encoding='utf-8')
        
        print(f"✅ Fichiers d'exemple créés dans le dossier '{examples_dir}'")
        print("   - annuaire_personnes.txt : Exemple d'annuaire")
        print("   - contextes_phrases.txt : Phrases avec placeholders")
        print("   - texte_test.txt : Texte d'exemple à pseudonymiser")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Erreur lors de la création des exemples: {e}")
        return False

def main():
    """Fonction principale d'installation"""
    print("=" * 60)
    print("🚀 INSTALLATION DU PSEUDONYMISEUR DE TEXTES")
    print("=" * 60)
    
    # Vérifications préliminaires
    if not check_python_version():
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("📦 INSTALLATION DES DÉPENDANCES")
    print("=" * 60)
    
    # Installation des dépendances
    if not install_dependencies():
        print("❌ Échec de l'installation des dépendances")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("📥 TÉLÉCHARGEMENT DES MODÈLES")
    print("=" * 60)
    
    # Téléchargement des modèles
    if not download_spacy_models():
        print("❌ Échec du téléchargement des modèles")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🧪 TESTS DE VALIDATION")
    print("=" * 60)
    
    # Tests
    if not test_installation():
        print("❌ Les tests ont échoué")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("📄 CRÉATION DES EXEMPLES")
    print("=" * 60)
    
    # Création des exemples
    create_example_files()
    
    print("\n" + "=" * 60)
    print("🎉 INSTALLATION TERMINÉE AVEC SUCCÈS!")
    print("=" * 60)
    
    print("\n📋 PROCHAINES ÉTAPES:")
    print("1. Lancez l'application: python pseudonymizer_gui.py")
    print("2. Pour le fine-tuning, utilisez les fichiers dans 'exemples_finetuning/'")
    print("3. Consultez la documentation pour plus d'informations")
    
    print("\n💡 CONSEILS:")
    print("- Utilisez fr_core_news_md pour un bon équilibre vitesse/précision")
    print("- Le fine-tuning améliore significativement les résultats")
    print("- Testez d'abord avec le fichier texte_test.txt")

if __name__ == "__main__":
    main()