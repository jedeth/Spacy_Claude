#!/usr/bin/env python3
"""
Script de test pour vérifier le module de fine-tuning
Crée des fichiers de test et vérifie que tout fonctionne
"""

import tempfile
import os
from pathlib import Path
import sys

def create_test_files():
    """Crée des fichiers de test temporaires"""
    
    # Créer un dossier temporaire
    test_dir = Path(tempfile.mkdtemp(prefix="pseudonymizer_test_"))
    print(f"📁 Dossier de test créé : {test_dir}")
    
    # Fichier annuaire de test
    annuaire_content = """Jean Dupont
Marie Martin
Pierre Durand
Sophie Leclerc
Antoine Bernard"""
    
    annuaire_file = test_dir / "test_annuaire.txt"
    annuaire_file.write_text(annuaire_content, encoding='utf-8')
    
    # Fichier contextes de test
    contextes_content = """{PERSON} travaille chez {ORG} depuis janvier.
L'entreprise {ORG} est située à {LOC}.
{PERSON} de {LOC} contacte {PERSON} par email.
{PERSON} est responsable marketing chez {ORG}.
La société {ORG} basée à {LOC} recrute."""
    
    contextes_file = test_dir / "test_contextes.txt"
    contextes_file.write_text(contextes_content, encoding='utf-8')
    
    # Ajout de quelques organisations et lieux pour les tests
    orgs_content = """Microsoft France
Google Paris
Amazon Web Services
Apple Europe"""
    
    orgs_file = test_dir / "test_organisations.txt"
    orgs_file.write_text(orgs_content, encoding='utf-8')
    
    lieux_content = """Paris
Lyon
Marseille
Toulouse
Nice"""
    
    lieux_file = test_dir / "test_lieux.txt"
    lieux_file.write_text(lieux_content, encoding='utf-8')
    
    return test_dir, annuaire_file, contextes_file, orgs_file, lieux_file

def test_data_generation():
    """Test de génération de données"""
    print("\n🧪 Test de génération de données...")
    
    try:
        from spacy_finetuning_module import DataGenerator
        
        # Création des fichiers de test
        test_dir, annuaire_file, contextes_file, orgs_file, lieux_file = create_test_files()
        
        # Test du générateur
        generator = DataGenerator()
        
        # Chargement des fichiers
        print("📁 Chargement des annuaires...")
        count_persons = generator.load_directory_file(str(annuaire_file), "PERSON")
        count_orgs = generator.load_directory_file(str(orgs_file), "ORG")
        count_lieux = generator.load_directory_file(str(lieux_file), "LOC")
        
        print(f"✅ Personnes chargées : {count_persons}")
        print(f"✅ Organisations chargées : {count_orgs}")
        print(f"✅ Lieux chargés : {count_lieux}")
        
        # Chargement des contextes
        count_contexts = generator.load_context_templates(str(contextes_file))
        print(f"✅ Contextes chargés : {count_contexts}")
        
        # Génération des données
        print("🔄 Génération de 50 exemples d'entraînement...")
        training_data = generator.generate_training_data(50)
        
        print(f"✅ {len(training_data)} exemples générés")
        
        # Affichage de quelques exemples
        print("\n📋 Exemples générés :")
        for i, (text, annotation) in enumerate(training_data[:3]):
            print(f"\nExemple {i+1}:")
            print(f"Texte: {text}")
            print(f"Entités: {annotation['entities']}")
        
        # Sauvegarde des données de test
        output_file = test_dir / "training_data_test.json"
        generator.save_training_data(str(output_file))
        print(f"✅ Données sauvegardées dans : {output_file}")
        
        return True, test_dir
        
    except Exception as e:
        print(f"❌ Erreur lors du test de génération : {e}")
        return False, None

def test_spacy_model():
    """Test du modèle spaCy"""
    print("\n🧪 Test du modèle spaCy...")
    
    try:
        from spacy_finetuning_module import SpacyFineTuner
        
        # Test de chargement du modèle
        fine_tuner = SpacyFineTuner()
        print("📦 Tentative de chargement du modèle de base...")
        
        success = fine_tuner.load_base_model()
        if success:
            print("✅ Modèle de base chargé avec succès")
            return True
        else:
            print("❌ Échec du chargement du modèle")
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors du test du modèle : {e}")
        print("💡 Assurez-vous que spaCy et le modèle français sont installés :")
        print("   pip install spacy")
        print("   python -m spacy download fr_core_news_md")
        return False

def test_gui_import():
    """Test d'import de l'interface graphique"""
    print("\n🧪 Test d'import de l'interface...")
    
    try:
        from spacy_finetuning_module import FineTuningGUI
        print("✅ Interface graphique importée avec succès")
        return True
    except Exception as e:
        print(f"❌ Erreur lors de l'import de l'interface : {e}")
        return False

def cleanup_test_files(test_dir):
    """Nettoie les fichiers de test"""
    if test_dir and test_dir.exists():
        import shutil
        shutil.rmtree(test_dir)
        print(f"🧹 Dossier de test nettoyé : {test_dir}")

def main():
    """Fonction principale de test"""
    print("=" * 60)
    print("🧪 TESTS DU MODULE DE FINE-TUNING")
    print("=" * 60)
    
    test_results = []
    test_dir = None
    
    # Test 1: Import de l'interface
    result1 = test_gui_import()
    test_results.append(("Import interface", result1))
    
    # Test 2: Modèle spaCy
    result2 = test_spacy_model()
    test_results.append(("Modèle spaCy", result2))
    
    # Test 3: Génération de données
    if result2:  # Seulement si spaCy fonctionne
        result3, test_dir = test_data_generation()
        test_results.append(("Génération données", result3))
    else:
        test_results.append(("Génération données", False))
    
    # Résultats
    print("\n" + "=" * 60)
    print("📊 RÉSULTATS DES TESTS")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} : {status}")
        if result:
            passed += 1
    
    print(f"\nRésultat global : {passed}/{total} tests réussis")
    
    if passed == total:
        print("🎉 Tous les tests sont passés ! Le module est prêt à l'utilisation.")
    else:
        print("⚠️  Certains tests ont échoué. Vérifiez les dépendances.")
        print("\n💡 Solutions possibles :")
        print("1. Installez spaCy : pip install spacy")
        print("2. Téléchargez le modèle : python -m spacy download fr_core_news_md")
        print("3. Vérifiez votre environnement Python")
    
    # Nettoyage
    if test_dir:
        cleanup_test_files(test_dir)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Tests interrompus par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur critique lors des tests : {e}")
    finally:
        input("\nAppuyez sur Entrée pour fermer...")