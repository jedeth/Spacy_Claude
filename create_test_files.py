#!/usr/bin/env python3
"""
Crée des fichiers de test pour le fine-tuning
"""

import os
from pathlib import Path

def create_test_files():
    """Crée des fichiers de test simples"""
    
    # Création du dossier de test
    test_dir = Path("test_finetuning")
    test_dir.mkdir(exist_ok=True)
    print(f"📁 Dossier créé : {test_dir}")
    
    # 1. Fichier annuaire simple
    annuaire_content = """Jean Dupont
Marie Martin
Pierre Durand
Sophie Leclerc
Antoine Bernard
Camille Rousseau
Lucas Moreau
Emma Fournier"""
    
    annuaire_file = test_dir / "annuaire_test.txt"
    annuaire_file.write_text(annuaire_content, encoding='utf-8')
    print(f"✅ Annuaire créé : {annuaire_file}")
    
    # 2. Fichier contextes simple
    contextes_content = """{PERSON} travaille depuis janvier.
{PERSON} est employé chez {ORG}.
L'entreprise {ORG} recrute activement.
{PERSON} habite à {LOC} depuis longtemps.
{PERSON} de {LOC} contacte {PERSON} aujourd'hui.
{ORG} est basée à {LOC} en France.
{PERSON} rejoint {ORG} cette année.
L'équipe de {ORG} grandit rapidement."""
    
    contextes_file = test_dir / "contextes_test.txt"
    contextes_file.write_text(contextes_content, encoding='utf-8')
    print(f"✅ Contextes créés : {contextes_file}")
    
    # 3. Fichier organisations
    orgs_content = """Microsoft France
Google Paris
Amazon Web Services
Apple Europe
Meta France
Airbus Group
Total Energies
BNP Paribas"""
    
    orgs_file = test_dir / "organisations_test.txt"
    orgs_file.write_text(orgs_content, encoding='utf-8')
    print(f"✅ Organisations créées : {orgs_file}")
    
    # 4. Fichier lieux
    lieux_content = """Paris
Lyon
Marseille
Toulouse
Nice
Nantes
Strasbourg
Bordeaux"""
    
    lieux_file = test_dir / "lieux_test.txt"
    lieux_file.write_text(lieux_content, encoding='utf-8')
    print(f"✅ Lieux créés : {lieux_file}")
    
    # 5. Fichier texte à tester
    texte_test = """Jean Dupont travaille chez Microsoft France depuis janvier 2020.
Il habite à Paris dans le 15ème arrondissement.
Son email professionnel est jean.dupont@microsoft.com.
Son téléphone est le 06.12.34.56.78.

Il collabore avec Marie Martin de Google Paris, basée à Lyon.
Marie peut être jointe au 04.78.90.12.34 ou par email.

Leur prochaine réunion est prévue le 25 décembre 2024 dans les bureaux d'Amazon Web Services à Toulouse.
Le contact local est Pierre Durand."""
    
    texte_file = test_dir / "texte_a_pseudonymiser.txt"
    texte_file.write_text(texte_test, encoding='utf-8')
    print(f"✅ Texte de test créé : {texte_file}")
    
    # 6. Instructions d'utilisation
    instructions = f"""📋 FICHIERS DE TEST CRÉÉS

Dossier : {test_dir.absolute()}

Fichiers disponibles :
• annuaire_test.txt        - Noms de personnes
• organisations_test.txt   - Noms d'entreprises  
• lieux_test.txt          - Noms de villes
• contextes_test.txt      - Phrases avec placeholders
• texte_a_pseudonymiser.txt - Texte d'exemple

🧪 COMMENT TESTER :

1. Lancez le debug :
   python debug_finetuning.py

2. Dans l'interface :
   - Cliquez "Test Import spaCy"
   - Cliquez "Test Variables tkinter"  
   - Sélectionnez annuaire_test.txt
   - Cliquez "Charger Annuaire"
   - Observez la console de debug

3. Si problème détecté :
   - Regardez les messages d'erreur dans la console
   - Vérifiez les traces d'erreur complètes

🔧 UTILISATION NORMALE :
1. Ouvrez l'application : python launch_stable.py
2. Onglet "Fine-tuning"
3. Chargez annuaire_test.txt (type: Personnes)
4. Chargez contextes_test.txt
5. Générez 20 exemples, 5 itérations (pour test rapide)
"""
    
    instructions_file = test_dir / "README.txt"
    instructions_file.write_text(instructions, encoding='utf-8')
    print(f"✅ Instructions créées : {instructions_file}")
    
    print(f"\n🎉 Tous les fichiers de test ont été créés dans {test_dir}/")
    print(f"\n💡 Prochaine étape : python debug_finetuning.py")
    
    return test_dir

if __name__ == "__main__":
    create_test_files()
    input("\nAppuyez sur Entrée pour fermer...")