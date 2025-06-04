#!/usr/bin/env python3
"""
Script de démarrage pour Streamlit Cloud
Gère le healthcheck et lance l'application
"""

import os
import sys
import time
import subprocess

def check_health():
    """Vérification basique de santé"""
    try:
        import streamlit
        import pandas
        import numpy
        import plotly
        import sklearn
        import xgboost
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False

def main():
    """Point d'entrée principal"""
    print("🚁 DroneStab AI - Démarrage Streamlit Cloud")
    
    # Attendre que les imports soient prêts
    max_retries = 5
    for attempt in range(max_retries):
        if check_health():
            print("✅ Healthcheck OK")
            break
        else:
            print(f"⚠️ Tentative {attempt + 1}/{max_retries}")
            time.sleep(2)
    else:
        print("❌ Healthcheck failed after max retries")
        sys.exit(1)
    
    # Importer et lancer l'application
    try:
        import main
        print("🚀 Application lancée")
    except Exception as e:
        print(f"❌ Erreur lors du lancement: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
