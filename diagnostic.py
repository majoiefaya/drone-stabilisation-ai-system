#!/usr/bin/env python3
"""
Script de diagnostic pour Streamlit Cloud
Vérifie que tous les composants sont prêts pour le déploiement
"""

import os
import sys
import subprocess
import importlib
import traceback

def check_file_exists(filepath, description):
    """Vérifie qu'un fichier existe"""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {filepath}")
    return exists

def check_import(module_name):
    """Vérifie qu'un module peut être importé"""
    try:
        importlib.import_module(module_name)
        print(f"✅ Module {module_name}: OK")
        return True
    except ImportError as e:
        print(f"❌ Module {module_name}: {e}")
        return False

def check_python_version():
    """Vérifie la version de Python"""
    version = sys.version_info
    if version >= (3, 8):
        print(f"✅ Python {version.major}.{version.minor}: OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}: Version trop ancienne (min 3.8)")
        return False

def main():
    print("🚁 DroneStab AI - Diagnostic Streamlit Cloud")
    print("=" * 50)
    
    errors = []
    
    # 1. Vérification Python
    print("\n📋 Vérification Python:")
    if not check_python_version():
        errors.append("Python version incompatible")
    
    # 2. Vérification des fichiers requis
    print("\n📁 Vérification des fichiers:")
    required_files = [
        ("main.py", "Application principale"),
        ("requirements.txt", "Dépendances Python"),
        (".streamlit/config.toml", "Configuration Streamlit"),
        ("packages.txt", "Dépendances système")
    ]
    
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            errors.append(f"Fichier manquant: {filepath}")
    
    # 3. Vérification des modules Python
    print("\n🐍 Vérification des modules:")
    required_modules = [
        "streamlit",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "plotly",
        "sklearn",
        "xgboost"
    ]
    
    for module in required_modules:
        if not check_import(module):
            errors.append(f"Module manquant: {module}")
    
    # 4. Vérification du contenu requirements.txt
    print("\n📦 Vérification requirements.txt:")
    try:
        with open("requirements.txt", "r") as f:
            content = f.read()
            if "streamlit" in content:
                print("✅ Streamlit trouvé dans requirements.txt")
            else:
                print("❌ Streamlit manquant dans requirements.txt")
                errors.append("Streamlit manquant dans requirements.txt")
    except Exception as e:
        print(f"❌ Erreur lecture requirements.txt: {e}")
        errors.append("Erreur requirements.txt")
    
    # 5. Test de syntaxe main.py
    print("\n🔍 Vérification syntaxe main.py:")
    try:
        with open("main.py", "r") as f:
            code = f.read()
            compile(code, "main.py", "exec")
        print("✅ Syntaxe main.py: OK")
    except SyntaxError as e:
        print(f"❌ Erreur syntaxe main.py: {e}")
        errors.append("Erreur syntaxe main.py")
    except Exception as e:
        print(f"❌ Erreur main.py: {e}")
        errors.append("Erreur main.py")
    
    # 6. Test Streamlit
    print("\n🚀 Test Streamlit:")
    try:
        result = subprocess.run([
            sys.executable, "-m", "streamlit", "hello"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Streamlit fonctionne")
        else:
            print(f"❌ Erreur Streamlit: {result.stderr}")
            errors.append("Streamlit ne fonctionne pas")
    except subprocess.TimeoutExpired:
        print("⚠️ Timeout test Streamlit (normal)")
    except Exception as e:
        print(f"❌ Erreur test Streamlit: {e}")
        errors.append("Erreur test Streamlit")
    
    # Résumé
    print("\n" + "=" * 50)
    if errors:
        print("❌ DIAGNOSTIC ÉCHOUÉ")
        print("\n🔧 Erreurs à corriger:")
        for error in errors:
            print(f"  - {error}")
        print("\n💡 Recommandations:")
        print("  1. Installer les modules manquants: pip install -r requirements.txt")
        print("  2. Vérifier la syntaxe des fichiers Python")
        print("  3. Consulter STREAMLIT_CLOUD_DEPLOY.md")
        return False
    else:
        print("✅ DIAGNOSTIC RÉUSSI")
        print("🚀 Prêt pour déploiement Streamlit Cloud!")
        return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"🚨 Erreur fatale: {e}")
        print(traceback.format_exc())
        sys.exit(1)
