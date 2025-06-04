@echo off
REM Script de démarrage pour l'application Streamlit DroneStab AI (Windows)
REM Ce script installe les dépendances et lance l'application

echo 🚁 Démarrage de DroneStab AI - Pipeline de Stabilisation Drone
echo ================================================================

REM Vérifier si Python est installé
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python n'est pas installé. Veuillez installer Python 3.8+
    pause
    exit /b 1
)

echo ✅ Python détecté
python --version

REM Installer les dépendances
echo 📦 Installation des dépendances...
pip install -r requirements.txt

if errorlevel 1 (
    echo ❌ Erreur lors de l'installation des dépendances
    pause
    exit /b 1
)

echo ✅ Dépendances installées avec succès

REM Lancer l'application Streamlit
echo 🚀 Lancement de l'application Streamlit...
echo 📱 L'application sera accessible sur: http://localhost:8501
echo 🔗 Pour l'arrêter, utilisez Ctrl+C
echo.

streamlit run app.py --server.port 8501 --server.address localhost

pause
