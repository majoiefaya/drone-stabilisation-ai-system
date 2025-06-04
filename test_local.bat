@echo off
echo ============================================
echo 🚁 DroneStab AI - Test de Deployment Local
echo ============================================
echo.

echo 🔍 Verification des fichiers requis...
if not exist "main.py" (
    echo ❌ main.py manquant!
    pause
    exit /b 1
)

if not exist "requirements.txt" (
    echo ❌ requirements.txt manquant!
    pause
    exit /b 1
)

if not exist ".streamlit\config.toml" (
    echo ❌ .streamlit\config.toml manquant!
    pause
    exit /b 1
)

echo ✅ Tous les fichiers requis sont presents
echo.

echo 📦 Installation des dependances...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo ❌ Erreur lors de l'installation des dependances!
    pause
    exit /b 1
)

echo ✅ Dependances installees avec succes
echo.

echo 🚀 Lancement de l'application Streamlit...
echo 👉 L'application va s'ouvrir sur http://localhost:8501
echo 🔄 Appuyez sur Ctrl+C pour arreter l'application
echo.

streamlit run main.py

pause
