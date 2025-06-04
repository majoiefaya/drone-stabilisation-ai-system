#!/bin/bash

echo "============================================"
echo "🚁 DroneStab AI - Test de Deployment Local"
echo "============================================"
echo

echo "🔍 Verification des fichiers requis..."

if [ ! -f "main.py" ]; then
    echo "❌ main.py manquant!"
    exit 1
fi

if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt manquant!"
    exit 1
fi

if [ ! -f ".streamlit/config.toml" ]; then
    echo "❌ .streamlit/config.toml manquant!"
    exit 1
fi

echo "✅ Tous les fichiers requis sont presents"
echo

echo "📦 Installation des dependances..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Erreur lors de l'installation des dependances!"
    exit 1
fi

echo "✅ Dependances installees avec succes"
echo

echo "🚀 Lancement de l'application Streamlit..."
echo "👉 L'application va s'ouvrir sur http://localhost:8501"
echo "🔄 Appuyez sur Ctrl+C pour arreter l'application"
echo

streamlit run main.py
