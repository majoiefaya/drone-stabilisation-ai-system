#!/bin/bash

# Script de démarrage pour l'application Streamlit DroneStab AI
# Ce script installe les dépendances et lance l'application

echo "🚁 Démarrage de DroneStab AI - Pipeline de Stabilisation Drone"
echo "================================================================"

# Vérifier si Python est installé
if ! command -v python &> /dev/null; then
    echo "❌ Python n'est pas installé. Veuillez installer Python 3.8+"
    exit 1
fi

echo "✅ Python détecté: $(python --version)"

# Installer les dépendances
echo "📦 Installation des dépendances..."
pip install -r requirements.txt

# Vérifier si l'installation s'est bien passée
if [ $? -eq 0 ]; then
    echo "✅ Dépendances installées avec succès"
else
    echo "❌ Erreur lors de l'installation des dépendances"
    exit 1
fi

# Lancer l'application Streamlit
echo "🚀 Lancement de l'application Streamlit..."
echo "📱 L'application sera accessible sur: http://localhost:8501"
echo "🔗 Pour l'arrêter, utilisez Ctrl+C"
echo ""

streamlit run app.py --server.port 8501 --server.address localhost
