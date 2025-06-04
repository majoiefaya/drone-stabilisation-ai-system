# Dockerfile pour l'application DroneStab AI
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de requirements
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY . .

# Exposer le port Streamlit
EXPOSE 8501

# Créer un utilisateur non-root pour la sécurité
RUN useradd -m -u 1000 droneuser && chown -R droneuser:droneuser /app
USER droneuser

# Commande par défaut pour lancer l'application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
