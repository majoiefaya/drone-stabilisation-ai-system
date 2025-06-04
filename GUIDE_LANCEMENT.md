# 🚀 Guide de Lancement - DroneStab AI

## 📋 Méthodes de Lancement

### 🎯 **Méthode Rapide (Recommandée)**

#### Windows
```bash
# Double-clic sur le fichier ou dans PowerShell/CMD
start_app.bat
```

#### Linux/Mac
```bash
chmod +x start_app.sh
./start_app.sh
```

### 🛠️ **Méthode Manuelle**

#### 1. Vérifier Python
```bash
python --version
# Requis : Python 3.8+
```

#### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

#### 3. Lancer l'application
```bash
streamlit run app.py
```

### 🐳 **Méthode Docker**

```bash
# Option 1 : Docker Compose (recommandé)
docker-compose up --build

# Option 2 : Docker direct
docker build -t drone-stabilization .
docker run -p 8501:8501 drone-stabilization
```

## 🌐 Accès à l'Application

- **Local** : http://localhost:8501
- **Réseau** : http://votre-ip:8501

## ⚡ Démarrage Rapide (30 secondes)

```bash
# 1. Ouvrir terminal dans le dossier du projet
cd "chemin/vers/modele_de_stabilisation_dun_drone"

# 2. Installer dépendances (première fois uniquement)
pip install streamlit pandas numpy scikit-learn matplotlib plotly

# 3. Lancer l'app
streamlit run app.py
```

## 🔧 Résolution de Problèmes

### ❌ Erreur : "streamlit command not found"
```bash
# Solution
pip install streamlit
# ou
python -m pip install streamlit
```

### ❌ Erreur : "Module not found"
```bash
# Installer modules manquants
pip install -r requirements.txt
# ou installer individuellement
pip install pandas numpy scikit-learn
```

### ❌ Erreur : "Port already in use"
```bash
# Utiliser un autre port
streamlit run app.py --server.port 8502
```

### ❌ Erreur : "Permission denied"
```bash
# Windows : Exécuter en tant qu'administrateur
# Linux/Mac : 
sudo chmod +x start_app.sh
```

## 🎛️ Fonctionnalités Disponibles

Une fois l'application lancée, vous aurez accès à :

### 📊 **Analyse des Données**
- Exploration interactive des datasets
- Visualisations statistiques avancées
- Détection automatique d'anomalies

### 🤖 **Entraînement ML**
- Comparaison de 4 modèles ML
- Validation croisée automatique
- Optimisation hyperparamètres

### 🔍 **Explicabilité IA**
- Analyse SHAP interactive
- Importance des features
- Interprétation des prédictions

### 📈 **Évaluation Modèles**
- Métriques de performance
- Visualisations comparatives
- Tests de robustesse

### 🧪 **Tests Avancés**
- Simulation conditions extrêmes
- Analyse de sensibilité
- Validation par sous-groupes

### ⚡ **Prédictions Temps Réel**
- Interface de saisie manuelle
- Upload de fichiers test
- Résultats instantanés

## 💡 Conseils d'Utilisation

### 🚀 **Pour Débutants**
1. Commencer par "📊 Analyse des Données"
2. Explorer les visualisations
3. Tester "🤖 Entraînement Modèles"
4. Utiliser "⚡ Prédictions" pour tester

### 🔬 **Pour Experts**
1. Utiliser "🧪 Tests Avancés" pour validation rigoureuse
2. Explorer "🔍 Explicabilité" pour insights métier
3. Optimiser via "📈 Évaluation Comparative"
4. Déployer avec Docker pour production

## 📱 Accès Mobile

L'application est responsive et fonctionne sur mobile :
- Ouvrir http://votre-ip:8501 sur téléphone
- Interface adaptée automatiquement
- Toutes fonctionnalités disponibles

## 🌍 Déploiement Cloud

### Streamlit Cloud (Gratuit)
1. Push code sur GitHub
2. Connecter Streamlit Cloud
3. Déploiement automatique

### Heroku/AWS/GCP
1. Utiliser Dockerfile fourni
2. Suivre documentation plateforme
3. Configuration variables d'environnement

## 🔗 Liens Utiles

- **Documentation Streamlit** : https://docs.streamlit.io
- **Troubleshooting** : https://docs.streamlit.io/troubleshooting
- **Communauté** : https://discuss.streamlit.io

---

## ✅ Checklist de Vérification

Avant de lancer l'application, vérifiez :

- [ ] Python 3.8+ installé
- [ ] Dépendances installées (`pip install -r requirements.txt`)
- [ ] Fichiers de données présents (CSV)
- [ ] Port 8501 disponible
- [ ] Connexion internet pour packages

## 🎉 Succès !

Si tout fonctionne, vous devriez voir :
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

**🎯 L'application DroneStab AI est maintenant prête à l'emploi !**
