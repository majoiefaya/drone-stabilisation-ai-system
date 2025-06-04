# 🚁 DroneStab AI - Application Streamlit

## 🚀 Démarrage Rapide

### 📋 Prérequis
- Python 3.8 ou supérieur
- pip (gestionnaire de packages Python)

### ⚡ Installation et Lancement

#### Windows
```bash
# Double-cliquer sur le fichier
start_app.bat

# Ou en ligne de commande
.\start_app.bat
```

#### Linux/macOS
```bash
# Rendre le script exécutable
chmod +x start_app.sh

# Lancer l'application
./start_app.sh
```

#### Installation Manuelle
```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py
```

### 🌐 Accès à l'Application

Une fois lancée, l'application sera accessible à l'adresse :
**http://localhost:8501**

## 🎮 Guide d'Utilisation

### 🏠 Page d'Accueil
- Vue d'ensemble du projet
- Métriques de performance
- Navigation vers les différentes fonctionnalités

### 📊 Exploration des Données
1. **Uploadez votre dataset** CSV de données de drone
2. **Visualisez** les distributions, corrélations, tendances
3. **Détectez** les outliers automatiquement
4. **Analysez** la qualité des données

### 🤖 Entraînement ML
1. **Configurez** les paramètres d'entraînement
2. **Sélectionnez** les modèles à comparer
3. **Lancez** l'entraînement automatique
4. **Suivez** le progrès en temps réel

### 📈 Évaluation
1. **Comparez** les performances des modèles
2. **Visualisez** prédictions vs réalité
3. **Analysez** les métriques par variable cible
4. **Identifiez** le meilleur modèle

### 🔍 Explicabilité
1. **Sélectionnez** un modèle pour l'analyse SHAP
2. **Explorez** l'importance des features
3. **Comprenez** les décisions du modèle
4. **Validez** la cohérence métier

### 🧪 Tests Avancés
1. **Validation Croisée** : Robustesse des modèles
2. **Test de Robustesse** : Résistance au bruit
3. **Test de Monotonie** : Cohérence logique

### 🚀 Prédiction Temps Réel
1. **Configurez** les données capteurs en direct
2. **Ajustez** les paramètres du drone
3. **Obtenez** les corrections instantanées
4. **Visualisez** en 3D les recommandations

## 🎨 Fonctionnalités de l'Interface

### ✨ Design Moderne
- Interface intuitive et responsive
- Navigation par onglets horizontaux
- Visualisations interactives Plotly
- Métriques en temps réel

### 🎛️ Contrôles Interactifs
- Sliders pour ajuster les paramètres
- Sélecteurs de fichiers par glisser-déposer
- Boutons d'action avec feedback visuel
- Barres de progression

### 📊 Visualisations Avancées
- Graphiques 2D et 3D interactifs
- Heatmaps de corrélation
- Box plots pour outliers
- Comparaisons multi-modèles

### 📱 Responsive Design
- Adaptation automatique à la taille d'écran
- Colonnes flexibles
- Sidebar rétractable
- Mobile-friendly

## 🔧 Fonctionnalités Techniques

### 🎯 Pipeline ML Complet
- Préprocessing automatique des données
- Entraînement de 4 modèles simultanément
- Validation croisée intégrée
- Sauvegarde/chargement des modèles

### 📊 Analyses Avancées
- Tests de robustesse au bruit
- Analyse de monotonie
- Explicabilité SHAP
- Diagnostic automatique

### 🚀 Performance Optimisée
- Mise en cache des résultats
- Calculs asynchrones
- Interface réactive
- Gestion d'erreurs robuste

## 🛠️ Structure du Code

```
app.py                          # Application Streamlit principale
├── DroneStabilizationPipeline  # Classe pipeline ML
├── show_home_page()           # Page d'accueil
├── show_data_exploration()    # Exploration données
├── show_model_training()      # Entraînement modèles
├── show_model_evaluation()    # Évaluation performances
├── show_model_explainability() # Explicabilité SHAP
├── show_advanced_tests()      # Tests avancés
└── show_real_time_prediction() # Prédiction temps réel
```

## 🎨 Personnalisation

### 🌈 Thème
Le thème est configurable dans `.streamlit/config.toml` :
- Couleurs primaires et secondaires
- Police et taille
- Mode sombre/clair

### 📊 Graphiques
Les visualisations utilisent Plotly pour :
- Interactivité maximale
- Zoom et pan
- Export en images
- Responsive design

### 🎛️ Interface
Personnalisation via CSS dans `app.py` :
- Cards avec bordures colorées
- Métriques avec indicateurs visuels
- Layout adaptatif

## 🔒 Sécurité

### 📁 Upload de Fichiers
- Limitation de taille (200MB max)
- Validation des formats CSV
- Nettoyage automatique des données

### 🛡️ Gestion d'Erreurs
- Try/catch sur toutes les opérations critiques
- Messages d'erreur utilisateur-friendly
- Fallbacks en cas d'échec

## 📊 Données Supportées

### 📋 Format Requis
Le dataset CSV doit contenir les colonnes suivantes :

**Features (13 colonnes):**
- `roll`, `pitch`, `yaw` : Orientation IMU (°)
- `ax`, `ay`, `az` : Accélérations (m/s²)
- `lat`, `lon`, `alt` : Position GPS
- `h1`, `h2`, `h3`, `h4` : État hélices (0-1)

**Targets (4 colonnes):**
- `delta_h1`, `delta_h2`, `delta_h3`, `delta_h4` : Corrections

### 📈 Exemples de Datasets
Des datasets d'exemple sont fournis :
- `drone_stabilization_dataset.csv` : Dataset principal
- `drone_stabilization_test_dataset.csv` : Données test
- `drone_takeoff_landing_or_indoor.csv` : Scenarios spécifiques

## 🚀 Déploiement

### 🌐 Déploiement Local
```bash
streamlit run app.py --server.port 8501
```

### ☁️ Déploiement Cloud
L'application est compatible avec :
- **Streamlit Cloud** : Déploiement gratuit
- **Heroku** : Avec Procfile
- **Docker** : Containerisation
- **AWS/Azure** : Cloud computing

### 🔗 Streamlit Cloud
1. Fork le repository
2. Connecter à Streamlit Cloud
3. Déployer automatiquement
4. Partager le lien public

## 🆘 Résolution de Problèmes

### ❓ Problèmes Fréquents

| Problème | Solution |
|----------|----------|
| **Erreur d'import** | `pip install -r requirements.txt` |
| **Port occupé** | Changer le port dans config.toml |
| **Upload échoue** | Vérifier format CSV et taille |
| **Modèles non entraînés** | Suivre l'ordre : Données → Entraînement → Évaluation |
| **Visualisations manquantes** | Réactualiser la page |

### 🔍 Debug
- Activer le mode debug dans config.toml
- Vérifier les logs dans le terminal
- Utiliser le widget de diagnostic intégré

## 📞 Support

### 🐛 Signaler un Bug
- Décrire le problème précisément
- Inclure les messages d'erreur
- Préciser l'environnement (OS, Python version)

### 💡 Demande de Fonctionnalité
- Expliquer le cas d'usage
- Proposer une interface utilisateur
- Estimer l'impact métier

## 🎉 Conclusion

Cette application Streamlit offre une interface complète et intuitive pour le pipeline de stabilisation de drone. Elle permet aux utilisateurs de tester facilement toutes les fonctionnalités développées dans le notebook Jupyter, avec une expérience utilisateur optimisée pour la démonstration et la production.

**🚀 Prêt à stabiliser des drones avec l'IA !**
