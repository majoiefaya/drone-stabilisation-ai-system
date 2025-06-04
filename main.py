"""
🚁 DroneStab AI - Pipeline de Stabilisation de Drone
Application Streamlit optimisée pour Streamlit Cloud

Auteur: Votre Nom
Version: 2.0
Déploiement: Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Configuration optimisée pour Streamlit Cloud
warnings.filterwarnings('ignore')
plt.style.use('default')

# Configuration de la page
st.set_page_config(
    page_title="🚁 DroneStab AI - Stabilisation Drone",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/votre-repo/drone-stabilization',
        'Report a bug': 'https://github.com/votre-repo/drone-stabilization/issues',
        'About': """
        # DroneStab AI
        Pipeline intelligent de stabilisation de drone avec IA.
        Développé avec Streamlit et déployé sur Streamlit Cloud.
        """
    }
)

# CSS optimisé pour Streamlit Cloud
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .success-metric {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .warning-metric {
        background: linear-gradient(135deg, #fcb045 0%, #fd1d1d 100%);
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 0 20px;
        font-weight: bold;
    }
    .upload-section {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Cache pour optimiser les performances sur Streamlit Cloud
@st.cache_data
def load_demo_data():
    """Charger des données de démonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    # Simulation de données de drone
    data = {
        'roll': np.random.normal(0, 5, n_samples),
        'pitch': np.random.normal(0, 5, n_samples),
        'yaw': np.random.normal(0, 10, n_samples),
        'ax': np.random.normal(0, 2, n_samples),
        'ay': np.random.normal(0, 2, n_samples),
        'az': np.random.normal(9.8, 1, n_samples),
        'lat': np.random.normal(48.8566, 0.01, n_samples),
        'lon': np.random.normal(2.3522, 0.01, n_samples),
        'alt': np.random.normal(100, 20, n_samples),
        'h1': np.random.uniform(0, 1, n_samples),
        'h2': np.random.uniform(0, 1, n_samples),
        'h3': np.random.uniform(0, 1, n_samples),
        'h4': np.random.uniform(0, 1, n_samples),
    }
    
    # Calcul des corrections basées sur la physique
    df = pd.DataFrame(data)
    df['delta_h1'] = -0.1 * df['roll'] + 0.05 * df['pitch'] + np.random.normal(0, 0.02, n_samples)
    df['delta_h2'] = 0.1 * df['roll'] + 0.05 * df['pitch'] + np.random.normal(0, 0.02, n_samples)
    df['delta_h3'] = -0.05 * df['roll'] - 0.1 * df['pitch'] + np.random.normal(0, 0.02, n_samples)
    df['delta_h4'] = 0.05 * df['roll'] - 0.1 * df['pitch'] + np.random.normal(0, 0.02, n_samples)
    
    return df

@st.cache_resource
class DroneStabilizationPipeline:
    """Pipeline optimisé pour Streamlit Cloud avec mise en cache"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = ['roll', 'pitch', 'yaw', 'ax', 'ay', 'az', 'lat', 'lon', 'alt', 'h1', 'h2', 'h3', 'h4']
        self.target_names = ['delta_h1', 'delta_h2', 'delta_h3', 'delta_h4']
        self.is_trained = False
        self.training_history = []
        
    def preprocess_data(self, df):
        """Préprocessing des données avec validation"""
        if df is None:
            return None, None
        
        # Vérification des colonnes requises
        missing_features = set(self.feature_names) - set(df.columns)
        missing_targets = set(self.target_names) - set(df.columns)
        
        if missing_features:
            st.error(f"❌ Colonnes features manquantes: {missing_features}")
            return None, None
        if missing_targets:
            st.error(f"❌ Colonnes targets manquantes: {missing_targets}")
            return None, None
        
        X = df[self.feature_names].copy()
        y = df[self.target_names].copy()
        
        # Nettoyage des données
        initial_size = len(X)
        X = X.dropna()
        y = y.loc[X.index]
        
        if len(X) < initial_size:
            st.warning(f"⚠️ {initial_size - len(X)} lignes supprimées (valeurs manquantes)")
        
        return X, y
    
    def train_models(self, X_train, y_train):
        """Entraîner les modèles avec barre de progression"""
        models_config = {
            'MLP Neural Network': MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=300,
                random_state=42,
                alpha=0.01
            ),
            'Random Forest': RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'Ridge Regression': Ridge(
                alpha=1.0,
                random_state=42
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=50,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
        }
        
        # Normalisation des données
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Interface de progression
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, (name, model) in enumerate(models_config.items()):
                status_text.text(f"🔄 Entraînement: {name}")
                
                try:
                    model.fit(X_train_scaled, y_train)
                    self.models[name] = model
                    
                    # Validation croisée rapide
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='neg_mean_squared_error')
                    
                    self.training_history.append({
                        'model': name,
                        'cv_score': -np.mean(cv_scores),
                        'cv_std': np.std(cv_scores)
                    })
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'entraînement de {name}: {e}")
                    continue
                
                progress_bar.progress((i + 1) / len(models_config))
            
            status_text.text("✅ Entraînement terminé!")
        
        self.is_trained = True
        return self.models
    
    def evaluate_models(self, X_test, y_test):
        """Évaluation complète des modèles"""
        if not self.is_trained:
            st.error("❌ Les modèles doivent être entraînés avant l'évaluation!")
            return None
        
        X_test_scaled = self.scaler.transform(X_test)
        results = {}
        
        for name, model in self.models.items():
            try:
                y_pred = model.predict(X_test_scaled)
                
                # Métriques d'évaluation
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Calcul de la précision par variable cible
                target_scores = {}
                for i, target in enumerate(self.target_names):
                    target_r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
                    target_scores[target] = target_r2
                
                results[name] = {
                    'MSE': mse,
                    'MAE': mae,
                    'R2': r2,
                    'predictions': y_pred,
                    'target_scores': target_scores
                }
                
            except Exception as e:
                st.error(f"❌ Erreur lors de l'évaluation de {name}: {e}")
                continue
        
        return results

def create_dashboard_metrics(results):
    """Créer un dashboard avec métriques visuelles"""
    if not results:
        return
    
    # Meilleur modèle
    best_model = min(results.keys(), key=lambda x: results[x]['MSE'])
    best_r2 = results[best_model]['R2']
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card success-metric">
            <h3>🏆 Meilleur Modèle</h3>
            <h2>{}</h2>
        </div>
        """.format(best_model), unsafe_allow_html=True)
    
    with col2:
        color_class = "success-metric" if best_r2 > 0.8 else "warning-metric" if best_r2 > 0.6 else "metric-card"
        st.markdown("""
        <div class="metric-card {}">
            <h3>📊 Score R²</h3>
            <h2>{:.3f}</h2>
        </div>
        """.format(color_class, best_r2), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 MSE</h3>
            <h2>{:.4f}</h2>
        </div>
        """.format(results[best_model]['MSE']), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 MAE</h3>
            <h2>{:.4f}</h2>
        </div>
        """.format(results[best_model]['MAE']), unsafe_allow_html=True)

def create_performance_comparison(results):
    """Graphique de comparaison des performances"""
    models = list(results.keys())
    metrics = ['MSE', 'MAE', 'R2']
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=metrics,
        specs=[[{'secondary_y': False}, {'secondary_y': False}, {'secondary_y': False}]]
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=values,
                name=metric,
                marker_color=colors[i % len(colors)],
                text=[f'{v:.3f}' for v in values],
                textposition='auto'
            ),
            row=1, col=i+1
        )
    
    fig.update_layout(
        title="🏁 Comparaison des Performances des Modèles",
        height=500,
        showlegend=False
    )
    
    return fig

def create_predictions_plot(y_test, results, model_name):
    """Graphique des prédictions vs réalité"""
    predictions = results[model_name]['predictions']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f'Correction {target}' for target in ['Hélice 1', 'Hélice 2', 'Hélice 3', 'Hélice 4']],
        vertical_spacing=0.1
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i in range(4):
        row = i // 2 + 1
        col = i % 2 + 1
        
        # Valeurs réelles
        fig.add_trace(
            go.Scatter(
                y=y_test.iloc[:, i].values[:100],  # Limiter pour la lisibilité
                mode='lines',
                name=f'Réel',
                line=dict(color='black', width=2),
                showlegend=(i == 0)
            ),
            row=row, col=col
        )
        
        # Prédictions
        fig.add_trace(
            go.Scatter(
                y=predictions[:100, i],
                mode='lines',
                name=f'Prédit',
                line=dict(color=colors[i], width=2, dash='dash'),
                showlegend=(i == 0)
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title=f"🎯 Prédictions vs Réalité - {model_name}",
        height=600,
        showlegend=True
    )
    
    return fig

def real_time_simulation():
    """Simulation temps réel pour démonstration"""
    st.markdown("### 🚁 Simulation Temps Réel")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Contrôles de simulation
        st.markdown("#### Paramètres de Vol")
        roll_input = st.slider("Roll (°)", -30, 30, 0)
        pitch_input = st.slider("Pitch (°)", -30, 30, 0)
        yaw_input = st.slider("Yaw (°)", -180, 180, 0)
        altitude = st.slider("Altitude (m)", 50, 200, 100)
        
        simulate_button = st.button("🚀 Simuler Correction", type="primary")
    
    with col2:
        if simulate_button:
            # Simulation des corrections
            corrections = {
                'Hélice 1': -0.1 * roll_input + 0.05 * pitch_input,
                'Hélice 2': 0.1 * roll_input + 0.05 * pitch_input,
                'Hélice 3': -0.05 * roll_input - 0.1 * pitch_input,
                'Hélice 4': 0.05 * roll_input - 0.1 * pitch_input
            }
            
            # Graphique en temps réel
            fig = go.Figure()
            
            for i, (name, value) in enumerate(corrections.items()):
                fig.add_trace(go.Bar(
                    x=[name],
                    y=[value],
                    name=name,
                    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][i]
                ))
            
            fig.update_layout(
                title="Corrections Calculées",
                yaxis_title="Correction (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Affichage des valeurs
            for name, value in corrections.items():
                color = "🟢" if abs(value) < 0.1 else "🟡" if abs(value) < 0.2 else "🔴"
                st.write(f"{color} **{name}**: {value:.3f}%")

def main():
    """Application principale optimisée pour Streamlit Cloud"""
    
    # En-tête avec animation
    st.markdown("""
    <div class="main-header">
        🚁 DroneStab AI - Pipeline de Stabilisation
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar avec informations
    with st.sidebar:
        st.markdown("### 🎛️ Panneau de Contrôle")
        st.markdown("---")
        
        # Sélection du mode
        mode = st.selectbox(
            "Mode d'utilisation:",
            ["🚀 Démonstration Rapide", "📊 Analyse Complète", "🔬 Mode Expert"]
        )
        
        st.markdown("---")
        st.markdown("""
        ### 📖 À Propos
        Cette application utilise l'IA pour prédire les corrections nécessaires 
        à la stabilisation d'un drone en temps réel.
        
        **Modèles supportés:**
        - 🧠 Réseau de Neurones (MLP)
        - 🌲 Random Forest
        - 📈 Régression Ridge
        - ⚡ XGBoost
        """)
        
        st.markdown("---")
        st.markdown("🏗️ **Développé avec Streamlit**")
        st.markdown("☁️ **Déployé sur Streamlit Cloud**")
    
    # Interface principale avec onglets
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Données", "🤖 Entraînement", "📈 Évaluation", "🚁 Simulation"])
    
    # Initialisation du pipeline
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = DroneStabilizationPipeline()
    
    pipeline = st.session_state.pipeline
    
    # Onglet 1: Gestion des données
    with tab1:
        st.markdown("### 📊 Gestion des Données")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Chargement des Données")
            
            # Option de chargement
            data_option = st.radio(
                "Source des données:",
                ["🎯 Utiliser les données de démonstration", "📁 Charger un fichier CSV"]
            )
            
            if data_option == "🎯 Utiliser les données de démonstration":
                if st.button("🚀 Charger les Données Démo", type="primary"):
                    with st.spinner("Génération des données de démonstration..."):
                        df = load_demo_data()
                        st.session_state.df = df
                        st.success("✅ Données de démonstration chargées!")
            
            else:
                uploaded_file = st.file_uploader(
                    "Sélectionnez votre fichier CSV",
                    type=['csv'],
                    help="Le fichier doit contenir les colonnes: roll, pitch, yaw, ax, ay, az, lat, lon, alt, h1, h2, h3, h4, delta_h1, delta_h2, delta_h3, delta_h4"
                )
                
                if uploaded_file is not None:
                    try:
                        df = pd.read_csv(uploaded_file)
                        st.session_state.df = df
                        st.success("✅ Fichier chargé avec succès!")
                    except Exception as e:
                        st.error(f"❌ Erreur lors du chargement: {e}")
        
        with col2:
            if 'df' in st.session_state:
                st.markdown("#### 📋 Informations")
                df = st.session_state.df
                
                st.info(f"""
                **📊 Statistiques:**
                - Lignes: {len(df):,}
                - Colonnes: {len(df.columns)}
                - Taille: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB
                """)
        
        # Aperçu des données
        if 'df' in st.session_state:
            st.markdown("#### 🔍 Aperçu des Données")
            df = st.session_state.df
            
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(df.head(), use_container_width=True)
            
            with col2:
                st.markdown("**📊 Statistiques Descriptives:**")
                st.dataframe(df.describe(), use_container_width=True)
    
    # Onglet 2: Entraînement
    with tab2:
        st.markdown("### 🤖 Entraînement des Modèles")
        
        if 'df' not in st.session_state:
            st.warning("⚠️ Veuillez d'abord charger des données dans l'onglet 'Données'")
        else:
            df = st.session_state.df
            
            # Préprocessing
            X, y = pipeline.preprocess_data(df)
            
            if X is not None and y is not None:
                st.success(f"✅ Données préparées: {len(X)} échantillons, {len(X.columns)} features")
                
                # Division des données
                test_size = st.slider("Taille du jeu de test (%)", 10, 40, 20) / 100
                
                if st.button("🚀 Lancer l'Entraînement", type="primary"):
                    with st.spinner("Entraînement en cours..."):
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42
                        )
                        
                        # Sauvegarde pour évaluation
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        
                        # Entraînement
                        models = pipeline.train_models(X_train, y_train)
                        
                        st.balloons()
                        st.success("🎉 Entraînement terminé avec succès!")
                        
                        # Résumé de l'entraînement
                        st.markdown("#### 📋 Résumé de l'Entraînement")
                        summary_df = pd.DataFrame(pipeline.training_history)
                        st.dataframe(summary_df, use_container_width=True)
    
    # Onglet 3: Évaluation
    with tab3:
        st.markdown("### 📈 Évaluation des Modèles")
        
        if not pipeline.is_trained:
            st.warning("⚠️ Veuillez d'abord entraîner les modèles dans l'onglet 'Entraînement'")
        else:
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            
            # Évaluation
            results = pipeline.evaluate_models(X_test, y_test)
            
            if results:
                # Dashboard des métriques
                create_dashboard_metrics(results)
                
                st.markdown("---")
                
                # Graphiques de comparaison
                col1, col2 = st.columns(2)
                
                with col1:
                    # Comparaison des performances
                    fig_comparison = create_performance_comparison(results)
                    st.plotly_chart(fig_comparison, use_container_width=True)
                
                with col2:
                    # Sélection du modèle pour visualisation détaillée
                    selected_model = st.selectbox(
                        "Modèle à analyser:",
                        list(results.keys()),
                        index=0
                    )
                    
                    # Graphique prédictions vs réalité
                    fig_pred = create_predictions_plot(y_test, results, selected_model)
                    st.plotly_chart(fig_pred, use_container_width=True)
                
                # Tableau détaillé des résultats
                st.markdown("#### 📊 Résultats Détaillés")
                results_df = pd.DataFrame({
                    'Modèle': list(results.keys()),
                    'MSE': [results[m]['MSE'] for m in results.keys()],
                    'MAE': [results[m]['MAE'] for m in results.keys()],
                    'R²': [results[m]['R2'] for m in results.keys()]
                }).round(4)
                
                st.dataframe(results_df, use_container_width=True)
    
    # Onglet 4: Simulation temps réel
    with tab4:
        real_time_simulation()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        🚁 DroneStab AI - Développé avec ❤️ et Streamlit | 
        📧 Contact: votre.email@example.com | 
        🔗 <a href='https://github.com/votre-repo'>GitHub</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()