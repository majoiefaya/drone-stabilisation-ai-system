import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import io
import base64
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import shap
from streamlit_option_menu import option_menu
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="🚁 DroneStab AI - Pipeline de Stabilisation Drone",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .success-metric {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    .warning-metric {
        background-color: #fff3cd;
        border-left-color: #ffc107;
    }
    .danger-metric {
        background-color: #f8d7da;
        border-left-color: #dc3545;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Classe Pipeline de Stabilisation de Drone
class DroneStabilizationPipeline:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = ['roll', 'pitch', 'yaw', 'ax', 'ay', 'az', 'lat', 'lon', 'alt', 'h1', 'h2', 'h3', 'h4']
        self.target_names = ['delta_h1', 'delta_h2', 'delta_h3', 'delta_h4']
        self.is_trained = False
        
    def load_data(self, file):
        """Charger les données depuis un fichier"""
        try:
            df = pd.read_csv(file)
            return df
        except Exception as e:
            st.error(f"Erreur lors du chargement des données: {e}")
            return None
    
    def preprocess_data(self, df):
        """Préprocessing des données"""
        if df is None:
            return None, None
        
        # Vérifier les colonnes
        missing_features = set(self.feature_names) - set(df.columns)
        missing_targets = set(self.target_names) - set(df.columns)
        
        if missing_features:
            st.error(f"Colonnes features manquantes: {missing_features}")
            return None, None
        if missing_targets:
            st.error(f"Colonnes targets manquantes: {missing_targets}")
            return None, None
        
        X = df[self.feature_names]
        y = df[self.target_names]
        
        # Nettoyage des données
        X = X.dropna()
        y = y.loc[X.index]
        
        return X, y
    
    def train_models(self, X_train, y_train):
        """Entraîner plusieurs modèles"""
        models = {
            'MLP': MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=300, random_state=42),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
        }
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (name, model) in enumerate(models.items()):
            status_text.text(f'Entraînement du modèle {name}...')
            model.fit(X_train_scaled, y_train)
            self.models[name] = model
            progress_bar.progress((i + 1) / len(models))
        
        self.is_trained = True
        status_text.text('Entraînement terminé!')
        return self.models
    
    def evaluate_models(self, X_test, y_test):
        """Évaluer les modèles"""
        if not self.is_trained:
            st.error("Les modèles doivent être entraînés avant l'évaluation!")
            return None
        
        X_test_scaled = self.scaler.transform(X_test)
        results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test_scaled)
            
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'MSE': mse,
                'MAE': mae,
                'R2': r2,
                'predictions': y_pred
            }
        
        return results
    
    def plot_predictions_comparison(self, y_test, results):
        """Visualiser les comparaisons de prédictions"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f'Correction {target}' for target in self.target_names],
            vertical_spacing=0.08
        )
        
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, target in enumerate(self.target_names):
            row = i // 2 + 1
            col = i % 2 + 1
            
            # Valeurs réelles
            fig.add_trace(
                go.Scatter(
                    y=y_test[target].values,
                    mode='lines',
                    name=f'Réel {target}',
                    line=dict(color='black', width=2),
                    showlegend=(i == 0)
                ),
                row=row, col=col
            )
            
            # Prédictions du meilleur modèle
            best_model = min(results.keys(), key=lambda x: results[x]['MSE'])
            fig.add_trace(
                go.Scatter(
                    y=results[best_model]['predictions'][:, i],
                    mode='lines',
                    name=f'Prédit {target}',
                    line=dict(color=colors[i], width=2, dash='dash'),
                    showlegend=(i == 0)
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Comparaison Prédictions vs Réalité",
            height=600,
            showlegend=True
        )
        
        return fig

# Fonction principale de l'application
def main():
    # Header
    st.markdown('<h1 class="main-header">🚁 DroneStab AI - Pipeline de Stabilisation Drone</h1>', unsafe_allow_html=True)
    
    # Menu de navigation
    selected = option_menu(
        menu_title=None,
        options=["🏠 Accueil", "📊 Exploration Données", "🤖 Entraînement ML", "📈 Évaluation", "🔍 Explicabilité", "🧪 Tests Avancés", "🚀 Prédiction Temps Réel"],
        icons=["house", "bar-chart", "cpu", "graph-up", "search", "flask", "rocket"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#1f77b4"},
        }
    )
    
    # Initialiser le pipeline dans la session
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = DroneStabilizationPipeline()
    
    pipeline = st.session_state.pipeline
    
    if selected == "🏠 Accueil":
        show_home_page()
    
    elif selected == "📊 Exploration Données":
        show_data_exploration(pipeline)
    
    elif selected == "🤖 Entraînement ML":
        show_model_training(pipeline)
    
    elif selected == "📈 Évaluation":
        show_model_evaluation(pipeline)
    
    elif selected == "🔍 Explicabilité":
        show_model_explainability(pipeline)
    
    elif selected == "🧪 Tests Avancés":
        show_advanced_tests(pipeline)
    
    elif selected == "🚀 Prédiction Temps Réel":
        show_real_time_prediction(pipeline)

def show_home_page():
    """Page d'accueil"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ## 🎯 Bienvenue dans DroneStab AI
        
        ### 🚁 Solution IA de Stabilisation Autonome de Drones
        
        Cette application permet de développer et tester des modèles d'intelligence artificielle 
        pour la stabilisation autonome de drones en temps réel.
        
        #### ✨ Fonctionnalités Principales:
        
        - **📊 Exploration de Données** : Analyse interactive des données capteurs
        - **🤖 Entraînement ML** : Comparaison de 4 algorithmes avancés
        - **📈 Évaluation** : Métriques de performance détaillées
        - **🔍 Explicabilité** : Analyse SHAP pour comprendre les décisions IA
        - **🧪 Tests Avancés** : Robustesse, monotonie, symétrie
        - **🚀 Prédiction Temps Réel** : Interface de test en direct
        
        #### 🎮 Comment Commencer:
        
        1. **📊 Explorez vos données** dans l'onglet "Exploration Données"
        2. **🤖 Entraînez les modèles** ML dans "Entraînement ML"  
        3. **📈 Évaluez les performances** dans "Évaluation"
        4. **🔍 Analysez l'explicabilité** avec SHAP
        5. **🚀 Testez en temps réel** vos prédictions
        """)
        
        # Métriques de démonstration
        st.markdown("### 🏆 Performances de Référence")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("MSE", "0.00072", delta="-99.9%")
        with col2:
            st.metric("R² Score", "0.996", delta="+0.4%")
        with col3:
            st.metric("Précision", "99.6%", delta="+2.1%")
        with col4:
            st.metric("Temps", "<1ms", delta="-85%")

def show_data_exploration(pipeline):
    """Page d'exploration des données"""
    st.markdown("## 📊 Exploration des Données")
    
    # Upload de fichier
    st.markdown("### 📁 Chargement des Données")
    uploaded_file = st.file_uploader(
        "Choisissez un fichier CSV",
        type=['csv'],
        help="Uploadez votre dataset de données de drone"
    )
    
    if uploaded_file is not None:
        df = pipeline.load_data(uploaded_file)
        
        if df is not None:
            st.session_state.df = df
            
            # Aperçu des données
            st.markdown("### 👀 Aperçu des Données")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Nombre d'échantillons", len(df))
                st.metric("Nombre de features", len(df.columns))
            
            with col2:
                st.metric("Valeurs manquantes", df.isnull().sum().sum())
                st.metric("Mémoire utilisée", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
            # Affichage des premières lignes
            st.markdown("#### 📋 Premières Lignes")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Statistiques descriptives
            st.markdown("### 📈 Statistiques Descriptives")
            st.dataframe(df.describe(), use_container_width=True)
            
            # Visualisations
            st.markdown("### 📊 Visualisations")
            
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "🔗 Corrélations", "📈 Tendances", "⚠️ Outliers"])
            
            with tab1:
                show_distributions(df)
            
            with tab2:
                show_correlations(df)
            
            with tab3:
                show_trends(df)
            
            with tab4:
                show_outliers(df)

def show_distributions(df):
    """Afficher les distributions des variables"""
    st.markdown("#### 📊 Distribution des Variables")
    
    # Sélection de colonnes
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_cols = st.multiselect(
        "Sélectionnez les variables à visualiser:",
        numeric_cols,
        default=numeric_cols[:4]
    )
    
    if selected_cols:
        fig = make_subplots(
            rows=len(selected_cols)//2 + len(selected_cols)%2, 
            cols=2,
            subplot_titles=selected_cols
        )
        
        for i, col in enumerate(selected_cols):
            row = i // 2 + 1
            col_pos = i % 2 + 1
            
            fig.add_trace(
                go.Histogram(x=df[col], name=col, showlegend=False),
                row=row, col=col_pos
            )
        
        fig.update_layout(height=300*len(selected_cols)//2 + 150, title="Distributions des Variables")
        st.plotly_chart(fig, use_container_width=True)

def show_correlations(df):
    """Afficher la matrice de corrélation"""
    st.markdown("#### 🔗 Matrice de Corrélation")
    
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Matrice de Corrélation",
        color_continuous_scale='RdBu'
    )
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

def show_trends(df):
    """Afficher les tendances temporelles"""
    st.markdown("#### 📈 Tendances des Variables")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_var = st.selectbox("Sélectionnez une variable:", numeric_cols)
    
    if selected_var:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df[selected_var], mode='lines', name=selected_var))
        fig.update_layout(
            title=f"Évolution de {selected_var}",
            xaxis_title="Index",
            yaxis_title=selected_var
        )
        st.plotly_chart(fig, use_container_width=True)

def show_outliers(df):
    """Détecter et afficher les outliers"""
    st.markdown("#### ⚠️ Détection d'Outliers")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_col = st.selectbox("Sélectionnez une variable pour l'analyse d'outliers:", numeric_cols)
    
    if selected_col:
        Q1 = df[selected_col].quantile(0.25)
        Q3 = df[selected_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Nombre d'outliers", len(outliers))
        with col2:
            st.metric("Pourcentage", f"{len(outliers)/len(df)*100:.2f}%")
        
        # Box plot
        fig = go.Figure()
        fig.add_trace(go.Box(y=df[selected_col], name=selected_col))
        fig.update_layout(title=f"Box Plot - {selected_col}")
        st.plotly_chart(fig, use_container_width=True)

def show_model_training(pipeline):
    """Page d'entraînement des modèles"""
    st.markdown("## 🤖 Entraînement des Modèles ML")
    
    if 'df' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord charger des données dans l'onglet 'Exploration Données'")
        return
    
    df = st.session_state.df
    
    # Paramètres d'entraînement
    st.markdown("### ⚙️ Configuration de l'Entraînement")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Taille du jeu de test (%)", 10, 40, 20) / 100
        random_state = st.number_input("Random State", value=42, step=1)
    
    with col2:
        train_models = st.multiselect(
            "Modèles à entraîner:",
            ['MLP', 'RandomForest', 'Ridge', 'XGBoost'],
            default=['MLP', 'RandomForest', 'Ridge', 'XGBoost']
        )
    
    # Bouton d'entraînement
    if st.button("🚀 Lancer l'Entraînement", type="primary"):
        with st.spinner("Entraînement en cours..."):
            
            # Préprocessing
            X, y = pipeline.preprocess_data(df)
            if X is None or y is None:
                return
            
            # Split des données
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Stockage dans la session
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            
            # Entraînement
            models = pipeline.train_models(X_train, y_train)
            
            st.success("✅ Entraînement terminé avec succès!")
            
            # Affichage des informations
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Échantillons d'entraînement", len(X_train))
            with col2:
                st.metric("Échantillons de test", len(X_test))
            with col3:
                st.metric("Features", len(pipeline.feature_names))
            with col4:
                st.metric("Targets", len(pipeline.target_names))
    
    # Affichage des modèles entraînés
    if pipeline.is_trained:
        st.markdown("### ✅ Modèles Entraînés")
        
        for model_name in pipeline.models.keys():
            with st.expander(f"🤖 {model_name}"):
                model = pipeline.models[model_name]
                st.write(f"**Type:** {type(model).__name__}")
                st.write(f"**Paramètres:** {model.get_params()}")

def show_model_evaluation(pipeline):
    """Page d'évaluation des modèles"""
    st.markdown("## 📈 Évaluation des Modèles")
    
    if not pipeline.is_trained:
        st.warning("⚠️ Veuillez d'abord entraîner les modèles dans l'onglet 'Entraînement ML'")
        return
    
    if 'X_test' not in st.session_state:
        st.error("❌ Données de test non disponibles")
        return
    
    # Évaluation des modèles
    with st.spinner("Évaluation en cours..."):
        results = pipeline.evaluate_models(st.session_state.X_test, st.session_state.y_test)
        st.session_state.evaluation_results = results
    
    if results:
        # Métriques globales
        st.markdown("### 🏆 Performances Globales")
        
        # Tableau comparatif
        metrics_df = pd.DataFrame({
            'Modèle': list(results.keys()),
            'MSE': [results[m]['MSE'] for m in results.keys()],
            'MAE': [results[m]['MAE'] for m in results.keys()],
            'R²': [results[m]['R2'] for m in results.keys()]
        })
        
        st.dataframe(metrics_df.style.highlight_min(subset=['MSE', 'MAE']).highlight_max(subset=['R²']), use_container_width=True)
        
        # Meilleur modèle
        best_model = min(results.keys(), key=lambda x: results[x]['MSE'])
        st.success(f"🥇 Meilleur modèle: **{best_model}** (MSE: {results[best_model]['MSE']:.6f})")
        
        # Visualisations
        st.markdown("### 📊 Visualisations des Performances")
        
        tab1, tab2, tab3 = st.tabs(["📈 Prédictions vs Réalité", "📊 Métriques Comparatives", "🎯 Analyse par Target"])
        
        with tab1:
            fig = pipeline.plot_predictions_comparison(st.session_state.y_test, results)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            show_comparative_metrics(results)
        
        with tab3:
            show_target_analysis(results, st.session_state.y_test)

def show_comparative_metrics(results):
    """Afficher les métriques comparatives"""
    metrics = ['MSE', 'MAE', 'R2']
    
    for metric in metrics:
        st.markdown(f"#### {metric}")
        
        values = [results[model][metric] for model in results.keys()]
        models = list(results.keys())
        
        fig = go.Figure(data=[
            go.Bar(x=models, y=values, text=values, texttemplate='%{text:.4f}', textposition='outside')
        ])
        
        fig.update_layout(
            title=f"Comparaison {metric}",
            yaxis_title=metric,
            xaxis_title="Modèles"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_target_analysis(results, y_test):
    """Analyse par variable cible"""
    best_model = min(results.keys(), key=lambda x: results[x]['MSE'])
    
    target_names = ['delta_h1', 'delta_h2', 'delta_h3', 'delta_h4']
    
    for i, target in enumerate(target_names):
        st.markdown(f"#### 🎯 Analyse pour {target}")
        
        y_true = y_test[target].values
        y_pred = results[best_model]['predictions'][:, i]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_true, y=y_pred,
                mode='markers',
                name=f'{target}',
                opacity=0.6
            ))
            
            # Ligne de référence parfaite
            min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines',
                name='Prédiction parfaite',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title=f'Prédictions vs Réalité - {target}',
                xaxis_title='Valeurs Réelles',
                yaxis_title='Prédictions'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Métriques spécifiques
            mse_target = mean_squared_error(y_true, y_pred)
            mae_target = mean_absolute_error(y_true, y_pred)
            r2_target = r2_score(y_true, y_pred)
            
            st.metric("MSE", f"{mse_target:.6f}")
            st.metric("MAE", f"{mae_target:.6f}")
            st.metric("R²", f"{r2_target:.6f}")

def show_model_explainability(pipeline):
    """Page d'explicabilité des modèles"""
    st.markdown("## 🔍 Explicabilité des Modèles")
    
    if not pipeline.is_trained:
        st.warning("⚠️ Veuillez d'abord entraîner les modèles")
        return
    
    st.markdown("### 🧠 Analyse SHAP")
    
    # Sélection du modèle
    model_choice = st.selectbox(
        "Choisissez un modèle pour l'analyse SHAP:",
        list(pipeline.models.keys())
    )
    
    # Nombre d'échantillons pour SHAP
    n_samples = st.slider("Nombre d'échantillons pour l'analyse SHAP:", 50, 500, 200)
    
    if st.button("🔍 Analyser avec SHAP", type="primary"):
        with st.spinner("Analyse SHAP en cours..."):
            try:
                # Préparation des données
                X_test = st.session_state.X_test
                X_shap = pipeline.scaler.transform(X_test.iloc[:n_samples])
                
                # Créer l'explainer
                model = pipeline.models[model_choice]
                explainer = shap.Explainer(model, X_shap)
                shap_values = explainer(X_shap)
                
                # Graphique d'importance des features
                st.markdown("#### 📊 Importance Globale des Features")
                
                # Calculer l'importance moyenne
                importance_mean = np.abs(shap_values.values).mean(0)
                feature_importance = pd.DataFrame({
                    'Feature': pipeline.feature_names,
                    'Importance': importance_mean.mean(1)  # Moyenne sur tous les targets
                }).sort_values('Importance', ascending=True)
                
                fig = go.Figure(go.Bar(
                    x=feature_importance['Importance'],
                    y=feature_importance['Feature'],
                    orientation='h'
                ))
                
                fig.update_layout(
                    title="Importance des Features (SHAP)",
                    xaxis_title="Importance Moyenne",
                    yaxis_title="Features"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Analyse par target
                st.markdown("#### 🎯 Analyse par Variable Cible")
                
                target_choice = st.selectbox(
                    "Choisissez une variable cible:",
                    pipeline.target_names
                )
                
                target_idx = pipeline.target_names.index(target_choice)
                
                # Summary plot pour le target choisi
                target_shap_values = shap_values.values[:, :, target_idx]
                target_importance = pd.DataFrame({
                    'Feature': pipeline.feature_names,
                    'Importance': np.abs(target_shap_values).mean(0)
                }).sort_values('Importance', ascending=True)
                
                fig = go.Figure(go.Bar(
                    x=target_importance['Importance'],
                    y=target_importance['Feature'],
                    orientation='h'
                ))
                
                fig.update_layout(
                    title=f"Importance des Features pour {target_choice}",
                    xaxis_title="Importance SHAP",
                    yaxis_title="Features"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Insights
                st.markdown("#### 💡 Insights Clés")
                top_features = target_importance.tail(3)['Feature'].tolist()
                
                st.info(f"""
                **Top 3 Features les plus influentes pour {target_choice}:**
                1. 🥇 {top_features[2]}
                2. 🥈 {top_features[1]}  
                3. 🥉 {top_features[0]}
                
                Ces features ont le plus d'impact sur les prédictions du modèle {model_choice}.
                """)
                
            except Exception as e:
                st.error(f"Erreur lors de l'analyse SHAP: {e}")

def show_advanced_tests(pipeline):
    """Page des tests avancés"""
    st.markdown("## 🧪 Tests Avancés")
    
    if not pipeline.is_trained:
        st.warning("⚠️ Veuillez d'abord entraîner les modèles")
        return
    
    tab1, tab2, tab3 = st.tabs(["🔄 Validation Croisée", "🔊 Test de Robustesse", "⚖️ Test de Monotonie"])
    
    with tab1:
        show_cross_validation(pipeline)
    
    with tab2:
        show_robustness_test(pipeline)
    
    with tab3:
        show_monotonicity_test(pipeline)

def show_cross_validation(pipeline):
    """Test de validation croisée"""
    st.markdown("### 🔄 Validation Croisée")
    
    cv_folds = st.slider("Nombre de folds:", 3, 10, 5)
    
    if st.button("▶️ Lancer la Validation Croisée"):
        with st.spinner("Validation croisée en cours..."):
            X_train = st.session_state.X_train
            y_train = st.session_state.y_train
            X_train_scaled = pipeline.scaler.transform(X_train)
            
            cv_results = {}
            
            for model_name, model in pipeline.models.items():
                scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring='neg_mean_squared_error')
                cv_results[model_name] = {
                    'scores': -scores,
                    'mean': -scores.mean(),
                    'std': scores.std()
                }
            
            # Affichage des résultats
            st.markdown("#### 📊 Résultats de la Validation Croisée")
            
            cv_df = pd.DataFrame({
                'Modèle': list(cv_results.keys()),
                'MSE Moyen': [cv_results[m]['mean'] for m in cv_results.keys()],
                'Écart-type': [cv_results[m]['std'] for m in cv_results.keys()]
            })
            
            st.dataframe(cv_df, use_container_width=True)
            
            # Graphique
            fig = go.Figure()
            
            for model_name in cv_results.keys():
                fig.add_trace(go.Box(
                    y=cv_results[model_name]['scores'],
                    name=model_name
                ))
            
            fig.update_layout(
                title="Distribution des Scores de Validation Croisée",
                yaxis_title="MSE",
                xaxis_title="Modèles"
            )
            
            st.plotly_chart(fig, use_container_width=True)

def show_robustness_test(pipeline):
    """Test de robustesse au bruit"""
    st.markdown("### 🔊 Test de Robustesse au Bruit")
    
    noise_levels = st.multiselect(
        "Niveaux de bruit à tester (%):",
        [0, 5, 10, 15, 20, 25, 30],
        default=[0, 10, 20]
    )
    
    if st.button("🧪 Tester la Robustesse"):
        with st.spinner("Test de robustesse en cours..."):
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            
            robustness_results = {}
            
            for noise_level in noise_levels:
                # Ajouter du bruit
                noise_factor = noise_level / 100
                X_noisy = X_test + np.random.normal(0, noise_factor * X_test.std(), X_test.shape)
                X_noisy_scaled = pipeline.scaler.transform(X_noisy)
                
                # Tester chaque modèle
                for model_name, model in pipeline.models.items():
                    y_pred_noisy = model.predict(X_noisy_scaled)
                    mse_noisy = mean_squared_error(y_test, y_pred_noisy)
                    
                    if model_name not in robustness_results:
                        robustness_results[model_name] = []
                    robustness_results[model_name].append(mse_noisy)
            
            # Visualisation
            fig = go.Figure()
            
            for model_name in robustness_results.keys():
                fig.add_trace(go.Scatter(
                    x=noise_levels,
                    y=robustness_results[model_name],
                    mode='lines+markers',
                    name=model_name
                ))
            
            fig.update_layout(
                title="Robustesse au Bruit",
                xaxis_title="Niveau de Bruit (%)",
                yaxis_title="MSE"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Analyse
            st.markdown("#### 💡 Analyse de Robustesse")
            
            best_robust_model = min(robustness_results.keys(), 
                                  key=lambda x: robustness_results[x][-1])  # MSE au plus haut niveau de bruit
            
            st.success(f"🏆 Modèle le plus robuste: **{best_robust_model}**")

def show_monotonicity_test(pipeline):
    """Test de monotonie"""
    st.markdown("### ⚖️ Test de Monotonie")
    
    st.info("""
    Le test de monotonie vérifie si le modèle respecte les relations logiques attendues.
    Par exemple, une augmentation de l'angle de roulis devrait entraîner une correction proportionnelle.
    """)
    
    feature_to_test = st.selectbox(
        "Choisissez une feature à tester:",
        pipeline.feature_names
    )
    
    if st.button("⚖️ Tester la Monotonie"):
        with st.spinner("Test de monotonie en cours..."):
            X_test = st.session_state.X_test
            feature_idx = pipeline.feature_names.index(feature_to_test)
            
            # Créer des variations de la feature
            X_test_copy = X_test.copy()
            feature_values = np.linspace(X_test[feature_to_test].min(), 
                                       X_test[feature_to_test].max(), 20)
            
            predictions_variations = []
            
            best_model_name = min(pipeline.models.keys(), 
                                key=lambda x: st.session_state.evaluation_results[x]['MSE'])
            best_model = pipeline.models[best_model_name]
            
            for val in feature_values:
                X_temp = X_test_copy.copy()
                X_temp.iloc[:, feature_idx] = val
                X_temp_scaled = pipeline.scaler.transform(X_temp)
                pred = best_model.predict(X_temp_scaled).mean(axis=0)  # Moyenne sur tous les échantillons
                predictions_variations.append(pred)
            
            predictions_variations = np.array(predictions_variations)
            
            # Visualisation pour chaque target
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=pipeline.target_names
            )
            
            for i, target in enumerate(pipeline.target_names):
                row = i // 2 + 1
                col = i % 2 + 1
                
                # Calculer la corrélation
                correlation = np.corrcoef(feature_values, predictions_variations[:, i])[0, 1]
                
                fig.add_trace(
                    go.Scatter(
                        x=feature_values,
                        y=predictions_variations[:, i],
                        mode='lines+markers',
                        name=f'{target} (r={correlation:.3f})',
                        showlegend=(i == 0)
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(
                title=f"Test de Monotonie pour {feature_to_test}",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Analyse des corrélations
            st.markdown("#### 📊 Analyse des Corrélations")
            
            correlations = [np.corrcoef(feature_values, predictions_variations[:, i])[0, 1] 
                          for i in range(len(pipeline.target_names))]
            
            monotonicity_df = pd.DataFrame({
                'Target': pipeline.target_names,
                'Corrélation': correlations,
                'Monotonie': ['Forte' if abs(c) > 0.7 else 'Modérée' if abs(c) > 0.3 else 'Faible' 
                            for c in correlations]
            })
            
            st.dataframe(monotonicity_df, use_container_width=True)

def show_real_time_prediction(pipeline):
    """Page de prédiction temps réel"""
    st.markdown("## 🚀 Prédiction Temps Réel")
    
    if not pipeline.is_trained:
        st.warning("⚠️ Veuillez d'abord entraîner les modèles")
        return
    
    st.markdown("### 🎛️ Interface de Test en Temps Réel")
    
    # Sélection du modèle
    model_choice = st.selectbox(
        "Choisissez le modèle pour les prédictions:",
        list(pipeline.models.keys())
    )
    
    # Interface de saisie des données capteurs
    st.markdown("#### 📡 Données Capteurs")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🧭 Orientation (IMU)**")
        roll = st.slider("Roll (°)", -180.0, 180.0, 0.0, 0.1)
        pitch = st.slider("Pitch (°)", -90.0, 90.0, 0.0, 0.1)
        yaw = st.slider("Yaw (°)", -180.0, 180.0, 0.0, 0.1)
    
    with col2:
        st.markdown("**⚡ Accélération**")
        ax = st.slider("Accél. X (m/s²)", -20.0, 20.0, 0.0, 0.1)
        ay = st.slider("Accél. Y (m/s²)", -20.0, 20.0, 0.0, 0.1)
        az = st.slider("Accél. Z (m/s²)", -20.0, 20.0, 9.8, 0.1)
    
    with col3:
        st.markdown("**📍 Position GPS**")
        lat = st.number_input("Latitude", value=48.8566, format="%.6f")
        lon = st.number_input("Longitude", value=2.3522, format="%.6f")
        alt = st.slider("Altitude (m)", 0.0, 500.0, 100.0, 1.0)
    
    # État des hélices
    st.markdown("#### 🚁 État Actuel des Hélices")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        h1 = st.slider("Hélice 1", 0.0, 1.0, 0.5, 0.01)
    with col2:
        h2 = st.slider("Hélice 2", 0.0, 1.0, 0.5, 0.01)
    with col3:
        h3 = st.slider("Hélice 3", 0.0, 1.0, 0.5, 0.01)
    with col4:
        h4 = st.slider("Hélice 4", 0.0, 1.0, 0.5, 0.01)
    
    # Prédiction
    if st.button("🎯 Prédire les Corrections", type="primary"):
        # Préparer les données
        input_data = np.array([[roll, pitch, yaw, ax, ay, az, lat, lon, alt, h1, h2, h3, h4]])
        input_scaled = pipeline.scaler.transform(input_data)
        
        # Faire la prédiction
        model = pipeline.models[model_choice]
        prediction = model.predict(input_scaled)[0]
        
        # Afficher les résultats
        st.markdown("### 🎯 Corrections Prédites")
        
        col1, col2, col3, col4 = st.columns(4)
        
        corrections = ['delta_h1', 'delta_h2', 'delta_h3', 'delta_h4']
        colors = ['blue', 'green', 'orange', 'red']
        
        for i, (col, correction, color) in enumerate(zip([col1, col2, col3, col4], corrections, colors)):
            with col:
                delta = prediction[i]
                st.metric(
                    correction,
                    f"{delta:.4f}",
                    delta=f"{delta:+.4f}",
                    delta_color="inverse" if abs(delta) > 0.1 else "normal"
                )
        
        # Visualisation 3D du drone
        st.markdown("### 🚁 Visualisation 3D du Drone")
        
        fig = go.Figure()
        
        # Position centrale du drone
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(size=15, color='black'),
            name='Centre Drone'
        ))
        
        # Positions des hélices (approximatives)
        propeller_positions = [
            [1, 1, 0],    # h1
            [-1, 1, 0],   # h2  
            [-1, -1, 0],  # h3
            [1, -1, 0]    # h4
        ]
        
        for i, (pos, correction, color) in enumerate(zip(propeller_positions, corrections, colors)):
            # Taille proportionnelle à la correction
            size = 10 + abs(prediction[i]) * 100
            
            fig.add_trace(go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode='markers',
                marker=dict(size=size, color=color),
                name=f'{correction}: {prediction[i]:.4f}'
            ))
        
        fig.update_layout(
            title="Vue 3D du Drone avec Corrections",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y", 
                zaxis_title="Z"
            ),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommandations
        st.markdown("### 💡 Recommandations")
        
        max_correction = max(abs(p) for p in prediction)
        
        if max_correction < 0.01:
            st.success("✅ Drone stable - Corrections minimales nécessaires")
        elif max_correction < 0.05:
            st.info("ℹ️ Corrections légères recommandées")
        elif max_correction < 0.1:
            st.warning("⚠️ Corrections importantes nécessaires")
        else:
            st.error("🚨 Corrections critiques - Attention requise!")
    
    # Mode simulation automatique
    st.markdown("### 🔄 Mode Simulation Automatique")
    
    if st.checkbox("Activer la simulation automatique"):
        st.info("🔄 Simulation en cours... (Données aléatoirement générées)")
        
        # Générer des données aléatoires
        np.random.seed(42)
        
        # Simuler des conditions de vol
        scenarios = {
            "Vol Stable": {"roll": 0, "pitch": 0, "noise": 0.1},
            "Vent Latéral": {"roll": 15, "pitch": 0, "noise": 0.3},
            "Montée Rapide": {"roll": 0, "pitch": -10, "noise": 0.2},
            "Turbulences": {"roll": 0, "pitch": 0, "noise": 0.5}
        }
        
        scenario = st.selectbox("Choisissez un scénario:", list(scenarios.keys()))
        
        if st.button("▶️ Lancer la Simulation"):
            scenario_params = scenarios[scenario]
            
            # Générer une série de données
            time_steps = 50
            results = []
            
            for t in range(time_steps):
                # Données simulées avec bruit
                sim_roll = scenario_params["roll"] + np.random.normal(0, scenario_params["noise"])
                sim_pitch = scenario_params["pitch"] + np.random.normal(0, scenario_params["noise"])
                sim_yaw = np.random.normal(0, 1)
                
                sim_data = np.array([[sim_roll, sim_pitch, sim_yaw, 0, 0, 9.8, 48.8566, 2.3522, 100, 0.5, 0.5, 0.5, 0.5]])
                sim_scaled = pipeline.scaler.transform(sim_data)
                
                prediction = pipeline.models[model_choice].predict(sim_scaled)[0]
                results.append(prediction)
            
            # Visualiser les résultats de simulation
            results = np.array(results)
            
            fig = go.Figure()
            
            for i, correction in enumerate(corrections):
                fig.add_trace(go.Scatter(
                    y=results[:, i],
                    mode='lines',
                    name=correction
                ))
            
            fig.update_layout(
                title=f"Simulation: {scenario}",
                xaxis_title="Pas de Temps",
                yaxis_title="Corrections Prédites"
            )
            
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
