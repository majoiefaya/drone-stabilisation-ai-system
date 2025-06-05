"""
Script de test rapide pour vérifier la visualisation 3D
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go

def test_visualization():
    """Test rapide de la visualisation 3D"""
    try:
        # Test simple de Scatter3d avec des paramètres valides
        fig = go.Figure()
        
        # Test 1: Marqueurs simples
        fig.add_trace(go.Scatter3d(
            x=[0, 1, 2],
            y=[0, 1, 2], 
            z=[0, 1, 2],
            mode='lines+markers',
            marker=dict(size=8, color='blue', symbol='circle'),
            name='Test Simple'
        ))
        
        # Test 2: Marqueurs avec couleurs
        fig.add_trace(go.Scatter3d(
            x=[0, 1, 2],
            y=[2, 1, 0], 
            z=[1, 2, 0],
            mode='markers',
            marker=dict(size=10, color='red', symbol='diamond'),
            name='Test Avancé'
        ))
        
        fig.update_layout(
            title="Test Visualisation 3D",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y", 
                zaxis_title="Z"
            )
        )
        
        return fig, "✅ Test réussi - Aucune erreur de visualisation"
        
    except Exception as e:
        return None, f"❌ Erreur détectée: {str(e)}"

if __name__ == "__main__":
    fig, message = test_visualization()
    print(message)
    if fig:
        print("🎉 La visualisation 3D devrait maintenant fonctionner correctement!")
