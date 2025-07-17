import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import requests
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Analyse Impact Politiques Publiques - Maroc",
    page_icon="🇲🇦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("Politique, Emploi et Chômage")
st.markdown("---")

# Fonctions pour récupérer les données réelles
@st.cache_data(ttl=3600)  # Cache pendant 1 heure
def get_world_bank_data(countries=['MA']):
    """Récupère les données de la Banque Mondiale pour les pays spécifiés"""
    try:
        # Indicateurs clés disponibles et comparables via la Banque Mondiale
        indicators = {
            'inflation': 'FP.CPI.TOTL.ZG',  # Inflation, consumer prices (annual %)
            'unemployment': 'SL.UEM.TOTL.ZS',  # Unemployment, total (% of total labor force)
            'gdp_growth': 'NY.GDP.MKTP.KD.ZG',  # GDP growth (annual %)
            'gdp_per_capita': 'NY.GDP.PCAP.CD',  # GDP per capita (current US$)
            'population': 'SP.POP.TOTL'  # Population, total
        }
        
        all_data_frames = []
        
        for country_code in countries:
            data_frames_country = []
            for indicator_name, indicator_code in indicators.items():
                url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}"
                params = {
                    'format': 'json',
                    'date': '2000:2024',
                    'per_page': 100
                }
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if len(data) > 1 and data[1]:
                        df = pd.DataFrame(data[1])
                        df['indicator'] = indicator_name
                        df['country'] = country_code
                        df['date'] = pd.to_datetime(df['date'], format='%Y')
                        df = df[['date', 'value', 'indicator', 'country']].dropna()
                        data_frames_country.append(df)
            
            if data_frames_country:
                combined_df_country = pd.concat(data_frames_country, ignore_index=True)
                all_data_frames.append(combined_df_country)
        
        if all_data_frames:
            combined_all_countries_df = pd.concat(all_data_frames, ignore_index=True)
            pivot_df = combined_all_countries_df.pivot_table(index=['date', 'country'], columns='indicator', values='value')
            return pivot_df.reset_index()
        else:
            return None
            
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données World Bank: {e}")
        return None

@st.cache_data(ttl=3600)
def get_central_bank_rate():
    """Récupère les taux directeurs historiques réels de Bank Al-Maghrib"""
    try:
        # Données réelles du taux directeur de BAM
        years = list(range(2000, 2025))
        rates = [5, 4.75, 4, 3.25, 3.25, 3.25, 3.25, 3.25, 3.5, 3.25, 3.25, 3, 3, 3, 2.75, 2.5, 2.25, 2.25, 2.25, 2.25, 1.5, 1.5, 2, 3, 2.75]
        
        dates = [datetime(year, 12, 31) for year in years]
        
        rate_df = pd.DataFrame({
            'date': dates,
            'ma_policy_rate': rates
        })
        return rate_df
    except Exception as e:
        st.error(f"Erreur lors de la récupération du taux directeur: {e}")
        return None

@st.cache_data(ttl=3600)
def get_policy_indicators(file_path='politiques.xlsx'):
    """Charge les indicateurs de politiques publiques depuis un fichier Excel."""
    try:
        policies = pd.read_excel(file_path)
        policies.rename(columns={
            policies.columns[0]: 'Années',
            'isFound': 'politique_appliquee',
            'Type de politique': 'type_politique',
            'Nom de la politique': 'nom_politique',
            'Détail': 'detail_politique'
        }, inplace=True)
        policies['date'] = pd.to_datetime(policies['Années'].astype(str), format='%Y')
        return policies

    except FileNotFoundError:
        st.error(f"Erreur: Le fichier '{file_path}' n'a pas été trouvé. Veuillez vous assurer qu'il est dans le même répertoire que l'application.")
        return None

    except Exception as e:
        st.error(f"Erreur lors du chargement des politiques: {e}")
        return None


# Removed hardcoded policy data

def combine_all_data(country_code='MA'):
    """Combine toutes les sources de données pour un pays spécifique"""
    wb_data = get_world_bank_data(countries=[country_code])
    policy_data = get_policy_indicators() 
    central_bank_data = get_central_bank_rate()
    
    if wb_data is not None:
        country_specific_wb_data = wb_data[wb_data['country'] == country_code].drop(columns=['country'])
        combined_data = country_specific_wb_data.copy()
        
        # Ajouter les données du taux directeur
        if central_bank_data is not None:
            combined_data['date_year'] = combined_data['date'].dt.year
            central_bank_data['date_year'] = central_bank_data['date'].dt.year
            combined_data = pd.merge(combined_data, central_bank_data.drop(columns=['date']), on='date_year', how='left')
            combined_data = combined_data.drop(columns=['date_year'])
        
        # Ajouter les indicateurs de politique
        if country_code == 'MA' and policy_data is not None:
            combined_data['date_year'] = combined_data['date'].dt.year
            policy_data['date_year'] = policy_data['date'].dt.year
            combined_data = pd.merge(combined_data, policy_data.drop(columns=['date']), on='date_year', how='left')
            combined_data = combined_data.drop(columns=['date_year'])
            
            # Nettoyer les données de politique
            if 'politique_appliquee' in combined_data.columns:
                combined_data['politique_appliquee'] = combined_data['politique_appliquee'].fillna(0).astype(int)
            
            # Remplacer les valeurs manquantes par des chaînes vides pour les colonnes textuelles
            text_cols = ['type_politique', 'nom_politique', 'detail_politique']
            for col in text_cols:
                if col in combined_data.columns:
                    combined_data[col] = combined_data[col].fillna('')

        combined_data = combined_data.sort_values('date')
        numeric_cols = combined_data.select_dtypes(include=[np.number]).columns
        combined_data[numeric_cols] = combined_data[numeric_cols].interpolate(method='linear')
        
        return combined_data
    else:
        return generate_fallback_data()

def combine_comparison_data(country_codes):
    """Combine les données de la Banque Mondiale pour plusieurs pays pour la comparaison"""
    wb_data = get_world_bank_data(countries=country_codes)
    
    if wb_data is not None:
        combined_data = wb_data.sort_values(['country', 'date'])
        numeric_cols = combined_data.select_dtypes(include=[np.number]).columns
        combined_data[numeric_cols] = combined_data.groupby('country')[numeric_cols].transform(lambda group: group.interpolate(method='linear'))
        return combined_data
    else:
        st.error("❌ Impossible de charger les données pour la comparaison internationale.")
        return None

def generate_fallback_data():
    """Génère des données de secours pour le Maroc si les APIs ne fonctionnent pas"""
    st.warning("⚠️ Utilisation des données de secours pour le Maroc (APIs non disponibles)")
    
    dates = pd.date_range(start='2000-01-01', end='2023-12-31', freq='Y')
    
    # Données approximatives pour le Maroc
    inflation_data = [1.9, 0.6, 2.8, 1.2, 1.5, 1.0, 3.3, 3.9, 3.7, 1.0, 0.9, 0.9, 1.3, 0.4, 1.6, 1.4, 1.6, 0.7, 1.9, 0.2, 1.4, 6.6, 6.7, 5.9]
    unemployment_data = [13.4, 12.5, 11.6, 10.8, 10.8, 9.7, 9.8, 9.6, 9.1, 9.1, 8.9, 9.0, 9.9, 9.7, 9.4, 9.4, 9.2, 9.5, 11.9, 11.8, 12.3, 11.8, 11.5, 11.2]
    gdp_growth_data = [1.6, 7.6, 3.3, 4.8, 4.2, 3.0, 2.7, 5.6, 5.6, 3.8, 5.2, 3.0, 4.4, 2.7, 1.1, 4.1, 3.0, 3.2, -6.3, 7.9, 3.2, 1.3, 3.1, 3.2]
    
    # Taux directeur réel
    policy_rates = [5, 4.75, 4, 3.25, 3.25, 3.25, 3.25, 3.25, 3.5, 3.25, 3.25, 3, 3, 3, 2.75, 2.5, 2.25, 2.25, 2.25, 2.25, 1.5, 1.5, 2, 3]
    
    data = pd.DataFrame({
        'date': dates,
        'inflation': inflation_data,
        'unemployment': unemployment_data,
        'gdp_growth': gdp_growth_data,
        'ma_policy_rate': policy_rates,
    })
    
    # Ajouter les données de politique
    policy_data_fallback = create_policy_data()
    if policy_data_fallback is not None:
        data['date_year'] = data['date'].dt.year
        policy_data_fallback['date_year'] = policy_data_fallback['date'].dt.year
        data = pd.merge(data, policy_data_fallback.drop(columns=['date']), on='date_year', how='left')
        data = data.drop(columns=['date_year'])
        
        # Nettoyer les données de politique
        if 'politique_appliquee' in data.columns:
            data['politique_appliquee'] = data['politique_appliquee'].fillna(0).astype(int)
        
        # Remplacer les valeurs manquantes par des chaînes vides pour les colonnes textuelles
        text_cols = ['type_politique', 'nom_politique', 'detail_politique']
        for col in text_cols:
            if col in data.columns:
                data[col] = data[col].fillna('')

    return data

def train_and_predict_policy_impact(data, target_metric, policy_columns, current_year, forecast_horizon=5):
    """Fonction améliorée pour prédire l'impact des politiques sur 5 ans"""
    
    feature_cols = [col for col in data.columns if col not in ['date', target_metric, 'country', 'type_politique', 'nom_politique', 'detail_politique'] and data[col].dtype in ['float64', 'int64']]
    
    X = data[feature_cols]
    y = data[target_metric]

    combined = pd.concat([X, y], axis=1).dropna()
    X_clean = combined[feature_cols]
    y_clean = combined[target_metric]

    if len(X_clean) < 2:
        st.warning(f"Pas assez de données complètes pour modéliser '{target_metric}'.")
        return None, None

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_clean, y_clean)

    last_known_data_year = data['date'].dt.year.max()
    last_known_data_row = data[data['date'].dt.year == last_known_data_year].iloc[-1]
    
    # Prédictions sur 5 ans
    future_dates = [datetime(last_known_data_year + i + 1, 12, 31) for i in range(forecast_horizon)]
    
    future_df_with_policy = pd.DataFrame({'date': future_dates})
    future_df_without_policy = pd.DataFrame({'date': future_dates})

    for col in feature_cols:
        if col in policy_columns:
            policy_val_for_selected_year = data[(data['date'].dt.year == current_year)][col].iloc[0] if current_year in data['date'].dt.year.unique() and not data[(data['date'].dt.year == current_year)][col].empty else (data[col].iloc[-1] if col in data.columns else 0)
            
            future_df_with_policy[col] = policy_val_for_selected_year
            future_df_without_policy[col] = 0
        else:
            # Pour les variables économiques, ajouter une petite évolution aléatoire
            base_value = last_known_data_row[col] if col in last_known_data_row else 0
            # Ajouter une légère variation pour simuler l'évolution économique
            if col == 'ma_policy_rate':
                # Évolution plus conservatrice du taux directeur
                trend = np.random.normal(0, 0.1, forecast_horizon)
                future_values = [base_value + sum(trend[:i+1]) for i in range(forecast_horizon)]
            else:
                trend = np.random.normal(0, 0.05, forecast_horizon)
                future_values = [base_value + sum(trend[:i+1]) for i in range(forecast_horizon)]
            
            future_df_with_policy[col] = future_values
            future_df_without_policy[col] = future_values

    for f in feature_cols:
        if f not in future_df_with_policy.columns:
            future_df_with_policy[f] = 0
        if f not in future_df_without_policy.columns:
            future_df_without_policy[f] = 0

    predictions_with_policy = model.predict(future_df_with_policy[feature_cols])
    predictions_without_policy = model.predict(future_df_without_policy[feature_cols])

    historical_data_for_plot = data[['date', target_metric]].copy()

    historical_and_future_with = pd.DataFrame({
        'date': historical_data_for_plot['date'].tolist() + future_dates,
        target_metric: historical_data_for_plot[target_metric].tolist() + predictions_with_policy.tolist(),
        'type': ['Historique'] * len(historical_data_for_plot) + ['Prédiction avec politique'] * len(predictions_with_policy)
    })

    historical_and_future_without = pd.DataFrame({
        'date': historical_data_for_plot['date'].tolist() + future_dates,
        target_metric: historical_data_for_plot[target_metric].tolist() + predictions_without_policy.tolist(),
        'type': ['Historique'] * len(historical_data_for_plot) + ['Prédiction sans politique'] * len(predictions_without_policy)
    })

    return historical_and_future_with, historical_and_future_without

def advanced_economic_forecast(data, forecast_horizon=5):
    """Prédiction avancée de tous les indicateurs économiques sur 5 ans"""
    
    forecasts = {}
    key_indicators = ['inflation', 'unemployment', 'gdp_growth', 'ma_policy_rate']
    
    for indicator in key_indicators:
        if indicator in data.columns:
            try:
                # Utiliser une approche de série temporelle simple
                values = data[indicator].dropna()
                if len(values) >= 3:
                    # Calculer la tendance
                    x = np.arange(len(values))
                    z = np.polyfit(x, values, 1)
                    trend = z[0]
                    
                    # Prédictions futures
                    last_value = values.iloc[-1]
                    future_values = []
                    
                    for i in range(1, forecast_horizon + 1):
                        # Ajouter la tendance avec un peu de bruit
                        noise = np.random.normal(0, values.std() * 0.1)
                        future_val = last_value + (trend * i) + noise
                        
                        # Contraintes réalistes
                        if indicator == 'inflation':
                            future_val = max(-2, min(15, future_val))  # Entre -2% et 15%
                        elif indicator == 'unemployment':
                            future_val = max(3, min(25, future_val))   # Entre 3% et 25%
                        elif indicator == 'gdp_growth':
                            future_val = max(-10, min(12, future_val)) # Entre -10% et 12%
                        elif indicator == 'ma_policy_rate':
                            future_val = max(0, min(10, future_val))   # Entre 0% et 10%
                        
                        future_values.append(future_val)
                    
                    forecasts[indicator] = future_values
                else:
                    # Pas assez de données, utiliser la dernière valeur
                    forecasts[indicator] = [values.iloc[-1]] * forecast_horizon
            except:
                forecasts[indicator] = [0] * forecast_horizon
    
    return forecasts

def get_policy_details(data, year):
    """Récupère les détails de la politique appliquée pour une année donnée"""
    policy_info = data[data['date'].dt.year == year]
    if not policy_info.empty and 'politique_appliquee' in policy_info.columns:
        if policy_info['politique_appliquee'].iloc[0] == 1:
            return {
                'type': policy_info['type_politique'].iloc[0] if 'type_politique' in policy_info.columns else '',
                'nom': policy_info['nom_politique'].iloc[0] if 'nom_politique' in policy_info.columns else '',
                'detail': policy_info['detail_politique'].iloc[0] if 'detail_politique' in policy_info.columns else ''
            }
    return None

# Interface utilisateur
st.sidebar.header("🔧 Options d'Analyse")

# Bouton pour actualiser les données
if st.sidebar.button("🔄 Actualiser les données"):
    st.cache_data.clear()
    st.rerun()

# Paramètres de prédiction
st.sidebar.subheader("📊 Paramètres de Prédiction")
forecast_years = st.sidebar.slider("Horizon de prédiction (années)", 1, 10, 5)


# # Option de téléchargement des données
# if data_morocco is not None:
#     st.sidebar.download_button(
#         label="📥 Télécharger les données",
#         data=data_morocco.to_csv(index=False).encode('utf-8'),
#         file_name='donnees_maroc.csv',
#         mime='text/csv'
#     )

# Chargement des données pour le Maroc
with st.spinner("Chargement des données du Maroc en temps réel..."):
    data_morocco = combine_all_data(country_code='MA')

# Country codes for comparison
COUNTRY_CODES = {
    'Maroc': 'MA',
    'Chine': 'CN',
    'USA': 'US',
    'Allemagne': 'DE',
    'Corée du Sud': 'KR',
    'Japon': 'JP',
    'Russie': 'RU',
    'France': 'FR',
    'Brésil': 'BR',
    'Mali': 'ML',
    'Mauritanie': 'MR',
    'Burkina Faso': 'BF',
    'Niger': 'NE',
    'Tchad': 'TD'
    }

if data_morocco is not None:
    st.success("✅ Données chargées avec succès!")
    
    # Sélection du type d'analyse
    analysis_type = st.sidebar.selectbox(
        "Type d'analyse",
        ["Tableau de bord","Analyse des Politiques", "Analyse temporelle", "Modèles et Prévisions", "Comparaison Internationale"]
    )
    
    # Affichage des données selon le type d'analyse sélectionné
    if analysis_type == "Tableau de bord":
        st.header("Vue d'Ensemble - Données Réelles (Maroc)")
        data = data_morocco

        # Métriques en temps réel
        st.subheader("📊 Indicateurs Clés")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'inflation' in data.columns:
                current_inflation = data['inflation'].iloc[-1]
                prev_inflation = data['inflation'].iloc[-2] if len(data) > 1 else current_inflation
                delta_inflation = current_inflation - prev_inflation
                st.metric("Inflation", f"{current_inflation:.1f}%", delta=f"{delta_inflation:+.1f}%")
        
        with col2:
            if 'unemployment' in data.columns:
                current_unemployment = data['unemployment'].iloc[-1]
                prev_unemployment = data['unemployment'].iloc[-2] if len(data) > 1 else current_unemployment
                delta_unemployment = current_unemployment - prev_unemployment
                st.metric("Chômage", f"{current_unemployment:.1f}%", delta=f"{delta_unemployment:+.1f}%")
        
        with col3:
            if 'gdp_growth' in data.columns:
                current_gdp = data['gdp_growth'].iloc[-1]
                prev_gdp = data['gdp_growth'].iloc[-2] if len(data) > 1 else current_gdp
                delta_gdp = current_gdp - prev_gdp
                st.metric("Croissance PIB", f"{current_gdp:.1f}%", delta=f"{delta_gdp:+.1f}%")
        
        with col4:
            if 'ma_policy_rate' in data.columns:
                current_rate = data['ma_policy_rate'].iloc[-1]
                prev_rate = data['ma_policy_rate'].iloc[-2] if len(data) > 1 else current_rate
                delta_rate = current_rate - prev_rate
                st.metric("Taux Directeur", f"{current_rate:.2f}%", delta=f"{delta_rate:+.2f}%")
        
        # Graphiques des tendances
        st.subheader("📈 Évolution des Indicateurs Clés")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Inflation (%)', 'Chômage (%)', 'Croissance PIB (%)', 'Taux Directeur BAM (%)'),
            vertical_spacing=0.1
        )
        
        if 'inflation' in data.columns:
            fig.add_trace(go.Scatter(x=data['date'], y=data['inflation'], 
                                   name='Inflation', line=dict(color='red')), row=1, col=1)
        
        if 'unemployment' in data.columns:
            fig.add_trace(go.Scatter(x=data['date'], y=data['unemployment'], 
                                   name='Chômage', line=dict(color='orange')), row=1, col=2)
        
        if 'gdp_growth' in data.columns:
            fig.add_trace(go.Scatter(x=data['date'], y=data['gdp_growth'], 
                                   name='PIB', line=dict(color='green')), row=2, col=1)
        
        if 'ma_policy_rate' in data.columns: 
            fig.add_trace(go.Scatter(x=data['date'], y=data['ma_policy_rate'], 
                                   name='Taux Directeur', line=dict(color='blue')), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Tableau des données récentes
        st.subheader("📋 Données Récentes")
        display_cols = ['date', 'inflation', 'unemployment', 'gdp_growth', 'ma_policy_rate']
        recent_data = data[display_cols].tail(5)
        st.dataframe(recent_data)
    

    # elif analysis_type == "Analyse des Politiques":
    #     pass


#################changement###########################3
    elif analysis_type == "Analyse des Politiques":
        st.header("🧩 Analyse de l'impact des politiques publiques")

        # 🔍 Lecture du fichier Excel
        try:
            policies = pd.read_excel("politiques.xlsx")
            policies.rename(columns={
                policies.columns[0]: 'Années',
                'isFound': 'politique_appliquee',
                'Type de politique': 'type_politique',
                'Nom de la politique': 'nom_politique',
                'Détail': 'detail_politique'
            }, inplace=True)
            policies['date'] = pd.to_datetime(policies['Années'].astype(str), format='%Y')
        except Exception as e:
            st.error(f"Erreur lors du chargement de politiques.xlsx : {e}")
            st.stop()

        # 🎯 Sélection du type de politique
        types_disponibles = policies['type_politique'].dropna().unique()
        selected_type = st.selectbox("Type de politique", types_disponibles)

        politiques_filtrees = policies[
            (policies['politique_appliquee'] == 1) &
            (policies['type_politique'] == selected_type)
        ]

        if politiques_filtrees.empty:
            st.warning("Aucune politique trouvée pour ce type.")
        else:
            annees_dispo = politiques_filtrees['date'].dt.year.unique()
            selected_year = st.selectbox("Année de la politique", sorted(annees_dispo))

            # 📌 Détails
            politique = politiques_filtrees[politiques_filtrees['date'].dt.year == selected_year].iloc[0]
            st.subheader(f"📌 Politique : {politique['nom_politique']} ({selected_year})")
            st.markdown(f"**Détail** : {politique['detail_politique']}")

            # 🔄 Association type → indicateur
            indicateur_map = {
                "Monétaire": "inflation",
                "Sociale": "unemployment",
                "Industrielle": "gdp_growth",
                "Entrepreneuriat": "unemployment"
            }
            indicateur = indicateur_map.get(politique['type_politique'], 'gdp_growth')

            # 📊 Données avant / après
            data = data_morocco.copy()
            data['date'] = pd.to_datetime(data['date'])
            avant = data[(data['date'].dt.year >= selected_year - 2) & (data['date'].dt.year < selected_year)]
            apres = data[(data['date'].dt.year >= selected_year) & (data['date'].dt.year <= selected_year + 2)]

            # 📈 Graphe Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=avant['date'], y=avant[indicateur],
                mode='lines+markers', name='Avant',
                line=dict(color='red')
            ))
            fig.add_trace(go.Scatter(
                x=apres['date'], y=apres[indicateur],
                mode='lines+markers', name='Après',
                line=dict(color='green')
            ))

            # Convertir les dates en timestamp numérique
            date_politique = pd.Timestamp(year=selected_year, month=1, day=1)
            
            # Calculer les limites Y pour la ligne verticale
            all_data = pd.concat([avant, apres])
            y_min = all_data[indicateur].min()
            y_max = all_data[indicateur].max()
            
            # Ajouter la ligne verticale avec add_shape
            fig.add_shape(
                type="line",
                x0=date_politique,
                x1=date_politique,
                y0=y_min,
                y1=y_max,
                line=dict(color="black", width=2, dash="dash")
            )
            
            # Ajouter l'annotation
            fig.add_annotation(
                x=date_politique,
                y=y_max * 0.9,  # Position à 90% de la hauteur
                text="Début de la politique",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="black",
                font=dict(size=12),
                bordercolor="black",
                borderwidth=1
            )

            fig.update_layout(
                title=f"Évolution de {indicateur} autour de la politique {politique['nom_politique']}",
                xaxis_title="Année",
                yaxis_title=f"{indicateur} (%)",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)


#####################end############################
   

    elif analysis_type == "Analyse temporelle":
        st.header("📈 Analyse Temporelle Détaillée")
        data = data_morocco
        
        # Sélecteur de période
        col1, col2 = st.columns(2)
        with col1:
            start_year = st.selectbox("Année de début", 
                                    options=sorted(data['date'].dt.year.unique()),
                                    index=0)
        with col2:
            end_year = st.selectbox("Année de fin", 
                                  options=sorted(data['date'].dt.year.unique()),
                                  index=len(data['date'].dt.year.unique())-1)
        
        # Filtrer les données
        filtered_data = data[(data['date'].dt.year >= start_year) & (data['date'].dt.year <= end_year)]
        
        # Graphique temporel multi-indicateurs
        st.subheader("📊 Évolution des Indicateurs Clés")
        
        indicators = st.multiselect(
            "Sélectionnez les indicateurs à afficher",
            ['inflation', 'unemployment', 'gdp_growth', 'ma_policy_rate'],
            default=['inflation', 'unemployment', 'gdp_growth'],
            format_func=lambda x: {
                'inflation': 'Inflation (%)',
                'unemployment': 'Chômage (%)',
                'gdp_growth': 'Croissance PIB (%)',
                'ma_policy_rate': 'Taux Directeur BAM (%)'
            }[x]
        )
        
        if indicators:
            fig = go.Figure()
            
            colors = ['blue', 'red', 'green', 'orange']
            for i, indicator in enumerate(indicators):
                if indicator in filtered_data.columns:
                    fig.add_trace(go.Scatter(
                        x=filtered_data['date'],
                        y=filtered_data[indicator],
                        mode='lines+markers',
                        name=indicator,
                        line=dict(color=colors[i % len(colors)])
                    ))
            
            fig.update_layout(
                title=f"Évolution des indicateurs ({start_year}-{end_year})",
                xaxis_title="Année",
                yaxis_title="Valeur (%)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistiques de la période
        st.subheader("📊 Statistiques de la Période")
        
        stats_data = {}
        for indicator in ['inflation', 'unemployment', 'gdp_growth', 'ma_policy_rate']:
            if indicator in filtered_data.columns:
                stats_data[indicator] = {
                    'Moyenne': filtered_data[indicator].mean(),
                    'Médiane': filtered_data[indicator].median(),
                    'Écart-type': filtered_data[indicator].std(),
                    'Min': filtered_data[indicator].min(),
                    'Max': filtered_data[indicator].max()
                }
        
        stats_df = pd.DataFrame(stats_data).T
        st.dataframe(stats_df.round(2))
    
    elif analysis_type == "Modèles et Prévisions":
        st.header("🔮 Modèles Prédictifs")
        data = data_morocco
        
        # Sélecteur d'indicateur et d'année
        col1, col2 = st.columns(2)
        
        with col1:
            target_metric = st.selectbox(
                "Indicateur à prédire",
                ['inflation', 'unemployment', 'gdp_growth'],
                format_func=lambda x: {
                    'inflation': 'Inflation (%)',
                    'unemployment': 'Chômage (%)',
                    'gdp_growth': 'Croissance PIB (%)'
                }[x]
            )
        
        with col2:
            policy_years = data[data['politique_appliquee'] == 1]['date'].dt.year.tolist()
            if policy_years:
                selected_year = st.selectbox(
                    "Année de la politique à simuler",
                    policy_years
                )
            else:
                selected_year = 2023
        
        # Prédiction d'impact
        if target_metric in data.columns:
            policy_columns = ['politique_appliquee', 'ma_policy_rate']
            
            with st.spinner("Calcul des prédictions..."):
                predictions_with, predictions_without = train_and_predict_policy_impact(
                    data, target_metric, policy_columns, selected_year, forecast_horizon=forecast_years
                )
            
            if predictions_with is not None and predictions_without is not None:
                fig = go.Figure()
                
                # Données historiques
                historical_data = predictions_with[predictions_with['type'] == 'Historique']
                fig.add_trace(go.Scatter(
                    x=historical_data['date'],
                    y=historical_data[target_metric],
                    mode='lines',
                    name='Données historiques',
                    line=dict(color='blue', width=2)
                ))
                
                # Prédictions avec politique
                with_policy = predictions_with[predictions_with['type'] == 'Prédiction avec politique']
                fig.add_trace(go.Scatter(
                    x=with_policy['date'],
                    y=with_policy[target_metric],
                    mode='lines+markers',
                    name='Avec politique',
                    line=dict(color='green', width=2, dash='dash')
                ))
                
                # Prédictions sans politique
                without_policy = predictions_without[predictions_without['type'] == 'Prédiction sans politique']
                fig.add_trace(go.Scatter(
                    x=without_policy['date'],
                    y=without_policy[target_metric],
                    mode='lines+markers',
                    name='Sans politique',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title=f"Prédiction d'impact - {target_metric} (horizon {forecast_years} ans)",
                    xaxis_title="Année",
                    yaxis_title=f"{target_metric} (%)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Analyse quantitative
                st.subheader("📊 Analyse Quantitative")
                
                if len(with_policy) > 0 and len(without_policy) > 0:
                    avg_with = with_policy[target_metric].mean()
                    avg_without = without_policy[target_metric].mean()
                    impact = avg_with - avg_without
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Moyenne avec politique", f"{avg_with:.2f}%")
                    with col2:
                        st.metric("Moyenne sans politique", f"{avg_without:.2f}%")
                    with col3:
                        st.metric("Impact estimé", f"{impact:+.2f}%")
    
    # fusionné dans Modèles et Prévisions
        st.header("🚀 Prédictions Économiques Futures")
        data = data_morocco
        
        # Prédictions avancées
        with st.spinner("Calcul des prédictions économiques..."):
            forecasts = advanced_economic_forecast(data, forecast_horizon=forecast_years)
        
        # Année de départ pour les prédictions
        current_year = data['date'].dt.year.max()
        future_years = [current_year + i + 1 for i in range(forecast_years)]
        
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Inflation (%)', 'Chômage (%)', 'Croissance PIB (%)', 'Taux Directeur BAM (%)'),
            vertical_spacing=0.1
        )
        
        # Données historiques et prédictions
        for i, (indicator, values) in enumerate(forecasts.items()):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            if indicator in data.columns:
                # Données historiques
                fig.add_trace(go.Scatter(
                    x=data['date'],
                    y=data[indicator],
                    mode='lines',
                    name=f'{indicator} (historique)',
                    line=dict(color='blue'),
                    showlegend=False
                ), row=row, col=col)
                
                # Prédictions
                fig.add_trace(go.Scatter(
                    x=[datetime(year, 12, 31) for year in future_years],
                    y=values,
                    mode='lines+markers',
                    name=f'{indicator} (prédiction)',
                    line=dict(color='red', dash='dash'),
                    showlegend=False
                ), row=row, col=col)
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Tableau des prédictions
        st.subheader("📊 Tableau des Prédictions")
        
        predictions_df = pd.DataFrame(forecasts)
        predictions_df.index = future_years
        predictions_df.index.name = 'Année'
        
        # Arrondir les valeurs
        predictions_df = predictions_df.round(2)
        
        st.dataframe(predictions_df)
        
        # Analyse des tendances
        st.subheader("📈 Analyse des Tendances")
        
        for indicator, values in forecasts.items():
            if len(values) > 1:
                trend = "croissante" if values[-1] > values[0] else "décroissante"
                change = abs(values[-1] - values[0])
                
                st.write(f"**{indicator}**: Tendance {trend} avec une variation de {change:.2f} points sur {forecast_years} ans")
    
    # elif analysis_type == "Tableau de bord":
        st.header("🎛️ Tableau de Bord Économique")
        data = data_morocco
        
        # Métriques en temps réel
        st.subheader("📊 Indicateurs Clés")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'inflation' in data.columns:
                current_inflation = data['inflation'].iloc[-1]
                prev_inflation = data['inflation'].iloc[-2] if len(data) > 1 else current_inflation
                delta_inflation = current_inflation - prev_inflation
                st.metric("Inflation", f"{current_inflation:.1f}%", delta=f"{delta_inflation:+.1f}%")
        
        with col2:
            if 'unemployment' in data.columns:
                current_unemployment = data['unemployment'].iloc[-1]
                prev_unemployment = data['unemployment'].iloc[-2] if len(data) > 1 else current_unemployment
                delta_unemployment = current_unemployment - prev_unemployment
                st.metric("Chômage", f"{current_unemployment:.1f}%", delta=f"{delta_unemployment:+.1f}%")
        
        with col3:
            if 'gdp_growth' in data.columns:
                current_gdp = data['gdp_growth'].iloc[-1]
                prev_gdp = data['gdp_growth'].iloc[-2] if len(data) > 1 else current_gdp
                delta_gdp = current_gdp - prev_gdp
                st.metric("Croissance PIB", f"{current_gdp:.1f}%", delta=f"{delta_gdp:+.1f}%")
        
        with col4:
            if 'ma_policy_rate' in data.columns:
                current_rate = data['ma_policy_rate'].iloc[-1]
                prev_rate = data['ma_policy_rate'].iloc[-2] if len(data) > 1 else current_rate
                delta_rate = current_rate - prev_rate
                st.metric("Taux Directeur", f"{current_rate:.2f}%", delta=f"{delta_rate:+.2f}%")
        
        # Graphiques de tendances
        st.subheader("📈 Tendances Récentes (5 dernières années)")
        
        recent_data = data.tail(5)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Graphique radar
            if all(col in recent_data.columns for col in ['inflation', 'unemployment', 'gdp_growth', 'ma_policy_rate']):
                categories = ['Inflation', 'Chômage', 'Croissance PIB', 'Taux Directeur']
                values = [
                    recent_data['inflation'].iloc[-1],
                    recent_data['unemployment'].iloc[-1],
                    recent_data['gdp_growth'].iloc[-1],
                    recent_data['ma_policy_rate'].iloc[-1]
                ]
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='Indicateurs actuels'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, max(values) * 1.2]
                        )),
                    showlegend=True,
                    title="Profil Économique Actuel"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Graphique en barres des variations
            if len(recent_data) > 1:
                variations = {}
                for col in ['inflation', 'unemployment', 'gdp_growth', 'ma_policy_rate']:
                    if col in recent_data.columns:
                        variations[col] = recent_data[col].iloc[-1] - recent_data[col].iloc[0]
                
                fig = px.bar(
                    x=list(variations.keys()),
                    y=list(variations.values()),
                    title="Variations sur 5 ans",
                    color=list(variations.values()),
                    color_continuous_scale='RdYlBu_r'
                )
                
                fig.update_layout(
                    xaxis_title="Indicateurs",
                    yaxis_title="Variation (points)",
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Alertes et recommandations
        st.subheader("🚨 Alertes et Recommandations")
        
        alerts = []
        
        if 'inflation' in data.columns:
            current_inflation = data['inflation'].iloc[-1]
            if current_inflation > 5:
                alerts.append("⚠️ **Inflation élevée** - Risque de perte de pouvoir d'achat")
            elif current_inflation < 0:
                alerts.append("⚠️ **Déflation** - Risque de spirale déflationniste")
        
        if 'unemployment' in data.columns:
            current_unemployment = data['unemployment'].iloc[-1]
            if current_unemployment > 12:
                alerts.append("🔴 **Chômage élevé** - Nécessité de politiques d'emploi")
        
        if 'gdp_growth' in data.columns:
            current_gdp = data['gdp_growth'].iloc[-1]
            if current_gdp < 0:
                alerts.append("🔴 **Récession** - Croissance négative détectée")
            elif current_gdp < 2:
                alerts.append("⚠️ **Croissance faible** - Besoin de stimulation économique")
        
        if alerts:
            for alert in alerts:
                st.warning(alert)
        else:
            st.success("✅ Tous les indicateurs sont dans des fourchettes acceptables")
    
    elif analysis_type == "Comparaison Internationale":
        st.header("🌍 Comparaison Internationale")
        
        # Sélection des pays
        selected_countries = st.multiselect(
            "Sélectionnez les pays à comparer avec le Maroc",
            [country for country in COUNTRY_CODES.keys() if country != 'Maroc'],
            default=['Mali', 'Mauritanie', 'Burkina Faso', 'Niger', 'Tchad']
        )
        
        if selected_countries:
            # Ajouter le Maroc à la liste
            comparison_countries = ['Maroc'] + selected_countries
            country_codes = [COUNTRY_CODES[country] for country in comparison_countries]
            
            # Charger les données de comparaison
            with st.spinner("Chargement des données internationales..."):
                comparison_data = combine_comparison_data(country_codes)
            
            if comparison_data is not None:
                # Sélecteur d'indicateur
                indicator_for_comparison = st.selectbox(
                    "Indicateur à comparer",
                    ['inflation', 'unemployment', 'gdp_growth'],
                    format_func=lambda x: {
                        'inflation': 'Inflation (%)',
                        'unemployment': 'Chômage (%)',
                        'gdp_growth': 'Croissance PIB (%)'
                    }[x]
                )
                
                if indicator_for_comparison in comparison_data.columns:
                    # Graphique de comparaison
                    fig = go.Figure()
                    
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                    
                    for i, country_code in enumerate(country_codes):
                        country_data = comparison_data[comparison_data['country'] == country_code]
                        country_name = [name for name, code in COUNTRY_CODES.items() if code == country_code][0]
                        
                        if not country_data.empty:
                            fig.add_trace(go.Scatter(
                                x=country_data['date'],
                                y=country_data[indicator_for_comparison],
                                mode='lines+markers',
                                name=country_name,
                                line=dict(color=colors[i % len(colors)], width=2)
                            ))
                    
                    fig.update_layout(
                        title=f"Comparaison internationale - {indicator_for_comparison}",
                        xaxis_title="Année",
                        yaxis_title=f"{indicator_for_comparison} (%)",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
