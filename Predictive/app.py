from textwrap import dedent

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# 1. CONFIGURATION & STYLE
# ==========================================
# Configuration de base de la page (Titre de l'onglet, icône, mise en page large)
st.set_page_config(page_title="Airbnb Price Predictor", page_icon="🏡", layout="wide")

# Injection de CSS personnalisé (HTML)
# C'est ici qu'on définit le look "Airbnb" (couleurs rouge/gris, cartes ombrées, polices).
# Streamlit permet d'injecter du CSS via st.markdown avec unsafe_allow_html=True.
st.markdown("""
<style>
    :root { --airbnb-red: #FF5A5F; --airbnb-dark: #484848; --bg-light: #f7f7f7; }
    [data-testid="stAppViewContainer"] { background-color: var(--bg-light) !important; }
    h1 { color: var(--airbnb-red) !important; text-align: center; font-weight: 800 !important; }
    .result-card { background: white; padding: 30px; border-radius: 18px; box-shadow: 0 10px 25px rgba(0,0,0,0.08); border: 1px solid #e8e8e8; }
    .big-price { font-size: 55px !important; font-weight: 900 !important; color: var(--airbnb-red) !important; margin: 0; }
    .subtitle { font-size: 17px; color: #6f6f6f !important; margin: 0; }
    .label-header { font-weight: bold; color: var(--airbnb-dark); margin-bottom: 5px; }
</style>
""", unsafe_allow_html=True)


# Le décorateur @st.cache_resource est CRITIQUE.
# Il charge le modèle une seule fois au lancement de l'app et le garde en mémoire RAM.
# Sans ça, le modèle serait rechargé depuis le disque à chaque fois qu'un utilisateur change une option (très lent).
@st.cache_resource
def load_model():
    try:
        # Charge le pipeline complet sauvegardé (Prétraitement + XGBoost/RandomForest)
        return joblib.load("airbnb_model_prod.pkl")
    except:
        return None


pipeline = load_model()

# ==========================================
# 2. INTERFACE (FORMULAIRE À GAUCHE)
# ==========================================
st.title("🏡 Estimation du prix en temps réel")
st.markdown("---")

# Création de deux colonnes : une grande (2/3) pour les inputs, une petite (1/3) pour le résultat
left, right = st.columns([2, 1])

with left:
    st.subheader("⚙️ Caractéristiques du logement")

    # Sous-colonnes pour organiser les champs proprement
    c1, c2 = st.columns(2)
    with c1:
        # Sélecteurs (Dropdowns)
        city = st.selectbox("📍 Ville", ["NYC", "LA", "SF", "DC", "Chicago", "Boston"])
        prop_type = st.selectbox("🏠 Type de bien", ["Apartment", "House", "Condominium", "Loft", "Townhouse"])
        cleaning_fee = st.selectbox("🧹 Frais de ménage", ["True", "False"])

    with c2:
        # Mapping pour traduire l'affichage (Français) en valeur comprise par le modèle (Anglais)
        room_map = {"Logement entier": "Entire home/apt", "Chambre privée": "Private room", "Partagé": "Shared room"}
        # L'utilisateur voit les clés (FR), on récupère les valeurs (EN)
        room_type = room_map[st.selectbox("🔑 Type d'espace", list(room_map.keys()))]
        cancel_policy = st.selectbox("📝 Politique d'annulation", ["flexible", "moderate", "strict"])
        instant_book = st.selectbox("⚡ Réservation instantanée", ["True", "False"])

    st.write("")
    # Slider pour les valeurs numériques continues
    accommodates = st.slider("👥 Capacité d'accueil (Voyageurs)", 1, 16, 2)

    cc1, cc2 = st.columns(2)
    with cc1: bedrooms = st.number_input("🛏️ Nombre de chambres", 0, 10, 1)
    with cc2: bathrooms = st.number_input("🚿 Nombre de salles de bain", 0, 10, 1)

    st.markdown("### ✨ Équipements disponibles")
    # Checkbox pour les variables binaires (Oui/Non)
    eq1, eq2, eq3 = st.columns(3)
    with eq1:
        has_wifi = st.checkbox("📡 Wifi", True)  # Pré-coché par défaut
        has_ac = st.checkbox("❄️ Climatisation")
    with eq2:
        has_kitchen = st.checkbox("🍳 Cuisine")
        has_parking = st.checkbox("🚗 Parking")
    with eq3:
        has_tub = st.checkbox("🛁 Jacuzzi")
        has_view = st.checkbox("🌆 Vue exceptionnelle")

# ==========================================
# 3. LOGIQUE DE CALCUL AUTOMATIQUE
# ==========================================
# Streamlit relance tout le script à chaque interaction.
# Dès qu'on change un slider à gauche, ce code s'exécute et met à jour la colonne de droite.
with right:
    st.subheader("💰 Estimation finale")

    if pipeline:
        # 1. Préparation des coordonnées
        # Le modèle a appris avec latitude/longitude, mais l'utilisateur choisit une ville.
        # On injecte donc les coordonnées du centre-ville correspondant.
        coords = {'NYC': [40.71, -74.00], 'LA': [34.05, -118.24], 'SF': [37.77, -122.41],
                  'DC': [38.90, -77.03], 'Chicago': [41.87, -87.62], 'Boston': [42.36, -71.05]}

        # 2. Création du DataFrame d'entrée
        # On doit reconstruire EXACTEMENT le même format de données que celui utilisé lors de l'entraînement (X_train).
        # Note : Pour les champs qu'on ne demande pas à l'utilisateur (ex: review_scores_rating),
        # on met des valeurs par défaut (médianes ou moyennes) pour ne pas bloquer le modèle.
        input_df = pd.DataFrame([{
            'accommodates': accommodates, 'bathrooms': bathrooms, 'bedrooms': bedrooms, 'beds': bedrooms,
            'latitude': coords[city][0], 'longitude': coords[city][1],
            'number_of_reviews': 25, 'review_scores_rating': 95,  # Valeurs par défaut
            # On somme les équipements pour créer une feature 'amenities_count'
            'amenities_count': 10 + has_wifi + has_ac + has_kitchen + has_parking + has_tub + has_view,
            'description_len': 500, 'host_days_active': 1000, 'days_since_review': 30,
            'has_wifi': int(has_wifi), 'has_air_conditioning': int(has_ac), 'has_pool': 0,
            'has_kitchen': int(has_kitchen), 'has_free_parking': int(has_parking), 'has_gym': 0,
            'has_hot_tub': int(has_tub), 'has_view': int(has_view),
            'property_type': prop_type, 'room_type': room_type, 'cancellation_policy': cancel_policy,
            'city': city, 'cleaning_fee': cleaning_fee, 'instant_bookable': instant_book
        }])

        try:
            # 3. Prédiction
            # Le pipeline gère le OneHotEncoding et l'Imputation automatiquement grâce au préprocesseur intégré.
            log_price = pipeline.predict(input_df)[0]

            # 4. Conversion Log -> Prix réel
            # Comme le modèle a prédit un log_price, on doit utiliser l'exponentielle (np.exp) pour revenir en dollars.
            price = np.exp(log_price)

            # 5. Affichage dynamique du résultat via HTML
            # On crée une belle "Card" HTML pour afficher le prix en gros.
            st.markdown(dedent(f"""
                <div class="result-card">
                    <p class="subtitle">Prix suggéré :</p>
                    <p class="big-price">{price:.0f} $</p>
                    <p class="subtitle">par nuit</p>
                    <hr style="margin: 20px 0; border: none; border-top: 1px solid #eee;">
                    <p style="font-weight:700;color:#484848;margin-bottom:5px;">Marché local</p>
                    <p style="color:#008489;font-size:20px;font-weight:700;margin:0;">
                        Fourchette : {price * 0.92:.0f}$ – {price * 1.08:.0f}$
                    </p>
                    <p style="font-size:12px; color:#999; margin-top:10px;">
                        Basé sur les tendances actuelles de la ville de {city}.
                    </p>
                </div>
            """), unsafe_allow_html=True)

            st.info("💡 Modifiez n'importe quel champ à gauche pour voir l'impact immédiat sur le prix.")

        except Exception as e:
            st.error(f"Erreur de prédiction : {e}")
    else:
        # Message d'erreur si le fichier .pkl n'est pas trouvé
        st.warning("⚠️ Modèle non chargé. Vérifiez que 'airbnb_model_prod.pkl' est présent.")