import streamlit as st  # La librairie magique pour créer l'app web
import pandas as pd
import numpy as np
import re  # Pour nettoyer les noms de colonnes (XGBoost/LightGBM n'aiment pas les caractères bizarres)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Les 3 Mousquetaires (Les modèles concurrents)
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ==========================================
# 1. CONFIGURATION DE LA PAGE
# ==========================================
# Définit le titre de l'onglet du navigateur et utilise toute la largeur de l'écran
st.set_page_config(page_title="Airbnb IA - Battle Royale", layout="wide")

st.title("🥊 Airbnb Battle Royale : RF vs XGBoost vs LightGBM")
st.markdown("---")  # Une ligne de séparation horizontale

# ==========================================
# 2. PARAMÈTRES (BARRE LATÉRALE / SIDEBAR)
# ==========================================
# Tout ce qui commence par st.sidebar se met dans le volet gauche.

st.sidebar.header("1. Données")
# Champ texte pour dire où est le fichier
csv_path = st.sidebar.text_input("Chemin CSV", "data.csv")
# Case à cocher pour activer/désactiver le nettoyage des prix extrêmes
remove_outliers = st.sidebar.checkbox("Filtrer Outliers (Prix extrêmes)", value=True)

st.sidebar.header("2. Paramètres IA")
# Slider pour choisir la taille du test (5% à 50%)
test_size = st.sidebar.slider("Test Set Size", 0.05, 0.5, 0.2)
random_state = st.sidebar.number_input("Seed", 0, 9999, 42)  # Pour figer le hasard

st.sidebar.markdown("---")
# --- Configuration du Random Forest ---
st.sidebar.subheader("🌲 RandomForest")
rf_est = st.sidebar.slider("RF - Arbres", 50, 500, 250, 10)  # Combien d'arbres ?
rf_depth = st.sidebar.slider("RF - Depth Max", 5, 50, 22)  # Profondeur max

st.sidebar.markdown("---")
# --- Configuration de XGBoost ---
st.sidebar.subheader("🚀 XGBoost")
xgb_est = st.sidebar.slider("XGB - Estimators", 100, 2000, 1000, 100)
xgb_depth = st.sidebar.slider("XGB - Max Depth", 3, 10, 6)  # XGBoost préfère des arbres peu profonds
xgb_lr = st.sidebar.slider("XGB - Learning Rate", 0.01, 0.3, 0.05)  # Vitesse d'apprentissage

st.sidebar.markdown("---")
# --- Configuration de LightGBM ---
# LightGBM est souvent plus rapide que XGBoost
st.sidebar.subheader("⚡ LightGBM")
lgbm_est = st.sidebar.slider("LGBM - Estimators", 100, 2000, 1000, 100)
lgbm_depth = st.sidebar.slider("LGBM - Max Depth", -1, 15, -1)  # -1 veut dire "illimité"
lgbm_lr = st.sidebar.slider("LGBM - Learning Rate", 0.01, 0.3, 0.05)

st.sidebar.markdown("---")
# Combien d'exemples on veut voir dans le tableau comparatif à la fin
nb_samples = st.sidebar.slider("Exemples pour le tableau final", 1, 50, 10)

# Le bouton qui déclenche tout le processus
do_train = st.sidebar.button("🚀 LANCER LA BATAILLE")


# ==========================================
# 3. PRÉPARATION DES DONNÉES (CACHE)
# ==========================================
# @st.cache_data est important pour optimiser Streamlit.
# Si les paramètres d'entrée (ici le chemin CSV) ne changent pas,
# Streamlit va réutiliser les données déjà chargées au lieu de relire le CSV à chaque clic.
@st.cache_data
def load_and_prep_data(path):
    # --- Chargement du CSV ---
    df = pd.read_csv(path)

    # --- Nettoyage basique ---
    # Remplissage des valeurs manquantes dans 'amenities' et 'description' par des chaînes vides
    # Cela évite les erreurs lors des transformations suivantes
    df['amenities'] = df['amenities'].fillna("")
    df['description'] = df['description'].fillna("")

    # --- Feature Engineering ---
    # Création de nouvelles colonnes à partir des données existantes pour enrichir le dataset

    # Nombre d'équipements listés dans 'amenities'
    df['amenities_count'] = df['amenities'].apply(lambda x: len(str(x).split(',')) if x else 0)

    # Longueur de la description (nombre de caractères)
    df['description_len'] = df['description'].apply(lambda x: len(str(x)) if x else 0)

    # Nettoyage du texte des équipements pour enlever caractères spéciaux
    df['amenities_clean'] = df['amenities'].str.replace('[{}"/]', '', regex=True)

    # --- One-Hot Encoding manuel pour équipements "premium" ---
    # On crée une colonne binaire pour chaque équipement considéré important
    premium = ['Wifi', 'Air conditioning', 'Pool', 'Kitchen', 'Free parking', 'Gym', 'Hot tub', 'View']
    new_cols = []  # liste des nouvelles colonnes créées
    for item in premium:
        col_name = f'has_{item.replace(" ", "_").lower()}'  # nom de la colonne propre
        # 1 si l'équipement est présent, 0 sinon
        df[col_name] = df['amenities_clean'].str.contains(item, case=False, regex=False).astype(int)
        new_cols.append(col_name)

    # --- Gestion des dates ---
    # On convertit les colonnes dates en objets datetime
    now = pd.to_datetime('2017-10-01')  # référence pour calcul des durées
    for col in ['host_since', 'last_review']:
        df[col] = pd.to_datetime(df[col], errors='coerce')  # 'coerce' remplace les erreurs par NaN

    # Création de nouvelles variables numériques basées sur les dates
    df['host_days_active'] = (now - df['host_since']).dt.days  # nombre de jours depuis l'inscription de l'hôte
    df['days_since_review'] = (now - df['last_review']).dt.days  # nombre de jours depuis le dernier commentaire

    # --- Remplissage des valeurs manquantes ---
    # Pour les colonnes numériques : on remplace par la médiane
    num_fills = ['host_days_active', 'days_since_review', 'bathrooms', 'bedrooms', 'beds', 'review_scores_rating']
    for col in num_fills:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Pour les colonnes catégorielles : on remplace par "Unknown" et on force le type string
    cat_feats = ['property_type', 'room_type', 'cancellation_policy', 'city', 'cleaning_fee', 'instant_bookable']
    for col in cat_feats:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

    # --- Filtrage des Outliers (optionnel selon la checkbox) ---
    target = 'log_price'
    if remove_outliers:
        # On élimine les 1% plus bas et 1% plus hauts du prix
        low, high = df[target].quantile(0.01), df[target].quantile(0.99)
        df = df[(df[target] >= low) & (df[target] <= high)]

    # --- Sélection finale des colonnes utiles ---
    # Colonnes numériques de base
    base_nums = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 'latitude', 'longitude',
                 'number_of_reviews', 'review_scores_rating', 'amenities_count',
                 'description_len', 'host_days_active', 'days_since_review']

    # On garde les colonnes numériques + colonnes premium + colonnes catégorielles
    final_cols = [c for c in (base_nums + new_cols + cat_feats) if c in df.columns]

    # Retourne le DataFrame préparé, les colonnes finales, les colonnes catégorielles et la cible
    return df, final_cols, cat_feats, target


# ==========================================
# 4. EXÉCUTION ET AFFICHAGE
# ==========================================
# Le code ci-dessous ne s'exécute que si on clique sur le bouton "LANCER"
if do_train:
    # 1. Chargement des données via la fonction cachée
    df_full, final_cols, cat_feats, target = load_and_prep_data(csv_path)

    # Affiche un petit spinner "Chargement..." pendant les calculs
    with st.spinner("🧠 Les modèles s'affrontent..."):

        # --- Préparation finale pour Scikit-Learn ---
        X_raw = df_full[final_cols].copy()

        # get_dummies transforme le texte en colonnes de 0 et 1 (One-Hot Encoding)
        # drop_first=True évite la redondance (ex: si pas Paris et pas Lyon, c'est forcément Marseille)
        X_encoded = pd.get_dummies(X_raw, columns=cat_feats, drop_first=True).fillna(0)

        # NETTOYAGE CRITIQUE DES NOMS DE COLONNES
        # XGBoost et LightGBM plantent s'il y a des espaces ou symboles <, >, [ ] dans les noms de colonnes
        X_encoded = X_encoded.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

        y = df_full[target]

        # Séparation Train/Test
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=test_size,
                                                            random_state=random_state)

        # --- Entraînement des 3 Modèles ---

        # 1. Random Forest
        model_rf = RandomForestRegressor(
            n_estimators=rf_est,
            max_depth=rf_depth,
            min_samples_split=7,
            max_features=None,
            n_jobs=-1,
            random_state=random_state
        )
        model_rf.fit(X_train, y_train)

        # 2. XGBoost
        model_xgb = XGBRegressor(n_estimators=xgb_est, max_depth=xgb_depth, learning_rate=xgb_lr, n_jobs=-1,
                                 random_state=random_state)
        model_xgb.fit(X_train, y_train)

        # 3. LightGBM (Souvent le plus rapide)
        model_lgbm = LGBMRegressor(n_estimators=lgbm_est, max_depth=lgbm_depth, learning_rate=lgbm_lr, n_jobs=-1,
                                   random_state=random_state, verbose=-1)  # verbose=-1 pour le faire taire
        model_lgbm.fit(X_train, y_train)

    # --- SECTION 1 : DASHBOARD DES SCORES ---
    st.header("📊 Scoreboard Final")

    # Calcul des scores R² (Précision globale)
    r2_rf = r2_score(y_test, model_rf.predict(X_test))
    r2_xgb = r2_score(y_test, model_xgb.predict(X_test))
    r2_lgbm = r2_score(y_test, model_lgbm.predict(X_test))

    scores = {"RandomForest": r2_rf, "XGBoost": r2_xgb, "LightGBM": r2_lgbm}
    # Trouve qui a le score max
    winner_name = max(scores, key=scores.get)

    # Affichage en 3 colonnes
    c1, c2, c3 = st.columns(3)
    c1.metric("🌲 RandomForest", f"{r2_rf:.2%}")
    c2.metric("🚀 XGBoost", f"{r2_xgb:.2%}")
    c3.metric("⚡ LightGBM", f"{r2_lgbm:.2%}")

    # Annonce du vainqueur + Ballons
    st.success(f"🏆 Le gagnant est **{winner_name}** avec **{scores[winner_name]:.2%}** de précision !")
    st.balloons()

    # --- SECTION 2 : ANALYSE DÉTAILLÉE ---
    st.divider()
    # Création d'onglets pour organiser l'affichage
    t1, t2 = st.tabs(["🔬 Comparaison par Annonce", "📈 Importance des Variables"])

    # Onglet 1 : Tableau comparatif
    with t1:
        st.subheader(f"Top {nb_samples} prédictions (Prix en $)")
        # On prend quelques lignes au hasard dans le test
        idx = np.random.choice(X_test.index, size=min(nb_samples, len(X_test)), replace=False)

        # On crée un tableau comparatif
        # np.exp() est INDISPENSABLE car on a prédit le log_price, il faut revenir au prix réel ($)
        comp_df = pd.DataFrame({
            "Réel": np.exp(y_test.loc[idx]),
            "Pred_RF": np.exp(model_rf.predict(X_test.loc[idx])),
            "Pred_XGB": np.exp(model_xgb.predict(X_test.loc[idx])),
            "Pred_LGBM": np.exp(model_lgbm.predict(X_test.loc[idx]))
        })

        # Calcul de l'erreur pour le gagnant
        winning_model = {"RandomForest": model_rf, "XGBoost": model_xgb, "LightGBM": model_lgbm}[winner_name]
        prefix = 'RF' if winner_name == 'RandomForest' else 'XGB' if winner_name == 'XGBoost' else 'LGBM'
        comp_df["Erreur ($)"] = abs(comp_df["Réel"] - comp_df[f"Pred_{prefix}"])

        # Affichage stylisé avec couleurs (Rouge = grosse erreur, Vert = bonne prédiction)
        st.dataframe(comp_df.style.format("{:.0f}")
                     .background_gradient(subset=["Erreur ($)"], cmap="Reds")
                     .highlight_min(subset=["Pred_RF", "Pred_XGB", "Pred_LGBM"], color="lightgreen", axis=1))

    # Onglet 2 : Feature Importance (Qu'est-ce qui compte le plus ?)
    with t2:
        st.subheader(f"Qu'est-ce qui influence le prix selon {winner_name} ?")
        # On récupère les importances calculées par le modèle gagnant
        importances = winning_model.feature_importances_
        # On trie pour avoir les plus grandes barres en haut
        feat_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False).head(15)

        # Graphique
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis", ax=ax)
        ax.set_title(f"Top 15 Features - {winner_name}")
        st.pyplot(fig)  # Affiche le graphique matplotlib dans Streamlit

else:
    # Message d'accueil si on n'a pas encore cliqué
    st.info("👈 Configure tes modèles et lance la bataille dans la barre latérale !")