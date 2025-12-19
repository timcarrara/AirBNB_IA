import pandas as pd
import numpy as np
import re # Module pour les expressions régulières (manipulation de texte avancée)
from xgboost import XGBRegressor # Le modèle "Champion", souvent plus fort que Random Forest
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib # INDISPENSABLE : C'est lui qui permet de sauvegarder le modèle dans un fichier

# =========================================================
# 1. CHARGEMENT ET NETTOYAGE (LOGIQUE 71%)
# =========================================================
# Cette partie est cruciale : on prépare les ingrédients avant de cuisiner.

print("⏳ Chargement des données...")
try:
    df = pd.read_csv("data.csv")
except FileNotFoundError:
    print("❌ Fichier 'data.csv' introuvable.")
    exit()

# --- A. Filtrage des Outliers (Valeurs extrêmes) ---
# Pourquoi ? Si tu as des châteaux à 10 000$ ou des erreurs à 0$,
# cela perturbe le modèle. On garde les 98% "normaux" (entre 1% et 99%).
target = 'log_price'
low, high = df[target].quantile(0.01), df[target].quantile(0.99)
df = df[(df[target] >= low) & (df[target] <= high)]
print(f"✅ Outliers filtrés. Lignes restantes : {len(df)}")

# --- B. Nettoyage de base ---
# On remplace les vides (NaN) par du texte vide pour éviter les bugs lors des calculs de texte.
df['amenities'] = df['amenities'].fillna("")
df['description'] = df['description'].fillna("")

# --- C. Feature Engineering (Création de nouvelles colonnes) ---
# C'est l'art de donner des indices supplémentaires au modèle.

# 1. Compter les équipements et la longueur de la description
# Le modèle ne sait pas lire "TV, Wifi, Pool", mais il comprend "3 équipements".
df['amenities_count'] = df['amenities'].apply(lambda x: len(str(x).split(',')) if x else 0)
df['description_len'] = df['description'].apply(lambda x: len(str(x)) if x else 0)

# 2. Parsing des équipements Premium (La technique "One-Hot manuelle")
# On nettoie le texte des équipements (enlève les accolades {} et guillemets "")
df['amenities_clean'] = df['amenities'].str.replace('[{}"/]', '', regex=True)

# Liste des mots-clés qui font augmenter le prix
premium_amenities = ['Wifi', 'Air conditioning', 'Pool', 'Kitchen', 'Free parking', 'Gym', 'Hot tub', 'View']
new_cols = []

# Pour chaque mot-clé, on crée une colonne (ex: has_pool) avec 1 (oui) ou 0 (non)
for item in premium_amenities:
    col_name = f'has_{item.replace(" ", "_").lower()}'
    # .str.contains cherche le mot dans le texte
    df[col_name] = df['amenities_clean'].str.contains(item, case=False, regex=False).astype(int)
    new_cols.append(col_name)

# 3. Dates (Transformation en durée)
# Les ordis ne comprennent pas "12 Janvier 2015". Ils comprennent "Actif depuis 500 jours".
now = pd.to_datetime('2017-10-01') # Date de référence (fixée pour l'exercice)
for col in ['host_since', 'last_review']:
    df[col] = pd.to_datetime(df[col], errors='coerce') # Convertit en format Date

# On calcule la différence en jours
df['host_days_active'] = (now - df['host_since']).dt.days
df['days_since_review'] = (now - df['last_review']).dt.days

# =========================================================
# 2. DÉFINITION DES COLONNES (IDENTIQUE AU MODÈLE 3)
# =========================================================
# On liste ce qui rentre dans le modèle.
# Note : on ajoute les nouvelles colonnes créées plus haut (new_cols, amenities_count...)

# Variables Catégorielles (Texte -> deviendra des 0 et 1)
categorical_features = ['property_type', 'room_type', 'cancellation_policy', 'city', 'cleaning_fee', 'instant_bookable']

# Variables Numériques (Chiffres)
numeric_features = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 'latitude', 'longitude',
                    'number_of_reviews', 'review_scores_rating', 'amenities_count',
                    'description_len', 'host_days_active', 'days_since_review'] + new_cols

# Séparation X (Features) et y (Cible)
X = df[numeric_features + categorical_features]
y = df[target]

# =========================================================
# 3. CRÉATION DU PIPELINE DE PRODUCTION
# =========================================================
# Le Pipeline est la chaîne de montage automatisée.

# Le préprocesseur gère les NaNs et l'encodage texte automatiquement
preprocessor = ColumnTransformer(
    transformers=[
        # Pour les chiffres : on remplace les trous par la médiane
        ('num', SimpleImputer(strategy='median'), numeric_features),
        # Pour le texte : OneHotEncoder.
        # sparse_output=False : crée un tableau normal lisible.
        # handle_unknown='ignore' : CRUCIAL. Si demain une nouvelle ville apparait, le modèle ne plantera pas (il mettra 0).
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

# Modèle XGBoost
# XGBoost construit les arbres les uns après les autres pour corriger les erreurs des précédents.
model = XGBRegressor(
    n_estimators=1000,    # 1000 arbres (c'est beaucoup, donc précis)
    learning_rate=0.05,   # Vitesse d'apprentissage lente pour éviter le surapprentissage
    max_depth=6,          # Profondeur max des arbres
    n_jobs=-1,            # Utilise tous les coeurs du CPU
    random_state=42
)

# Pipeline final : D'abord le nettoyage, ensuite le modèle
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# =========================================================
# 4. ENTRAÎNEMENT ET VALIDATION
# =========================================================
# Ici, on teste d'abord si le modèle est bon en le coupant en deux.

print("\n" + "="*40)
print("🔍 VALIDATION AVANT SAUVEGARDE")
print("="*40)

# On coupe : 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# On entraîne sur les 80%
pipeline.fit(X_train, y_train)
# On prédit sur les 20% cachés
y_pred = pipeline.predict(X_test)

# On note la performance
score_r2 = r2_score(y_test, y_pred)
print(f"📊 Précision R² : {score_r2:.2%}") # Affiche en pourcentage (ex: 72.50%)

if score_r2 > 0.70:
    print("🚀 Score excellent (> 70%) ! Préparation du modèle final...")
else:
    print("⚠️  Le score est un peu plus bas que prévu. Vérifie tes hyperparamètres.")

# =========================================================
# 5. SAUVEGARDE FINALE
# =========================================================
# C'est l'étape "Mise en production".

# Pourquoi on refait un fit ?
# Maintenant qu'on sait que le modèle marche bien (grâce à l'étape 4),
# on veut qu'il apprenne sur 100% des données pour être le plus intelligent possible
# avant de l'enregistrer dans un fichier.
print("\n🔄 Entraînement final sur 100% du dataset...")
pipeline.fit(X, y)

# On sauvegarde tout le pipeline (nettoyage + modèle) dans un fichier .pkl
output_file = 'airbnb_model_prod.pkl'
joblib.dump(pipeline, output_file)

print(f"✅ MODÈLE SAUVEGARDÉ : {output_file}")
print("💡 Dans ton App Streamlit, tu n'auras qu'à faire : model = joblib.load('airbnb_model_prod.pkl')")