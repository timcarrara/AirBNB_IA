import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ===============================
# 1. CHARGEMENT DES DONNÉES
# ===============================
df = pd.read_csv("data.csv")  # Chargement du CSV
target = 'log_price'          # Variable à prédire

# Sélection des colonnes :
# - numériques (quantité, prix, coordonnées…)
num_cols = ['bathrooms', 'longitude', 'latitude', 'accommodates', 'number_of_reviews', 'bedrooms']
# - catégorielles (texte/catégories)
cat_cols = ['room_type', 'city']

X = df[num_cols + cat_cols]  # Features
y = df[target]               # Cible

# Séparation en données d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 2. PIPELINE DE NETTOYAGE ET ENCODAGE
# ===============================
# ColumnTransformer applique un traitement spécifique selon le type de colonne
preprocessor = ColumnTransformer(transformers=[
    # Colonnes numériques : remplacer les NaN par la médiane
    ('num', SimpleImputer(strategy='median'), num_cols),

    # Colonnes catégorielles : transformer le texte en colonnes binaires (One-Hot)
    # handle_unknown='ignore' gère les nouvelles catégories inconnues
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# Pipeline complet : nettoyage + modèle RandomForest
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42, n_jobs=-1))
])

# ===============================
# 3. OPTIMISATION DES HYPERPARAMÈTRES
# ===============================
param_grid = {
    'model__n_estimators': [200, 250],
    'model__max_depth': [20, 22],
    'model__min_samples_split': [5, 7],
    'model__max_features': ['sqrt']
}

search = RandomizedSearchCV(
    full_pipeline,
    param_distributions=param_grid,
    n_iter=4, cv=3,
    verbose=1,
    random_state=42,
    scoring='r2'
)

search.fit(X_train, y_train)  # Entraînement + recherche des meilleurs paramètres

# ===============================
# 4. AFFICHAGE DES RÉSULTATS
# ===============================
print("\n=== Meilleurs paramètres ===")
for param, value in search.best_params_.items():
    print(f"{param.replace('model__','')} : {value}")
print(f"Score R² CV : {search.best_score_:.4f}")

# ===============================
# 5. ÉVALUATION SUR LE TEST
# ===============================
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)
final_r2 = r2_score(y_test, y_pred)
print(f"Score final R² sur test : {final_r2:.4f}")

# Visualisation : prédictions vs réel
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, alpha=0.2, color='#2c3e50')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
plt.title(f"Prédictions vs Réalité (R² = {final_r2:.3f})")
plt.xlabel("Prix Réel (log)")
plt.ylabel("Prix Prédit (log)")
plt.grid(True, alpha=0.3)
plt.show()
