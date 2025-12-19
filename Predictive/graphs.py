# ====================================================
# 1. IMPORTATION DES BIBLIOTHÈQUES
# ====================================================
import pandas as pd         # Pour manipuler les données (tableaux type Excel)
import matplotlib.pyplot as plt # Pour tracer des graphiques de base
import seaborn as sns       # Pour des graphiques plus beaux et statistiques
# Importation de la fonction qui permet de couper nos données en deux (Train / Test)
from sklearn.model_selection import train_test_split

# ----------------------------------------------------
# Charger le fichier CSV
# On lit le fichier 'data.csv' et on le stocke dans une variable 'df' (DataFrame)
df = pd.read_csv("data.csv")

# ====================================================
#     PARTIE 0 : SÉPARATION TRAIN / TEST (80/20)
# ====================================================
# C'est l'étape la plus importante pour éviter la triche !
# On cache 20% des données (Test) que le modèle ne verra JAMAIS pendant l'entraînement.
# - test_size=0.2 : 20% pour le test, 80% pour l'entraînement.
# - random_state=42 : C'est comme figer le hasard. Si tu relances le code, 
#   le mélange sera exactement le même (pratique pour comparer les résultats).
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# On affiche juste les tailles pour vérifier que la coupure a bien fonctionné.
print("--- Séparation des données ---")
print(f"Taille totale du dataset : {len(df)}")
print(f"Taille du set d'entraînement (80%) : {len(df_train)}")
print(f"Taille du set de test (20%) : {len(df_test)}")
print("------------------------------\n")

# ATTENTION : À partir d'ici, on ne touche plus à 'df_test' jusqu'à la toute fin.
# Toutes les analyses et graphiques se font UNIQUEMENT sur 'df_train'.

# ----------------------------------------------------
# Exploration rapide des données
print("Aperçu des données (train) :")
print(df_train.head()) # Affiche les 5 premières lignes

print("\nColonnes :")
print(df_train.columns.tolist()) # Affiche la liste des noms de colonnes

print("\nInformations sur le dataset (train) :")
# .info() est très utile : il dit s'il y a des valeurs manquantes (null) 
# et le type de données (int, float, object/texte).
print(df_train.info())

print("\nStatistiques descriptives (train) :")
# .describe() donne la moyenne, le min, le max, la médiane (50%) des chiffres.
print(df_train.describe())
# ----------------------------------------------------


# ====================================================
#         PARTIE 1 : VISUALISATION (sur df_train)
# ====================================================
# Le but ici est de comprendre les données avec les yeux avant de lancer les maths.

# Définit le style visuel des graphiques (fond blanc avec grille)
sns.set(style="whitegrid")

# ---
## 1. Distribution de la variable cible (log_price)
# On regarde à quoi ressemble la courbe des prix.
print("\nAffichage des graphiques (sur 80% des données)...")

plt.figure(figsize=(10, 6))
# histplot dessine un histogramme. 'kde=True' ajoute la ligne de courbe lisse par-dessus.
sns.histplot(df_train['log_price'], kde=True, bins=50) 
plt.title('Distribution du Log-Prix (Train Set)')
plt.xlabel('Log-Prix')
plt.ylabel('Fréquence')
plt.show()

# ---
## 2. Relation prix et variables catégorielles (Type de chambre)
# Le Boxplot (boîte à moustaches) est génial pour voir la médiane et les écarts.
plt.figure(figsize=(12, 7))
sns.boxplot(x='room_type', y='log_price', data=df_train) 
plt.title('Log-Prix en fonction du Type de Chambre (Train Set)')
plt.xlabel('Type de Chambre')
plt.ylabel('Log-Prix')
plt.show()

# ---
## 3. Relation prix et variables numériques (Capacité d'accueil)
# Est-ce que plus on peut accueillir de gens, plus c'est cher ?
plt.figure(figsize=(12, 7))
sns.boxplot(x='accommodates', y='log_price', data=df_train) 
plt.title('Log-Prix en fonction du Nombre de Personnes Accueillies (Train Set)')
plt.xlabel('Nombre de personnes (accommodates)')
plt.ylabel('Log-Prix')
plt.show()

# ---
## 4. Comptage des Types de Propriété
# On compte combien il y a d'appartements, de maisons, etc.
plt.figure(figsize=(12, 10))
# countplot compte les occurrences. 'order' permet de trier du plus fréquent au moins fréquent.
sns.countplot(y='property_type', data=df_train, order=df_train['property_type'].value_counts().index) 
plt.title('Nombre de logements par Type de Propriété (Train Set)')
plt.xlabel('Nombre')
plt.ylabel('Type de Propriété')
plt.show()

# ---
## 5. Matrice de corrélation (Heatmap)
# C'est une carte de chaleur pour voir quelles colonnes numériques varient ensemble.
# Si c'est rouge (proche de 1), quand l'une monte, l'autre monte aussi.
plt.figure(figsize=(12, 10))
# On ne garde que les chiffres pour faire des maths de corrélation
numeric_cols = df_train.select_dtypes(include=['float64', 'int64', 'bool']) 
corr_matrix = numeric_cols.corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matrice de Corrélation des Variables Numériques (Train Set)')
plt.show()

# ---
## 6. Analyse géographique
# On dessine les points sur un plan latitude/longitude.
plt.figure(figsize=(12, 10))
sns.scatterplot(
    x='longitude',
    y='latitude',
    data=df_train,
    hue='log_price', # La couleur change selon le prix
    palette='viridis', # Nuance de couleurs
    alpha=0.3,       # Transparence (pour voir si les points se superposent)
    s=5,             # Taille des points
    legend=True
)
plt.title('Distribution Géographique des Logements (Train Set)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

print("Graphiques (Partie 1) affichés.")


# ====================================================
#   PARTIE 2 : CORRÉLATION PRIX VS. CATÉGORIES (sur df_train)
# ====================================================

print("\nAffichage des graphiques (Prix vs. Catégories sur Train Set)...")
sns.set(style="whitegrid")

# ---
## 1. Prix en fonction de la Ville
plt.figure(figsize=(12, 7))
sns.boxplot(x='city', y='log_price', data=df_train) 
plt.title('Log-Prix en fonction de la Ville (Train Set)')
plt.xlabel('Ville')
plt.ylabel('Log-Prix')
plt.show()

# ---
## 2. Prix en fonction du Type de Propriété (Top 10 seulement)
# Il y a trop de types de propriétés, on ne garde que les 10 plus fréquents pour que le graphique soit lisible.
top_10_property_types = df_train['property_type'].value_counts().nlargest(10).index 
df_top10_prop = df_train[df_train['property_type'].isin(top_10_property_types)] 

plt.figure(figsize=(16, 8))
sns.boxplot(x='property_type', y='log_price', data=df_top10_prop, order=top_10_property_types)
plt.title('Log-Prix pour les 10 Types de Propriété (Train Set)')
plt.xlabel('Type de Propriété')
plt.ylabel('Log-Prix')
plt.xticks(rotation=45) # On penche les textes pour qu'ils ne se chevauchent pas
plt.show()

# ---
## 3. Prix vs Frais de Ménage
# 'cleaning_fee' est souvent True ou False (ou montant). Ici on voit l'impact global.
plt.figure(figsize=(8, 6))
sns.boxplot(x='cleaning_fee', y='log_price', data=df_train) 
plt.title('Log-Prix avec ou sans Frais de Ménage (Train Set)')
plt.xlabel('Frais de Ménage (cleaning_fee)')
plt.ylabel('Log-Prix')
plt.show()

# ---
## 4. Prix vs Politique d'Annulation
plt.figure(figsize=(12, 7))
sns.boxplot(x='cancellation_policy', y='log_price', data=df_train) 
plt.title('Log-Prix en fonction de la Politique d\'Annulation (Train Set)')
plt.xlabel('Politique d\'Annulation')
plt.ylabel('Log-Prix')
plt.show()

print("Tous les graphiques (Partie 1 & 2) sont affichés.")
print("\nLe set de test (df_test) est prêt et mis de côté.")


# ====================================================
#     PARTIE 3 : PRÉPARATION ET MODÉLISATION
# ====================================================

# Imports spécifiques pour construire l'intelligence artificielle
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor # Notre modèle (Forêt aléatoire)
from sklearn.impute import SimpleImputer # Pour remplir les trous (valeurs manquantes)
from sklearn.preprocessing import OneHotEncoder # Pour transformer le texte en chiffres
from sklearn.pipeline import Pipeline # Pour créer une chaîne d'actions automatique
from sklearn.compose import ColumnTransformer # Pour traiter différemment colonnes numériques et textes
from sklearn.metrics import mean_squared_error, r2_score # Pour noter le modèle

print("\n--- Début de la préparation des données ---")

# --- 1. Sélection des Features (Caractéristiques) ---
# 'target' c'est la CIBLE : ce qu'on veut prédire.
target = 'log_price'

# On liste les colonnes qu'on va donner au modèle pour qu'il apprenne.
# On divise en deux listes car on ne les traite pas pareil.

# Liste des colonnes avec des CHIFFRES
numerical_features = [
    'accommodates', 'bathrooms', 'bedrooms', 'beds',
    'latitude', 'longitude', 'number_of_reviews', 'review_scores_rating'
]

# Liste des colonnes avec du TEXTE (Catégories)
categorical_features = [
    'property_type', 'room_type', 'cancellation_policy', 
    'cleaning_fee', 'city'
]

# On combine les deux pour avoir la liste complète des ingrédients
features = numerical_features + categorical_features

# --- Création des X (Ingrédients) et Y (Cible) ---
# X = les données d'entrée, y = la réponse attendue

# Pour l'ENTRAÎNEMENT
X_train = df_train[features]
y_train = df_train[target]

# Pour le TEST (on prépare déjà, pour plus tard)
X_test = df_test[features]
y_test = df_test[target]

print(f"Features sélectionnées : {features}")

# --- 2. Pipeline de Prétraitement (Nettoyage automatique) ---

# A. Traitement des chiffres
# S'il manque une valeur (NaN), on met la MÉDIANE de la colonne à la place.
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# B. Traitement du texte
# 1. S'il manque une valeur, on met la plus fréquente.
# 2. On transforme le texte en chiffres binaires (OneHotEncoder).
#    Exemple : "Paris" devient [1, 0, 0], "Lyon" devient [0, 1, 0].
#    handle_unknown='ignore' : Si une ville inconnue apparait dans le futur, on ne plante pas, on met des 0 partout.
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# C. Le Chef d'orchestre (ColumnTransformer)
# Il envoie les colonnes numériques vers 'numerical_transformer'
# et les colonnes textes vers 'categorical_transformer'.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# --- 3. Création du Modèle (Le Cerveau) ---
# On utilise un Random Forest (Forêt Aléatoire).
# C'est une multitude d'arbres de décision qui votent pour donner un prix.
# - n_estimators=100 : On crée 100 arbres.
# - max_depth=15 : On limite la profondeur des arbres pour éviter qu'ils apprennent "par cœur" (overfitting).
# - n_jobs=-1 : On utilise toute la puissance du processeur de l'ordi.
model_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15) 

# --- 4. Pipeline Complet ---
# On crée un "Tuyau" unique qui fait : 
# Données brutes -> Nettoyage (preprocessor) -> Prédiction (model)
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model_rf)])

# --- 5. Entraînement ---
print("\n--- Début de l'entraînement du Random Forest (cela peut prendre un moment)... ---")
# C'est ici que la magie opère. La méthode .fit() lance l'apprentissage.
# Le modèle regarde X_train et essaie de deviner y_train.
pipeline.fit(X_train, y_train)
print("--- Entraînement terminé ! ---")


# ====================================================
#     PARTIE 4 : ÉVALUATION DU MODÈLE
# ====================================================

print("\n--- Évaluation du modèle ---")

# --- 6. Prédictions ---
# Maintenant que le modèle est entraîné, on lui demande de prédire.
# On prédit sur le TRAIN (pour voir s'il a bien appris sa leçon)
y_pred_train = pipeline.predict(X_train)
# On prédit sur le TEST (C'est le vrai examen : des données qu'il n'a jamais vues)
y_pred_test = pipeline.predict(X_test)

# --- Calcul des notes (Métriques) ---

# RMSE (Root Mean Squared Error) : L'erreur moyenne au carré.
# Plus c'est proche de 0, mieux c'est. C'est l'écart moyen entre prédiction et réalité.
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

# R² (R-carré) : Note sur 1 (ou 100%).
# 1 = Prédiction parfaite. 0 = Le modèle est nul (aussi bon qu'une simple moyenne).
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

# Affichage des résultats
print("\n--- Performances du modèle ---")
print(f"Score sur le set d'ENTRAÎNEMENT (Train) :")
print(f"  RMSE : {rmse_train:.4f} (log_price)")
print(f"  R²   : {r2_train:.4f}") # Exemple: 0.85 veut dire qu'on explique 85% des variations de prix
print(f"\nScore sur le set de TEST :")
print(f"  RMSE : {rmse_test:.4f} (log_price)")
print(f"  R²   : {r2_test:.4f}")

# Petit diagnostic automatique
if r2_train > r2_test + 0.1:
    print("\nINFO : Le score d'entraînement est bien meilleur que le score de test.")
    print("      -> C'est un signe de 'surapprentissage' (overfitting).")
    print("      -> Le modèle a appris par cœur les données d'entraînement mais généralise mal.")
else:
     print("\nINFO : Les scores Train et Test sont proches. C'est bon signe !")


# --- 7. Bonus : Importance des features ---
# On va demander au modèle : "Quels critères ont le plus influencé le prix ?"

# (Petite gymnastique technique pour récupérer les noms des colonnes après transformation)
ohe_feature_names = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
all_feature_names = numerical_features + list(ohe_feature_names)

# On récupère les scores d'importance calculés par le Random Forest
importances = pipeline.named_steps['model'].feature_importances_

# On met tout ça dans un tableau propre et on trie
feature_importance_df = pd.DataFrame({
    'feature': all_feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print("\n--- Top 20 des features les plus importantes ---")
print(feature_importance_df.head(20))

# Graphique des importances
plt.figure(figsize=(10, 12))
sns.barplot(
    x='importance',
    y='feature',
    data=feature_importance_df.head(20),
    palette='viridis'
)
plt.title('Top 20 des features les plus importantes')
plt.show()

# --- 8. Bonus : Graphique des prédictions ---
# On compare visuellement ce que le modèle a prédit (y) vs la réalité (x)
plt.figure(figsize=(10, 8))
sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.3)
# On trace une ligne rouge diagonale parfaite.
# Si tous les points bleus sont sur la ligne rouge, le modèle est parfait.
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', linewidth=2)
plt.title('Prédictions vs Vraies Valeurs (Test Set)')
plt.xlabel('Vraies Valeurs (y_test)')
plt.ylabel('Prédictions (y_pred_test)')
plt.show()