import streamlit as st
import pandas as pd
import googlemaps
import json
import os
from mistralai import Mistral
from time import sleep
import numpy as np

st.set_page_config(
    page_title="Assistant Airbnb IA",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Import de Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    .stApp {
        font-family: 'Poppins', sans-serif;
    }

    h1 {
        color: #FF5A5F;
        font-weight: 700;
        font-size: 3rem !important;
        text-align: center;
        padding: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    h2 {
        color: #484848;
        font-weight: 600;
        border-bottom: 3px solid #FF5A5F;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    
    h3 {
        color: #767676;
        font-weight: 500;
    }
 
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }

    .stButton>button {
        background: linear-gradient(90deg, #FF5A5F 0%, #FF385C 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 50px;
        box-shadow: 0 4px 15px rgba(255, 90, 95, 0.3);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 90, 95, 0.4);
        background: linear-gradient(90deg, #FF385C 0%, #FF5A5F 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f7f7f7 0%, #ffffff 100%);
    }
    
    [data-testid="stSidebar"] h2 {
        color: #FF5A5F;
        font-size: 1.5rem;
        text-align: center;
        border-bottom: 2px solid #FF5A5F;
        padding-bottom: 1rem;
    }
    

    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 10px;
        font-weight: 600;
        color: #484848;
    }
    
    .stRadio > div {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
    }

    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        font-size: 1rem;
        padding: 1rem;
    }
    
    .stTextArea textarea:focus {
        border-color: #FF5A5F;
        box-shadow: 0 0 0 0.2rem rgba(255, 90, 95, 0.25);
    }

    .stSelectbox > div > div {
        border-radius: 10px;
    }

    .stSpinner > div {
        border-top-color: #FF5A5F !important;
    }

    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #FF5A5F, transparent);
    }
    
    .stSuccess {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        border-radius: 5px;
    }
    
    .stError {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        border-radius: 5px;
    }

    .badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
    }

    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animated-card {
        animation: fadeIn 0.5s ease-out;
    }
    </style>
""", unsafe_allow_html=True)


# Configuration des clés API 
GOOGLE_API_KEY = "AIzaSyA1OsV1Nhzl2BN3I6EPrLh73CzA9G0yM6Q" 
MISTRAL_API_KEY = "7r3AZwvNae00ToPPtXydF2mHmipF7d5i" 

gmaps = googlemaps.Client(key=GOOGLE_API_KEY)
clientIA = Mistral(api_key=MISTRAL_API_KEY)


# Chargement des traductions
@st.cache_resource
def charger_traductions():
    try:
        with open('traductions.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Erreur : Le fichier 'traductions.json' est introuvable.")
        return {}
    except json.JSONDecodeError:
        st.error("Erreur : Le fichier 'traductions.json' est mal formaté.")
        return {}

TRADUCTIONS = charger_traductions()

def traduire(valeur, categorie):
    """Traduit une valeur selon sa catégorie."""
    return TRADUCTIONS.get(categorie, {}).get(valeur, valeur)


# Chargement des données
@st.cache_data
def charger_donnees():
    df = pd.read_csv('Airbnb_Data.csv')
    colonnes_interet = ['id', 'city', 'neighbourhood', 'latitude', 'longitude', 'property_type', 'room_type', 'accommodates', 'bedrooms', 'beds', 'bathrooms', 'amenities', 'name', 'log_price', 'bed_type', 'review_scores_rating', 'cancellation_policy', 'instant_bookable']
    colonnes_essentielles = ['id', 'city', 'neighbourhood', 'latitude', 'longitude', 'property_type', 'room_type', 'accommodates', 'bedrooms', 'beds', 'bathrooms', 'amenities', 'name']
    return df[colonnes_interet].dropna(subset=colonnes_essentielles)

df_filtre = charger_donnees()


# Gestion du cache pour Google Place
def charger_cache(nom_fichier='cache_places.json'):
    if os.path.exists(nom_fichier):
        try:
            with open(nom_fichier, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            st.warning("Le fichier cache_places.json est corrompu. Création d'un nouveau cache.")
            return {}
    return {}

def sauvegarder_cache(cache, nom_fichier='cache_places.json'):
    try:
        with open(nom_fichier, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Erreur lors de la sauvegarde du cache : {e}")

def get_nearby_places(lat, lng, cache, type_place='tourist_attraction', radius=1000):
    key = f"{lat},{lng},{type_place},{radius}"
    if key in cache:
        return cache[key]
    try:
        places_result = gmaps.places_nearby(
            location=(lat, lng),
            radius=radius,
            type=type_place,
            language='fr' 
        )
        results_to_cache = places_result.get('results', [])[:10] 
        cache[key] = results_to_cache
        sauvegarder_cache(cache)
        return results_to_cache
    except Exception as e:
        st.error(f"Erreur Google Maps pour (lat={lat}, lng={lng}): {e}")
        return []


# Appeler l'IA générative
def appeler_ia_generative(prompt, model="mistral-large-latest"):
    try:
        chat_response = clientIA.chat.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=700
        )
        return chat_response.choices[0].message.content
    except Exception as e:
        st.error(f"Erreur lors de l'appel à l'API Mistral : {e}")
        return "Désolé, l'IA ne peut pas répondre pour le moment."

def generer_description_ia(logement, cache_places):
    property_type_fr = traduire(logement['property_type'], 'property_types')
    room_type_fr = traduire(logement['room_type'], 'room_types')
    bed_type_fr = traduire(logement['bed_type'], 'bed_types')
    politique_annulation_fr = traduire(logement.get('cancellation_policy', 'non spécifiée'), 'cancellation_policies')

    lat = logement['latitude']
    lng = logement['longitude']
    
    places_touristiques = get_nearby_places(lat, lng, cache_places, type_place='tourist_attraction', radius=5000)
    places_transport = get_nearby_places(lat, lng, cache_places, type_place='transit_station', radius=1000)
    
    poi_noms = [place['name'] for place in places_touristiques[:5]]
    transport_noms = [place['name'] for place in places_transport[:5]]

    poi_texte = ""
    if poi_noms:
        poi_texte += f"Les attractions majeures à proximité sont : {', '.join(poi_noms)}. "
    if transport_noms:
        poi_texte += f"L'accès est facile grâce aux transports publics proches : {', '.join(transport_noms)}. "
        
    log_prix_base = logement.get('log_price', 0)
   
    if pd.notna(log_prix_base) and log_prix_base > 0:
        prix_estime = round(np.exp(log_prix_base))
    else:
        prix_estime = "un prix compétitif"

    note_avis = logement.get('review_scores_rating', 'Non noté')
    reservation_inst = "Oui" if logement.get('instant_bookable') else "Non"
    
    amenities = logement['amenities'].replace('{', '').replace('}', '').replace('"', '').replace(',', ', ')

    prompt = (
        "En tant qu'expert en marketing de voyage, rédige une description complète, structurée en paragraphes, et captivante pour cette annonce Airbnb. "
        "\n\n--- Informations sur le logement ---\n"
        f"Titre de l'annonce : **{logement['name']}**.\n"
        f"Type de bien : **{property_type_fr}** (mis à disposition en **{room_type_fr}**).\n"
        f"Localisation : **{logement['neighbourhood']}**, **{logement['city']}**.\n"
        f"Capacité : Accueille **{logement['accommodates']}** personnes. Il dispose de **{logement['bedrooms']}** chambres, **{logement['beds']}** lits (type principal : **{bed_type_fr}**) et **{logement['bathrooms']}** salles de bain.\n"
        f"Équipements : **{amenities}**.\n"
        "\n\n--- Points Forts et Proximité ---\n"
        f"Points d'intérêt aux alentours : {poi_texte if poi_texte else 'Le quartier est calme et bien desservi, mais aucun lieu célèbre n\'est spécifiquement listé à proximité immédiate.'}"
        "\n\n--- Qualité et Réservation ---\n"
        f"Note des voyageurs : **{note_avis}/100**.\n"
        f"Politique d'annulation : **{politique_annulation_fr}**.\n"
        f"Réservation instantanée : **{reservation_inst}**.\n"
        f"(Prix estimé par nuit : environ **{prix_estime}** USD, pour donner une idée du standing).\n"
        "\n\n--- Instructions pour l'IA ---\n"
        "1. **Toute la description doit être rédigée exclusivement en français.**\n" 
        "2. **Commencez par un titre accrocheur** basé sur le 'Titre de l'annonce' mais plus élaboré.\n"
        "3. **Mettez en valeur les équipements** (`amenities`) les plus courants, en **TRADUISANT SYSTÉMATIQUEMENT TOUS LES TERMES** de l'anglais vers le français.\n" 
        "4. **Utilisez un ton persuasif et haut de gamme**."
    )

    return appeler_ia_generative(prompt)

def generer_description_personnalisee(criteria):
    prompt = (
        "Tu es un expert en rédaction marketing Airbnb. "
        "L'utilisateur fournit des critères en langage naturel décrivant le type de logement qu'il souhaite. "
        "À partir de ces critères, rédige une description complète, élégante, persuasive et entièrement en français. "
        "Structure le texte en paragraphes, ajoute un titre accrocheur, et embellis légèrement le récit "
        "tout en restant cohérent avec les critères fournis.\n\n"
        f"--- Critères fournis par l'utilisateur ---\n{criteria}\n\n"
        "--- Instructions ---\n"
        "1. Ne pas inventer des données techniques trop spécifiques sauf si logique.\n"
        "2. Toujours rester plausible.\n"
        "3. Ton haut de gamme, rassurant, immersif.\n"
        "4. Toujours rédiger en français.\n"
    )
    return appeler_ia_generative(prompt)

def generer_idees_visite_ia(lat, lng, cache, type_place='tourist_attraction'):
    """Génère des idées d'activités pour un type de lieu donné."""
    places = get_nearby_places(lat, lng, cache, type_place)
    if not places:
        return "Aucune idée de visite de ce type trouvée à proximité (vérifiez que la clé Google Maps est valide)."

    lieux = "\n".join([f"- {place['name']} ({place['vicinity']})" for place in places[:5]])
    
    prompt = (
        f"Propose une liste d'activités attrayantes pour des voyageurs visitant le quartier. Le thème est : **{type_place.replace('_', ' ').upper()}**. "
        f"Voici quelques lieux à proximité :\n{lieux}\n\n"
        f"Formule une réponse naturelle et engageante, en mettant en avant les incontournables et en suggérant une petite description de chaque lieu. **Rédige la réponse uniquement en français.**"
    )
    return appeler_ia_generative(prompt)

def generer_planning_ia(lat, lng, cache, duree=7):
    """Génère un planning jour par jour pour le séjour, avec des activités variées chaque jour."""
    planning = {}
    types_lieux = ['tourist_attraction', 'restaurant', 'park', 'museum', 'cafe', 'bar', 'shopping_mall']

    idees_par_type = {}
    for type_lieu in types_lieux:
        with st.spinner(f"⏳ Recherche d'idées de type '{type_lieu.replace('_', ' ')}'..."):
            idees = generer_idees_visite_ia(lat, lng, cache, type_lieu)
            idees_par_type[type_lieu] = idees

    repartition_par_jour = {
        'tourist_attraction': 2,
        'restaurant': 2,
        'park': 1,
        'museum': 1,
        'cafe': 1,
        'bar': 1, 
        'shopping_mall': 0 
    }

    for jour in range(1, duree + 1):
        activites_jour = []
        for type_lieu, nb_activites in repartition_par_jour.items():
            if nb_activites > 0 and type_lieu in idees_par_type:
                activites = idees_par_type[type_lieu].split('\n')[:nb_activites]
                activites_jour.extend(activites)

        import random
        random.shuffle(activites_jour)

        prompt = (
            f"Crée un planning détaillé pour le **Jour {jour}** d'un séjour touristique. "
            f"Voici une liste variée d'activités et de lieux à visiter :\n"
            f"{'\n'.join(activites_jour)}\n\n"
            f"Structure la journée avec des horaires approximatifs (ex. : matin, midi, après-midi, soirée), "
            f"en mélangeant les types d'activités de manière réaliste. "
            f"Ajoute des conseils pratiques et des transitions logiques entre les activités. "
            f"**Rédige la réponse uniquement en français.**"
        )

        with st.spinner(f"⏳ Jour {jour} : Génération du planning varié..."):
            planning[f"Jour {jour}"] = appeler_ia_generative(prompt)

    return planning


# Interface Streamlit

cache = charger_cache()

st.markdown("<h1>🏠 AirBNB-IA</h1>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: #767676; font-size: 1.2rem; width: 80%; margin: 0 auto;'>
        Générez des descriptions sur mesure pour trouver ou promouvoir votre logement, explorez les environs et concevez votre itinéraire parfait avec l'IA générative
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# SIDEBAR
with st.sidebar:
    st.markdown("## 🎯 Sélection du Logement")
    
    villes = df_filtre['city'].unique()
    ville = st.selectbox("📍 Ville", sorted(villes), key="select_ville")
    
    quartiers = df_filtre[df_filtre['city'] == ville]['neighbourhood'].unique()
    quartier = st.selectbox("🏘️ Quartier", sorted(quartiers), key="select_quartier")
    
    df_quartier = df_filtre[(df_filtre['city'] == ville) & (df_filtre['neighbourhood'] == quartier)]
    
    if df_quartier.empty:
        st.error("❌ Aucun logement trouvé pour cette sélection.")
        st.stop()
    
    logement_id = st.selectbox("🏡 Logement", df_quartier['name'], key="select_logement")
    logement = df_quartier[df_quartier['name'] == logement_id].iloc[0].to_dict()
    lat, lng = logement['latitude'], logement['longitude']
    
    st.markdown("---")
    st.markdown("### 📊 Statistiques")
    st.metric("🏘️ Quartiers disponibles", len(quartiers))
    st.metric("🏠 Logements totaux", len(df_quartier))

# ZONE PRINCIPALE
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"### 🏠 {traduire(logement['property_type'], 'property_types')}")
    st.markdown(f"**📍 {quartier}, {ville}**")

with col2:
    if pd.notna(logement.get('review_scores_rating')):
        st.metric("⭐ Note", f"{logement['review_scores_rating']}/100")

st.markdown(f"""
    <div style='margin: 0 0.5rem 0.5rem 0.5rem;'>
        <span class='badge'>👥 {logement['accommodates']} personnes</span>
        <span class='badge'>🛏️ {int(logement['bedrooms'])} chambre{'s' if int(logement['bedrooms']) > 1 else ''}</span>
        <span class='badge'>🛋️ {int(logement['beds'])} lit{'s' if int(logement['beds']) > 1 else ''}</span>
        <span class='badge'>🚿 {int(logement['bathrooms'])} salle{'s' if int(logement['bathrooms']) > 1 else ''} de bain</span>
        <span class='badge'>📋 {traduire(logement['room_type'], 'room_types')}</span>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown("## 🎨 Fonctionnalités IA")

col1, col2, col3, col4 = st.columns(4)
onglets = [
    ("📝 Description", "Description du logement"),
    ("🗺️ Idées", "Idées de visite à thème"),
    ("📅 Planning", "Planning pour le séjour"),
    ("✍️ Personnalisé", "Description personnalisée (critères utilisateur)")
]

onglet = st.radio(
    "Choisissez une fonctionnalité :",
    [o[1] for o in onglets],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("<br>", unsafe_allow_html=True)

if onglet == "Description du logement":
    st.markdown("### 📝 Description Marketing")
    
    if st.button("✨ Générer la Description", key="btn_desc"):
        with st.spinner("🤖 L'IA Mistral rédige la description..."):
            description = generer_description_ia(logement, cache)
            st.markdown(f"""
                <div class='animated-card' style='background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                    {description}
                </div>
            """, unsafe_allow_html=True)

elif onglet == "Idées de visite à thème":
    st.markdown(f"### 🗺️ Découvrez {quartier}")
    
    type_lieu_options = {
        'Attractions touristiques 🎡': 'tourist_attraction',
        'Restaurants 🍽️': 'restaurant',
        'Parcs 🌳': 'park',
        'Musées 🎨': 'museum',
        'Cafés ☕': 'cafe',
        'Centres commerciaux 🛍️': 'shopping_mall',
        'Bars 🍻': 'bar'
    }
    
    type_lieu_label = st.selectbox("Choisissez un type d'activité", list(type_lieu_options.keys()), key="select_type_lieu")
    type_lieu = type_lieu_options[type_lieu_label]
    
    if st.button(f"🔍 Découvrir les {type_lieu_label.lower()}", key="btn_idees"):
        with st.spinner(f"🔎 Recherche en cours..."):
            idees = generer_idees_visite_ia(lat, lng, cache, type_lieu)
            st.markdown(f"""
                <div class='animated-card' style='background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                    {idees}
                </div>
            """, unsafe_allow_html=True)

elif onglet == "Planning pour le séjour":
    st.markdown(f"### 📅 Planning de Voyage")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        duree = st.slider("🕐 Durée du séjour (jours)", 1, 14, 7)
    with col2:
        st.markdown(f"<div style='text-align: center; padding-top: 1rem;'><h2>{duree}</h2><p>jours</p></div>", unsafe_allow_html=True)
    
    if st.button(f"🗓️ Générer le Planning", key="btn_planning"):
        with st.container():
            planning = generer_planning_ia(lat, lng, cache, duree)
            
            for jour, activites in planning.items():
                try:
                    lignes_non_vides = [l.strip() for l in activites.split('\n') if l.strip() and not l.strip().startswith('#')]
                    titre_activite = lignes_non_vides[0].strip('*- ').split('(')[0].strip()
                    if len(titre_activite) > 50:
                        titre_activite = titre_activite[:50] + "..." 
                except IndexError:
                    titre_activite = "Cliquer pour voir le détail"
                
                with st.expander(f"🗓️ **{jour}** - {titre_activite}", expanded=False):
                    st.markdown(activites)

elif onglet == "Description personnalisée (critères utilisateur)":
    st.markdown("### ✍️ Description Sur Mesure")
    st.info("💡 Décrivez votre logement idéal en langage naturel")
    
    criteres = st.text_area(
        "Vos critères :",
        placeholder="Exemple : Un loft moderne à New York avec vue panoramique, 3 chambres, cuisine américaine équipée, salle de sport, proche de Central Park, idéal pour les familles...",
        height=180,
        key="textarea_criteres"
    )
    
    if st.button("🎨 Créer la Description", key="btn_perso"):
        if criteres.strip():
            with st.spinner("✨ Création de votre description personnalisée..."):
                description_perso = generer_description_personnalisee(criteres)
                st.markdown(f"""
                    <div class='animated-card' style='background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                        {description_perso}
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Veuillez d'abord saisir vos critères.")

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #999; padding: 2rem 0;'>
        <p style='font-size: 0.9rem;'>Assistant Airbnb Génératif © 2025</p>
    </div>
""", unsafe_allow_html=True)

sauvegarder_cache(cache)
