import streamlit as st
from PIL import Image
import numpy as np
import json
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore
import os
import uuid
import pathlib
import time

# CONFIG STREAMLIT
st.set_page_config(page_title=" AgriDetect", layout="centered", page_icon="🌿")

# STYLESER
def show_footer():
    st.markdown("""
        <hr style="border: 1px solid #ddd;">
        <div style='text-align: center; color: gray;'>
            Classification des cultures agricoles, développée par A.Ablaye GIT
        </div>
    """, unsafe_allow_html=True)

def divider():
    st.markdown("<hr style='border:1px solid #EEE;'>", unsafe_allow_html=True)

# SIDEBAR PERSONNALISÉE
with st.sidebar:
    st.image("CropDetect.png", use_column_width=True)  
    st.title("🌍 Langue / Language")
    langue = st.selectbox("Choisissez votre langue", ["Français", "Anglais", "Arabe"])
    st.markdown("---")
    st.markdown("📁 [Télécharger le dataset](https://...)")
    st.markdown("💻 [Voir le code source](https://github.com/...)")
    st.markdown("✉️ Contact : [ablaye@sylla.com](mailto:email@example.com)")

# TITRE MULTILINGUE 
titres = {
    "Français": "🌱 Détection intelligente des cultures agricoles",
    "Anglais": "🌱 Smart detection of agricultural crops",
    "Arabe": "🌱 الكشف الذكي عن المحاصيل الزراعية"
}

sous_titres = {
    "Français": "Importez une image de plante ou culture pour que le modèle vous dise de quoi il s'agit 🌾.",
    "Anglais": "Upload a plant or crop image to identify it using the model 🌾.",
    "Arabe": "حمّل صورة لنبات أو محصول للتعرف عليه باستخدام النموذج 🌾."
}

st.markdown(f"## {titres[langue]}")
st.write(sous_titres[langue])
divider()

#TRADUCTIONS CLASSES
traductions = {
    "Cherry": {"fr": "Cerise", "ar": "كرز"},
    "Coffee-plant": {"fr": "Caféier", "ar": "نبتة القهوة"},
    "Cucumber": {"fr": "Concombre", "ar": "خيار"},
    "Fox_nut(Makhana)": {"fr": "Graine de renard", "ar": "مخانا"},
    "Lemon": {"fr": "Citron", "ar": "ليمون"},
    "Olive-tree": {"fr": "Olivier", "ar": "شجرة الزيتون"},
    "Pearl_millet(bajra)": {"fr": "Mil perlé", "ar": "دخن لؤلؤي"},
    "Tobacco-plant": {"fr": "Plante de tabac", "ar": "نبتة التبغ"},
    "almond": {"fr": "Amande", "ar": "لوز"},
    "banana": {"fr": "Banane", "ar": "موز"},
    "cardamom": {"fr": "Cardamome", "ar": "هيل"},
    "chilli": {"fr": "Piment", "ar": "فلفل حار"},
    "clove": {"fr": "Clou de girofle", "ar": "قرنفل"},
    "coconut": {"fr": "Noix de coco", "ar": "جوز الهند"},
    "cotton": {"fr": "Coton", "ar": "قطن"},
    "gram": {"fr": "Pois chiche", "ar": "حمص"},
    "jowar": {"fr": "Sorgho", "ar": "جوار"},
    "jute": {"fr": "Jute", "ar": "الجوت"},
    "maize": {"fr": "Maïs", "ar": "ذرة"},
    "mustard-oil": {"fr": "Moutarde", "ar": "خردل"},
    "papaya": {"fr": "Papaye", "ar": "بابايا"},
    "pineapple": {"fr": "Ananas", "ar": "أناناس"},
    "rice": {"fr": "Riz", "ar": "أرز"},
    "soyabean": {"fr": "Soja", "ar": "فول الصويا"},
    "sugarcane": {"fr": "Canne à sucre", "ar": "قصب السكر"},
    "sunflower": {"fr": "Tournesol", "ar": "دوار الشمس"},
    "tea": {"fr": "Thé", "ar": "شاي"},
    "tomato": {"fr": "Tomate", "ar": "طماطم"},
    "vigna-radiati(Mung)": {"fr": "Haricot mungo", "ar": "مونج"},
    "wheat": {"fr": "Blé", "ar": "قمح"}
}

# CHARGEMENT DU MODÈLE 
@st.cache_resource
def load_model_and_classes():
    model = load_model("model-final.h5")
    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)
    index_to_class = {v: k for k, v in class_indices.items()}
    return model, index_to_class, class_indices

model, index_to_class, class_indices = load_model_and_classes()

#  INITIALISER L'HISTORIQUE 
if "historique" not in st.session_state:
    st.session_state.historique = []

#  UPLOAD IMAGE 
uploaded_file = st.file_uploader("📷 Choisissez une image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Image importée", use_column_width=True)

    # Spinner = Analyse en cours
    with st.spinner("🔎 Analyse en cours..."):
        time.sleep(1.5)

        img = image.resize((224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_class = index_to_class[predicted_index]
        confidence = prediction[0][predicted_index]

    # Traduction
    if langue == "Français":
        label_final = traductions.get(predicted_class, {}).get("fr", predicted_class)
    elif langue == "Arabe":
        label_final = traductions.get(predicted_class, {}).get("ar", predicted_class)
    else:
        label_final = predicted_class

    # Affichage
    st.write("### 🔍 Résultat de la prédiction")
    st.success(f"✅ Culture détectée : **{label_final}**")
    st.progress(float(confidence))
    st.write(f"📊 Confiance du modèle : `{confidence*100:.2f}%`")

    # Historique
    st.session_state.historique.append({
        "label": label_final,
        "confiance": f"{confidence*100:.2f}%",
        "classe_orig": predicted_class
    })

    # Correction possible 
    if "correction_active" not in st.session_state:
        st.session_state.correction_active = False

    if st.button("❌ Mauvaise prédiction ? Cliquez ici pour corriger"):
        st.session_state.correction_active = True

    if st.session_state.correction_active:
        st.write("### ✏️ Correction de l'étiquette")
        correct_label = st.selectbox("👉 Choisissez la bonne classe :", list(class_indices.keys()))
        if st.button("✅ Sauvegarder la correction"):
            BASE_DIR = pathlib.Path().resolve()
            corrections_dir = os.path.join(BASE_DIR, "Documents", "dataset", "corrections")
            correction_path = os.path.join(corrections_dir, correct_label)
            os.makedirs(correction_path, exist_ok=True)

            filename = f"{uuid.uuid4()}.jpg"
            image.save(os.path.join(correction_path, filename))

            st.success("📥 Image sauvegardée pour un futur réentrainement du modèle.")
            st.session_state.correction_active = False
            
                        # Vérifier le nombre total d'images dans le dossier corrections
            total_images = 0
            for root, dirs, files in os.walk(corrections_dir):
                total_images += len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

            if total_images >= 50:
                st.info("⏳ 50 images de correction atteintes. Le réentraînement peut maintenant être lancé.")

                # Écrire un signal pour déclencher un réentraînement (flag)
                with open("ready_to_retrain.txt", "w") as f:
                    f.write("true")
    # Affichage de l'historique
    if len(st.session_state.historique) > 0:
        divider()
        st.write("📚 Historique des prédictions")
        for i, h in enumerate(reversed(st.session_state.historique[-5:])):
            st.markdown(f"- **{h['label']}** ({h['confiance']}) – *({h['classe_orig']})*")

else:
    st.info("📂 Importez une image pour commencer la détection.")

divider()
show_footer()
