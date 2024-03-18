from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from io import BytesIO
import json
import functools

# Création d'une instance de l'application Flask
app = Flask(__name__)

# Clé API pour l'authentification
API_KEY = "8fd40b66-7bfe-4b34-a83d-a8f6200c790e"

# Configuration du limiteur de taux pour limiter le nombre de requêtes
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["5 per minute"]
)

# Chargement du modèle Keras précédemment entraîné
model = load_model('assets/final_model.h5')

# Chargement des classes de types d'articles et de couleurs de base à partir de fichiers JSON
with open('assets/article_type_classes.json') as f:
    article_type_classes = json.load(f)

with open('assets/base_colour_classes.json') as f:
    base_colour_classes = json.load(f)

# Décoration pour exiger une clé API pour accéder à certaines routes
def require_apikey(view_function):
    @functools.wraps(view_function)
    def decorated_function(*args, **kwargs):
        if request.headers.get('x-api-key') and request.headers.get('x-api-key') == API_KEY:
            return view_function(*args, **kwargs)
        else:
            return jsonify({"error": "Clé API manquante ou invalide"}), 401
    return decorated_function

# Route pour la prédiction d'images
@app.route('/predict', methods=['POST'])
@limiter.limit("5 per minute")  # Applique la limitation de taux
@require_apikey  # Nécessite une clé API
def predict():
    try:
        # Vérifie si une image a été envoyée dans la requête
        if 'image' not in request.files:
            return jsonify({"error": "Aucune image fournie"}), 400

        # Récupération de l'image envoyée
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Aucun fichier sélectionné"}), 400

        # Conversion de l'image pour le modèle
        img_bytes = BytesIO(file.read())
        img = Image.open(img_bytes).convert('RGB')
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.

        # Prédiction à l'aide du modèle
        article_type_pred, base_colour_pred = model.predict(img_array)

        # Obtention des labels correspondants
        article_type_label = article_type_classes[article_type_pred.argmax(axis=-1)[0]]
        base_colour_label = base_colour_classes[base_colour_pred.argmax(axis=-1)[0]]

        response = {'article_type': article_type_label, 'base_colour': base_colour_label}
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(response)

# Démarrage de l'application Flask
if __name__ == '__main__':
    app.run(debug=True)
