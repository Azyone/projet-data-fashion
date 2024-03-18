# Fashion Image Classifier API

## Description

Ce projet consiste en un modèle de Machine Learning basé sur TensorFlow et Keras, entraîné pour classifier des images de mode en termes de type d'article et de couleur de base. Le modèle utilise l'architecture Xception pour la classification. Une API Flask est également mise en place pour permettre l'utilisation facile du modèle via des requêtes HTTP.

## Installation

### Prérequis

- Python 3.x
- pip

### Installation des dépendances

Pour installer les dépendances nécessaires, exécutez la commande suivante :
pip install -r requirements.txt

## Utilisation

### Entraînement du Modèle

Pour entraîner le modèle, exécutez :
python train_model.py

### Lancement de l'API

Pour lancer l'API Flask, exécutez :
python api.py


## API Endpoints

`POST /predict`
- `image`: Fichier image à classifier.
- `x-api-key`: Clé API requise pour l'authentification.

## Modèle

Le modèle utilise l'architecture Xception avec une personnalisation pour produire deux sorties : type d'article et couleur de base. Les images sont redimensionnées en 224x224 pixels pour être compatibles avec le modèle.

## Cross-validation

Une validation croisée sur 5 folds est utilisée pour évaluer la performance du modèle.

## Courbes d'Apprentissage

Des courbes d'apprentissage pour la précision et la perte sont générées après l'entraînement pour visualiser la performance du modèle au fil des epochs.

## Sécurité de l'API

L'API utilise une clé API pour l'authentification et un système de limitation de taux pour éviter les abus.

## Auteurs

- Aymeric Fischer
