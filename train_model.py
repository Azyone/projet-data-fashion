import numpy as np
import pandas as pd
import json
import logging
import tensorflow as tf
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, GlobalMaxPool2D, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Configurer le logging pour afficher les erreurs
logging.basicConfig(level=logging.ERROR)

# Vérifier la disponibilité des GPU et configurer TensorFlow pour les utiliser
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
else:
    print("Aucun GPU trouvé, le modèle utilisera le CPU.")

# Charger les données et les préparer
def load_and_prepare_data(filepath):
    try:
        train_df = pd.read_csv(filepath, on_bad_lines='skip')
        # Réduire la taille du dataset pour accélérer l'entraînement
        train_df = train_df.sample(n=2000, random_state=42) 
        train_df["id"] = train_df["id"].astype(str).apply(lambda x: x + ".jpg")
        return train_df
    except Exception as e:
        logging.error(f"Erreur lors du chargement des données : {e}")
        raise

# Nettoyer les données
def clean_data(train_df):
    try:
        train_df.dropna(inplace=True)
        return train_df
    except Exception as e:
        logging.error(f"Erreur lors du nettoyage des données : {e}")
        raise

# Encoder les étiquettes de texte en nombres
def encode_labels(train_df):
    try:
        label_encoder_article = LabelEncoder()
        label_encoder_colour = LabelEncoder()
        train_df['articleType'] = label_encoder_article.fit_transform(train_df['articleType'])
        train_df['baseColour'] = label_encoder_colour.fit_transform(train_df['baseColour'])
        return train_df, label_encoder_article, label_encoder_colour
    except Exception as e:
        logging.error(f"Erreur lors de l'encodage des étiquettes : {e}")
        raise

# Sauvegarder les encodeurs d'étiquettes pour une utilisation future
def save_label_encoders(label_encoder_article, label_encoder_colour, filepath):
    try:
        article_type_classes = list(label_encoder_article.classes_)
        base_colour_classes = list(label_encoder_colour.classes_)
        with open(filepath + 'article_type_classes.json', 'w') as f:
            json.dump(article_type_classes, f)
        with open(filepath + 'base_colour_classes.json', 'w') as f:
            json.dump(base_colour_classes, f)
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde des encodeurs d'étiquettes : {e}")
        raise

# Créer des générateurs de données pour l'entraînement et la validation
def create_generators(train_df, batch_size, target_size, label_encoder_article, label_encoder_colour):
    try:
        datagen = ImageDataGenerator(
            rescale=1./255.,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )
        def generate_data_generator_for_two_outputs(gen, df, subset):
            genX1 = gen.flow_from_dataframe(
                df,
                directory="assets/images",
                x_col="id",
                y_col="articleType",
                subset=subset,
                batch_size=batch_size,
                seed=42,
                shuffle=True,
                class_mode="raw",
                target_size=target_size
            )
            genX2 = gen.flow_from_dataframe(
                df,
                directory="assets/images",
                x_col="id",
                y_col="baseColour",
                subset=subset,
                batch_size=batch_size,
                seed=42,
                shuffle=True,
                class_mode="raw",
                target_size=target_size
            )
            while True:
                X1i = genX1.next()
                X2i = genX2.next()
                yield X1i[0], [to_categorical(X1i[1], num_classes=len(label_encoder_article.classes_)), to_categorical(X2i[1], num_classes=len(label_encoder_colour.classes_))]
        return {
            "training": generate_data_generator_for_two_outputs(datagen, train_df, "training"),
            "validation": generate_data_generator_for_two_outputs(datagen, train_df, "validation")
        }
    except Exception as e:
        logging.error(f"Erreur lors de la création des générateurs : {e}")
        raise

# Construire et compiler le modèle
def build_and_compile_model(input_shape, num_article_types, num_base_colours):
    try:
        input_tensor = Input(shape=input_shape)
        base_model = Xception(include_top=False, input_tensor=input_tensor, weights='imagenet')
        base_model.trainable = False
        x = GlobalMaxPool2D()(base_model.output)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        article_output = Dense(num_article_types, activation='softmax', name='article_output')(x)
        colour_output = Dense(num_base_colours, activation='softmax', name='colour_output')(x)
        model = Model(inputs=base_model.input, outputs=[article_output, colour_output])
        optimizer_adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.99)
        model.compile(optimizer=optimizer_adam,
                      loss={'article_output': 'categorical_crossentropy', 'colour_output': 'categorical_crossentropy'},
                      metrics={'article_output': 'accuracy', 'colour_output': 'accuracy'})
        return model
    except Exception as e:
        logging.error(f"Erreur lors de la construction et de la compilation du modèle : {e}")
        raise

# Entraîner le modèle
def train_model(model, train_generator, val_generator, batch_size, steps_per_epoch, validation_steps, callbacks):
    try:
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=10,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        return history
    except Exception as e:
        logging.error(f"Erreur lors de l'entraînement du modèle : {e}")
        raise

# Effectuer une validation croisée
def cross_validate_model(train_df, label_encoder_article, label_encoder_colour, num_folds=3, batch_size=16):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_results = []
    fold_number = 0
    for train_index, val_index in kf.split(train_df):
        fold_number += 1
        train_fold_df = train_df.iloc[train_index]
        val_fold_df = train_df.iloc[val_index]
        train_size = len(train_fold_df)
        val_size = len(val_fold_df)
        steps_per_epoch = train_size // batch_size
        validation_steps = val_size // batch_size
        generators = create_generators(train_fold_df, batch_size, (224, 224), label_encoder_article, label_encoder_colour)
        train_generator = generators["training"]
        val_generator = generators["validation"]
        model = build_and_compile_model((224, 224, 3), len(label_encoder_article.classes_), len(label_encoder_colour.classes_))
        callbacks = [
            EarlyStopping(patience=6, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', verbose=2, factor=0.5, min_lr=0.00001),
            ModelCheckpoint('assets/model_checkpoint_fold_{}.h5'.format(fold_number), monitor='val_loss', verbose=1, save_best_only=True)
        ]
        history = train_model(model, train_generator, val_generator, batch_size, steps_per_epoch, validation_steps, callbacks)
        fold_results.append(history.history)
    model.save('assets/final_model.h5')
    print("Modèle sauvegardé dans 'assets/final_model.h5'")
    return fold_results

# Tracer les courbes d'apprentissage
def plot_learning_curves(fold_results):
    mean_article_accuracy = []
    mean_colour_accuracy = []
    mean_val_article_accuracy = []
    mean_val_colour_accuracy = []
    mean_loss = []
    mean_val_loss = []
    num_epochs = len(fold_results[0]['article_output_accuracy'])
    
    for epoch in range(num_epochs):
        epoch_article_accuracy = np.mean([fold['article_output_accuracy'][epoch] for fold in fold_results])
        epoch_colour_accuracy = np.mean([fold['colour_output_accuracy'][epoch] for fold in fold_results])
        epoch_val_article_accuracy = np.mean([fold['val_article_output_accuracy'][epoch] for fold in fold_results])
        epoch_val_colour_accuracy = np.mean([fold['val_colour_output_accuracy'][epoch] for fold in fold_results])
        epoch_loss = np.mean([fold['loss'][epoch] for fold in fold_results])
        epoch_val_loss = np.mean([fold['val_loss'][epoch] for fold in fold_results])
        
        mean_article_accuracy.append(epoch_article_accuracy)
        mean_colour_accuracy.append(epoch_colour_accuracy)
        mean_val_article_accuracy.append(epoch_val_article_accuracy)
        mean_val_colour_accuracy.append(epoch_val_colour_accuracy)
        mean_loss.append(epoch_loss)
        mean_val_loss.append(epoch_val_loss)

    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(range(1, num_epochs + 1), mean_article_accuracy, label='Article Training Accuracy')
    plt.plot(range(1, num_epochs + 1), mean_val_article_accuracy, label='Article Validation Accuracy')
    plt.plot(range(1, num_epochs + 1), mean_colour_accuracy, label='Colour Training Accuracy')
    plt.plot(range(1, num_epochs + 1), mean_val_colour_accuracy, label='Colour Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(range(1, num_epochs + 1), mean_loss, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), mean_val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        filepath = 'assets/fashion.csv'
        train_df = load_and_prepare_data(filepath)
        train_df = clean_data(train_df)
        train_df, label_encoder_article, label_encoder_colour = encode_labels(train_df)
        save_label_encoders(label_encoder_article, label_encoder_colour, 'assets/')
        batch_size = 16
        target_size = (224, 224)
        num_article_types = len(label_encoder_article.classes_)
        num_base_colours = len(label_encoder_colour.classes_)
        fold_results = cross_validate_model(train_df, label_encoder_article, label_encoder_colour, num_folds=5)
        plot_learning_curves(fold_results)
        for i, result in enumerate(fold_results):
            print(f"Résultats du Fold {i+1}: {result}")
    except Exception as e:
        logging.error(f"Erreur dans le script principal : {e}")