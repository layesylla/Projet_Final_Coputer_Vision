import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

corrections_path = "corrections"
model_path = "model-final.h5"
flag_path = "ready_to_retrain.txt"

if os.path.exists(flag_path):
    print("✅ 50 images détectées, démarrage du réentraînement...")

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

    train_gen = datagen.flow_from_directory(
        corrections_path,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        corrections_path,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        subset='validation'
    )

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(len(train_gen.class_indices), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_gen, validation_data=val_gen, epochs=3)

    model.save(model_path)
    shutil.rmtree(corrections_path)
    os.remove(flag_path)

    print("✅ Nouveau modèle sauvegardé et corrections supprimées.")
else:
    print("⚠️ Moins de 50 corrections. Rien à faire.")
