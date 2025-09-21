
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout

def build_model(input_shape=(224,224,3), num_classes=2):
    base = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    for layer in base.layers:
        layer.trainable = False  

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    preds = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=preds)
    return model
