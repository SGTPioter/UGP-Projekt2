from tensorflow import keras
from tensorflow.keras import layers

from data import CLASS_NAMES, IMAGE_SHAPE


def build_cnn(augment: bool, seed: int) -> keras.Model:
    inputs = keras.Input(shape=IMAGE_SHAPE, dtype="uint8")
    x = layers.Rescaling(1 / 255.0)(inputs)
    if augment:
        x = layers.RandomFlip("horizontal", seed=seed)(x)
        x = layers.RandomTranslation(0.1, 0.1, fill_mode="reflect", seed=seed + 1)(x)

    for block, (filters, dropout) in enumerate(((32, 0.1), (64, 0.2), (128, 0.3))):
        for convolution in range(2):
            x = layers.Conv2D(
                filters, 3, padding="same", use_bias=False,
                kernel_initializer=keras.initializers.HeNormal(seed=seed + 2 * block + convolution),
            )(x)
            x = layers.BatchNormalization(momentum=0.9)(x)
            x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(dropout, seed=seed + block)(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(
        len(CLASS_NAMES), kernel_initializer=keras.initializers.GlorotUniform(seed=seed)
    )(x)
    return keras.Model(inputs, outputs, name="cnn")


class MobileNetTransfer:
    IMAGE_SIZE = 96
    FINE_TUNE_FROM = "block_13_expand"

    def __init__(self) -> None:
        self.backbone = keras.applications.MobileNetV2(
            input_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3),
            include_top=False,
            weights="imagenet",
            pooling="avg",
        )
        self.backbone.trainable = False
        inputs = keras.Input(shape=IMAGE_SHAPE, dtype="uint8")
        x = layers.Resizing(self.IMAGE_SIZE, self.IMAGE_SIZE)(inputs)
        x = layers.Rescaling(1 / 127.5, offset=-1)(x)
        features = self.backbone(x, training=False)
        self.extractor = keras.Model(inputs, features, name="feature_extractor")
        self.head = keras.Sequential(
            [keras.Input(shape=self.extractor.output_shape[1:]),
             layers.Dense(len(CLASS_NAMES), name="logits")],
            name="classification_head",
        )
        self.model = keras.Model(inputs, self.head(features), name="mobilenetv2")

    def unfreeze_last_blocks(self) -> None:
        names = [layer.name for layer in self.backbone.layers]
        first = names.index(self.FINE_TUNE_FROM)
        self.backbone.trainable = True
        for index, layer in enumerate(self.backbone.layers):
            layer.trainable = index >= first and not isinstance(layer, layers.BatchNormalization)
