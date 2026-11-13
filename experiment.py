import platform
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import tensorflow as tf
from tensorflow import keras

from data import CLASS_NAMES, Cifar10Data, ImageSplit
from evaluation import evaluate, save_class_examples, save_summary, write_json
from models import MobileNetTransfer, build_cnn


EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 4


@dataclass(frozen=True)
class ImageClassificationExperiment:
    train_per_class: int
    val_per_class: int
    test_per_class: int
    epochs: int
    fine_tune_epochs: int
    batch_size: int
    seed: int
    output: Path

    def run(self) -> None:
        keras.utils.set_random_seed(self.seed)
        data = Cifar10Data.load(
            self.train_per_class, self.val_per_class, self.test_per_class, self.seed,
        )
        directory = self.output / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        directory.mkdir(parents=True)
        write_json(directory / "settings.json", {
            **asdict(self),
            "output": str(self.output),
            "classes": CLASS_NAMES,
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "platform": platform.platform(),
            "devices": [device.name for device in tf.config.list_physical_devices()],
            "cnn_learning_rate": 1e-3,
            "head_learning_rate": 1e-3,
            "fine_tune_learning_rate": 1e-4,
            "fine_tune_from": MobileNetTransfer.FINE_TUNE_FROM,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "reduce_lr_patience": REDUCE_LR_PATIENCE,
        })
        print(f"Wyniki: {directory.resolve()}", flush=True)
        save_class_examples(data.train, directory)

        results = []
        for variant, augment in (("cnn", False), ("cnn_augmented", True)):
            keras.backend.clear_session()
            keras.utils.set_random_seed(self.seed)
            model = build_cnn(augment, self.seed)
            history, elapsed = self.fit(model, data.train, data.validation, self.epochs, 1e-3)
            results.append(evaluate(
                variant, model, data.test, history, directory, self.batch_size, elapsed,
            ))
            save_summary(directory, results)
            del model

        keras.backend.clear_session()
        keras.utils.set_random_seed(self.seed)
        transfer = MobileNetTransfer()
        print("Ekstrakcja cech obrazów treningowych i walidacyjnych MobileNetV2", flush=True)
        start = perf_counter()
        features_train = transfer.extractor.predict(
            data.train.images, batch_size=self.batch_size, verbose=1,
        )
        features_val = transfer.extractor.predict(
            data.validation.images, batch_size=self.batch_size, verbose=1,
        )
        feature_seconds = perf_counter() - start
        history, elapsed = self.fit(
            transfer.head, ImageSplit(features_train, data.train.labels),
            ImageSplit(features_val, data.validation.labels), self.epochs, 1e-3,
        )
        results.append(evaluate(
            "feature_extraction", transfer.model, data.test, history, directory,
            self.batch_size, elapsed, feature_seconds=feature_seconds,
        ))
        save_summary(directory, results)
        del features_train, features_val

        transfer.unfreeze_last_blocks()
        history, fine_tune_seconds = self.fit(
            transfer.model, data.train, data.validation, self.fine_tune_epochs, 1e-4,
        )
        results.append(evaluate(
            "fine_tuning", transfer.model, data.test, history, directory,
            self.batch_size, fine_tune_seconds, previous_seconds=elapsed + feature_seconds,
        ))
        save_summary(directory, results)

    def fit(
        self, model: keras.Model, train: ImageSplit, validation: ImageSplit,
        epochs: int, learning_rate: float,
    ) -> tuple[dict, float]:
        optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4)
        optimizer.exclude_from_weight_decay(var_names=["bias", "beta", "gamma"])
        model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        train_dataset = (
            tf.data.Dataset.from_tensor_slices((train.images, train.labels))
            .shuffle(len(train.labels), seed=self.seed, reshuffle_each_iteration=True)
            .batch(self.batch_size)
        )
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (validation.images, validation.labels),
        ).batch(self.batch_size)
        start = perf_counter()
        history = model.fit(
            train_dataset, validation_data=val_dataset, epochs=epochs, shuffle=False, verbose=1,
            callbacks=[
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=REDUCE_LR_PATIENCE,
                    min_lr=learning_rate / 100,
                ),
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
                    restore_best_weights=True,
                ),
            ],
        ).history
        elapsed = perf_counter() - start
        return {key: np.asarray(values, dtype=float).tolist() for key, values in history.items()}, elapsed
