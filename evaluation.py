import csv
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import matplotlib
import numpy as np
import tensorflow as tf
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from tensorflow import keras

from data import CLASS_NAMES, ImageSplit

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_summary(output: Path, results: list[dict]) -> None:
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


def save_class_examples(train: ImageSplit, output: Path) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(10, 4), layout="constrained")
    for label, (ax, name) in enumerate(zip(axes.flat, CLASS_NAMES)):
        index = np.flatnonzero(train.labels == label)[0]
        ax.imshow(train.images[index], interpolation="nearest")
        ax.set_title(name)
        ax.axis("off")
    fig.savefig(output / "class_examples.png", dpi=150)
    plt.close(fig)


def evaluate(
    name: str,
    model: keras.Model,
    test: ImageSplit,
    history: dict,
    output: Path,
    batch_size: int,
    fit_seconds: float,
    feature_seconds: float = 0.0,
    previous_seconds: float = 0.0,
) -> dict:
    directory = output / name
    directory.mkdir()
    logits = model.predict(test.images, batch_size=batch_size, verbose=0)
    predictions = logits.argmax(axis=1)
    labels = range(len(CLASS_NAMES))
    report = classification_report(
        test.labels, predictions, labels=list(labels), target_names=CLASS_NAMES,
        zero_division=0, output_dict=True,
    )
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    benchmark_size = min(len(test.labels), batch_size)
    result = {
        "variant": name,
        "test_accuracy": report["accuracy"],
        "test_macro_f1": report["macro avg"]["f1-score"],
        "test_loss": float(loss(test.labels, logits)),
        "trainable_params": sum(int(np.prod(weight.shape)) for weight in model.trainable_weights),
        "total_params": model.count_params(),
        "epochs_run": len(history["loss"]),
        "best_epoch": best_epoch,
        "best_val_loss": history["val_loss"][best_epoch - 1],
        "fit_seconds": fit_seconds,
        "feature_extraction_seconds": feature_seconds,
        "total_training_seconds": previous_seconds + feature_seconds + fit_seconds,
        "inference_batch_size": benchmark_size,
        "inference_batch_ms": inference_time_ms(model, test.images[:benchmark_size]),
    }
    result["inference_ms_per_image"] = result["inference_batch_ms"] / benchmark_size
    write_json(directory / "metrics.json", result)
    write_json(directory / "classification_report.json", report)
    write_json(directory / "history.json", history)
    save_curves(history, best_epoch, directory)
    save_confusion_matrix(test.labels, predictions, name, directory)
    model.save(directory / "model.keras", include_optimizer=False)
    print(f"{name} | accuracy {result['test_accuracy']:.3f} | "
          f"macro F1 {result['test_macro_f1']:.3f} | "
          f"łącznie {result['total_training_seconds']:.2f} s", flush=True)
    return result


def save_confusion_matrix(
    true_labels: np.ndarray, predictions: np.ndarray, name: str, directory: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    display = ConfusionMatrixDisplay.from_predictions(
        true_labels, predictions, labels=list(range(len(CLASS_NAMES))),
        display_labels=CLASS_NAMES, normalize="true", colorbar=False,
        cmap="Blues", xticks_rotation=45, values_format=".1%",
        im_kw={"vmin": 0, "vmax": 1}, ax=ax,
    )
    fig.colorbar(
        display.im_, ax=ax, ticks=np.linspace(0, 1, 6),
        format=PercentFormatter(xmax=1, decimals=0),
        label="Odsetek w obrębie klasy rzeczywistej",
    )
    ax.set(xlabel="Klasa przewidziana", ylabel="Klasa rzeczywista", title=name)
    fig.savefig(directory / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def inference_time_ms(model: keras.Model, images: np.ndarray) -> float:
    batch = tf.convert_to_tensor(images)

    @tf.function
    def predict(inputs):
        return model(inputs, training=False)

    for _ in range(3):
        predict(batch).numpy()
    times = []
    for _ in range(15):
        start = perf_counter()
        predict(batch).numpy()
        times.append((perf_counter() - start) * 1000)
    return median(times)


def save_curves(history: dict, best_epoch: int, directory: Path) -> None:
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, key, title in zip(axes, ("loss", "accuracy"), ("Entropia krzyżowa", "Trafność")):
        ax.plot(epochs, history[key], marker=".", label="Trening")
        ax.plot(epochs, history[f"val_{key}"], marker=".", label="Walidacja")
        ax.axvline(best_epoch, color="gray", linestyle=":", label="Wybrana epoka")
        ax.set(xlabel="Epoka", ylabel=title)
        ax.legend()
    axes[1].set_ylim(0, 1)
    fig.savefig(directory / "learning_curves.png", dpi=150)
    plt.close(fig)
