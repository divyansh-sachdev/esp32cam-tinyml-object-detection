"""Trains a tiny object-presence CNN and exports it as an INT8-quantized
TensorFlow Lite Micro model, ready to embed on the ESP32-CAM.

Usage:
    python tools/train_and_quantize.py --data data/ --out src/model_data
"""
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


def build_model(input_shape=(96, 96, 1)):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(8, 3, activation="relu", strides=2),
        tf.keras.layers.Conv2D(16, 3, activation="relu", strides=2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])


def representative_dataset(data_dir: Path):
    def gen():
        for path in list(data_dir.glob("*.npy"))[:100]:
            sample = np.load(path).astype(np.float32) / 255.0
            yield [sample.reshape(1, 96, 96, 1)]
    return gen


def quantize(model: tf.keras.Model, data_dir: Path) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset(data_dir)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def export_c_array(tflite_bytes: bytes, out_path: Path):
    var_name = "g_model_data"
    hex_bytes = ", ".join(f"0x{b:02x}" for b in tflite_bytes)
    contents = (
        '#include "model_data.h"\n\n'
        f"alignas(8) const unsigned char {var_name}[] = {{\n{hex_bytes}\n}};\n"
        f"const int {var_name}_len = {len(tflite_bytes)};\n"
    )
    out_path.with_suffix(".cpp").write_text(contents)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path, help="Directory of labeled .npy samples")
    parser.add_argument("--out", default=Path("src/model_data"), type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    model = build_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # model.fit(train_ds, epochs=args.epochs)  # wire up your labeled dataset here

    tflite_bytes = quantize(model, args.data)
    export_c_array(tflite_bytes, args.out)
    print(f"Wrote {len(tflite_bytes)} bytes to {args.out.with_suffix('.cpp')}")


if __name__ == "__main__":
    main()
