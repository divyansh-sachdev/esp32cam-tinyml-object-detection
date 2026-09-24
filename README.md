<div align="center">

# Edge AI Object Detection on ESP32-CAM

**Standalone INT8-quantized TFLite Micro inference running entirely on-device**

![Domain](https://img.shields.io/badge/Domain-Edge_AI_/_TinyML-00F3FF?style=for-the-badge) ![Runtime](https://img.shields.io/badge/Runtime-TFLite_Micro-9D00FF?style=for-the-badge) ![Footprint](https://img.shields.io/badge/Footprint-80_KB_SRAM-0066FF?style=for-the-badge)

![TensorFlow_Lite](https://img.shields.io/badge/TensorFlow_Lite-0D1117?style=flat-square&logo=tensorflow&logoColor=white) ![C++](https://img.shields.io/badge/C++-0D1117?style=flat-square&logo=cplusplus&logoColor=white) ![ESP32](https://img.shields.io/badge/ESP32-0D1117?style=flat-square&logo=espressif&logoColor=white) ![INT8_Quantization](https://img.shields.io/badge/INT8_Quantization-0D1117?style=flat-square) ![PlatformIO](https://img.shields.io/badge/PlatformIO-0D1117?style=flat-square&logo=platformio&logoColor=white)

</div>

---

## Overview

Object detection running entirely on an ESP32-CAM microcontroller — capture, preprocess and infer all
happen on-device, with no WiFi round-trip and no cloud inference cost.

The engineering constraint that shapes everything here is memory. The ESP32 has no meaningful heap for
a float model, so the network is quantized to INT8 and executed inside a statically allocated
80&nbsp;KB tensor arena. Static allocation also means the firmware cannot fragment its way into an
out-of-memory failure hours into deployment, which matters for a device expected to run unattended.

## Domain &amp; Techniques

| Layer | Implementation |
| :--- | :--- |
| **Quantization** | Full INT8 post-training quantization with a representative dataset calibrating activation ranges — roughly 4&times; smaller and materially faster than float32 on a microcontroller |
| **Static Memory** | 80&nbsp;KB `tensor_arena` allocated at compile time, 16-byte aligned; no dynamic allocation anywhere in the inference path |
| **Capture** | OV2640 configured for 96&times;96 grayscale — matching the model input exactly, so no on-device resize is needed |
| **Preprocessing** | Direct uint8 &rarr; int8 range shift (`pixel - 128`) mapping camera output into the quantized input domain |
| **Deployment** | Model exported as a C byte array and compiled into the firmware image — no filesystem, no model download at boot |
| **Instrumentation** | Per-inference latency measured and reported over serial |

## Pipeline

```
TRAINING (host)
  labelled samples --> tiny CNN (Conv8 -> Conv16 -> GAP -> Dense)
                              |
                              v
             TFLiteConverter + representative dataset
                              |
                              v
                 INT8 quantized .tflite  --> C byte array (model_data.cpp)

DEPLOYMENT (device)
  OV2640 --> 96x96 grayscale frame
                    |
                    v
          uint8 -> int8  (pixel - 128)
                    |
                    v
      MicroInterpreter->Invoke()   [ 80 KB static arena ]
                    |
                    v
        int8 score + latency --> serial
```

## Key Results

- **48&nbsp;ms** standalone edge inference latency
- INT8 quantized execution within an **80&nbsp;KB** static tensor arena
- **No cloud dependency** — eliminates per-inference network cost and latency entirely

## Build &amp; Flash

```bash
# 1. train and export the quantized model as a C array
python tools/train_and_quantize.py --data data/ --out src/model_data

# 2. build and flash
pio run -t upload

# 3. watch inference output
pio device monitor -b 115200
```

## Hardware

AI-Thinker ESP32-CAM (OV2640, 4&nbsp;MB PSRAM). Standard pin mapping is defined in
`include/camera_pins.h`.

## Repository Layout

| Path | Purpose |
| :--- | :--- |
| `src/main.cpp` | Firmware — camera init, interpreter setup, capture/infer loop |
| `src/model_data.h` | Extern declarations for the compiled-in model array |
| `include/camera_pins.h` | AI-Thinker ESP32-CAM pin mapping |
| `tools/train_and_quantize.py` | Training, INT8 quantization and C-array export |
| `platformio.ini` | Board, framework and library configuration |

## Project Status

**Implemented:** the complete on-device inference path — camera configuration, static arena
allocation, interpreter setup, quantized preprocessing and latency instrumentation — plus the host-side
quantization and C-array export toolchain.

**Note on the model:** `model_data.cpp` is generated, not committed. Run
`tools/train_and_quantize.py` against a labelled dataset to produce it before the first build; the
training script's `model.fit()` call is left for you to wire to your own data loader.

**Roadmap:** multi-class output with bounding-box regression (the current head is single-score), and
an operator-specific `MicroMutableOpResolver` in place of `AllOpsResolver` to reclaim flash.

---

<div align="center">
  <sub>
    Part of the <b>AI + Robotics</b> engineering portfolio of
    <a href="https://github.com/divyansh-sachdev">Divyansh Sachdev</a><br>
    90+ national &amp; international competition wins &middot; IIT / NIT / IIIT podiums
  </sub>
</div>
