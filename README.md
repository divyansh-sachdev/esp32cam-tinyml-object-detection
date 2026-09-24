# Edge AI Object Detection on ESP32-CAM (TinyML)

Standalone object detection running entirely on an ESP32-CAM, using an INT8-quantized TensorFlow Lite Micro model — no cloud round-trip required.

## Tech Stack
TensorFlow Lite Micro · C++ · ESP32 · INT8 Quantization

## Key Results
- Achieved **48ms** standalone edge inference latency
- INT8 quantized TFLite Micro tensor execution running within **80KB SRAM**
- Eliminates cloud communication costs entirely — fully on-device inference

## Overview
The model is trained and quantized to INT8, then deployed as a TFLite Micro interpreter running directly on the ESP32-CAM's microcontroller. Camera frames are captured, preprocessed and run through the on-device model, with detections available with no network round-trip.
