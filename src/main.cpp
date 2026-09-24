#include <Arduino.h>
#include "esp_camera.h"
#include "camera_pins.h"

#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "model_data.h"

namespace {
constexpr int kTensorArenaSize = 80 * 1024;  // fits within the ESP32-CAM's available SRAM
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
}  // namespace

static bool initCamera() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_GRAYSCALE;
  config.frame_size = FRAMESIZE_96X96;
  config.fb_count = 1;

  return esp_camera_init(&config) == ESP_OK;
}

static bool initModel() {
  model = tflite::GetModel(g_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch");
    return false;
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return false;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
  return true;
}

static void copyFrameToInput(camera_fb_t* fb) {
  // Frame is already 96x96 grayscale; shift into the model's INT8 input range.
  for (size_t i = 0; i < fb->len && i < input->bytes; i++) {
    input->data.int8[i] = static_cast<int8_t>(fb->buf[i] - 128);
  }
}

void setup() {
  Serial.begin(115200);
  if (!initCamera()) {
    Serial.println("Camera init failed");
    return;
  }
  if (!initModel()) {
    Serial.println("Model init failed");
    return;
  }
  Serial.println("System ready");
}

void loop() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Frame capture failed");
    delay(50);
    return;
  }

  const uint32_t start = millis();
  copyFrameToInput(fb);

  if (interpreter->Invoke() == kTfLiteOk) {
    const uint32_t latency = millis() - start;
    const int8_t objectScore = output->data.int8[0];
    Serial.printf("inference=%ums score=%d\n", latency, objectScore);
  } else {
    Serial.println("Invoke() failed");
  }

  esp_camera_fb_return(fb);
}
