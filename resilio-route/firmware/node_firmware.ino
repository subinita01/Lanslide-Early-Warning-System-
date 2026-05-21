/**
 * firmware/node_firmware.ino
 * ──────────────────────────
 * ESP32 sensor node firmware for Resilio-Route.
 * Reads all sensors every 5 minutes and transmits via LoRa mesh.
 *
 * Hardware:
 *   - ESP32 or SAMD21 MCU
 *   - MEMS tilt sensor (I2C, e.g. MPU-6050)
 *   - Capacitive soil moisture probes × 6 (ADC pins)
 *   - Piezometer (analog, 0–3.3V → 0–100 kPa)
 *   - Tipping bucket rain gauge (interrupt pin)
 *   - LoRa SX1278 (SPI)
 *   - 5W solar + 18Ah LiPo
 *
 * Dependencies (PlatformIO lib_deps):
 *   - sandeepmistry/LoRa @ ^0.8.0
 *   - adafruit/Adafruit MPU6050 @ ^2.2.4
 *   - bblanchon/ArduinoJson @ ^6.21.3
 */

#include <Arduino.h>
#include <SPI.h>
#include <LoRa.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <ArduinoJson.h>

// ── Node config ────────────────────────────────────────────────────────────
#define NODE_ID          7          // Unique per node: 1..255
#define HIGHWAY          "NH-306"
#define KM_MARKER        132.0f

// ── Pin assignments ────────────────────────────────────────────────────────
#define LORA_SS          5
#define LORA_RST         14
#define LORA_DIO0        2
#define LORA_FREQ        865E6      // 865 MHz (India ISM band)

#define MOISTURE_PINS    {34, 35, 32, 33, 25, 26}   // ADC1 channels
#define PIEZOMETER_PIN   39
#define RAIN_GAUGE_PIN   27         // interrupt pin
#define BATTERY_PIN      36

// ── Timing ─────────────────────────────────────────────────────────────────
#define SAMPLE_INTERVAL  300000     // 5 minutes in ms
#define LORA_TX_POWER    17         // dBm

// ── Globals ────────────────────────────────────────────────────────────────
Adafruit_MPU6050 mpu;

volatile uint16_t rain_tips  = 0;  // each tip = 0.2 mm rainfall
float             tilt_prev  = 0.0f;
unsigned long     last_sample = 0;


// ── Sensor reads ────────────────────────────────────────────────────────────

float read_tilt_deg() {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    // Convert accelerometer to slope angle
    float tilt = atan2(a.acceleration.y,
                       sqrt(a.acceleration.x * a.acceleration.x +
                            a.acceleration.z * a.acceleration.z));
    return tilt * 180.0f / PI;
}

float read_moisture(uint8_t pin) {
    // Capacitive probe: 0V = wet (100%), 3.3V = dry (0%)
    int raw = analogRead(pin);
    float pct = map(raw, 4095, 1200, 0, 100);
    return constrain(pct, 0.0f, 100.0f);
}

float read_pore_pressure() {
    // Piezometer: 0V = 0 kPa, 3.3V = 100 kPa
    int raw = analogRead(PIEZOMETER_PIN);
    return (raw / 4095.0f) * 100.0f;
}

float read_battery_pct() {
    int raw = analogRead(BATTERY_PIN);
    float v = (raw / 4095.0f) * 3.3f * 2.0f;  // voltage divider
    return constrain(map(v * 100, 300, 420, 0, 100), 0.0f, 100.0f);
}

void IRAM_ATTR rain_isr() {
    rain_tips++;   // incremented on each tipping-bucket tip
}


// ── LoRa transmission ───────────────────────────────────────────────────────

void transmit_packet(JsonDocument& doc) {
    String payload;
    serializeJson(doc, payload);

    LoRa.beginPacket();
    LoRa.print(payload);
    LoRa.endPacket();

    Serial.print("[TX] ");
    Serial.println(payload);
}


// ── Setup ───────────────────────────────────────────────────────────────────

void setup() {
    Serial.begin(115200);
    Wire.begin();

    // ── MPU-6050 tilt sensor ──────────────────────────────────────────────
    if (!mpu.begin()) {
        Serial.println("[ERROR] MPU-6050 not found. Check wiring.");
        while (1) delay(10);
    }
    mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
    mpu.setFilterBandwidth(MPU6050_BAND_5_HZ);
    Serial.println("[OK] MPU-6050 initialised");

    // ── LoRa ──────────────────────────────────────────────────────────────
    LoRa.setPins(LORA_SS, LORA_RST, LORA_DIO0);
    if (!LoRa.begin(LORA_FREQ)) {
        Serial.println("[ERROR] LoRa init failed. Check wiring.");
        while (1) delay(10);
    }
    LoRa.setTxPower(LORA_TX_POWER);
    LoRa.setSpreadingFactor(10);     // SF10: range vs speed tradeoff
    LoRa.setSignalBandwidth(125E3);  // 125 kHz
    LoRa.setCodingRate4(5);
    Serial.println("[OK] LoRa initialised at 865 MHz");

    // ── Rain gauge interrupt ──────────────────────────────────────────────
    pinMode(RAIN_GAUGE_PIN, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(RAIN_GAUGE_PIN), rain_isr, FALLING);

    Serial.printf("[OK] Node NM-%02d ready on %s KM %.1f\n",
                  NODE_ID, HIGHWAY, KM_MARKER);
}


// ── Main loop ───────────────────────────────────────────────────────────────

void loop() {
    unsigned long now = millis();

    if (now - last_sample >= SAMPLE_INTERVAL) {
        last_sample = now;

        // ── Read all sensors ──────────────────────────────────────────────
        float tilt = read_tilt_deg();
        float tilt_rate = (tilt - tilt_prev) / (SAMPLE_INTERVAL / 60000.0f); // deg/min
        tilt_prev = tilt;

        uint8_t moisture_pins[] = MOISTURE_PINS;
        float moisture[6];
        for (int i = 0; i < 6; i++) {
            moisture[i] = read_moisture(moisture_pins[i]);
        }

        float pore     = read_pore_pressure();
        float battery  = read_battery_pct();
        int   rssi     = LoRa.packetRssi();

        // Rainfall since last reading (each tip = 0.2 mm)
        float rainfall_mm = rain_tips * 0.2f;
        rain_tips = 0;  // reset counter

        // ── Build JSON packet ─────────────────────────────────────────────
        StaticJsonDocument<512> doc;
        doc["node_id"]       = NODE_ID;
        doc["timestamp"]     = millis() / 1000;  // replace with RTC in production
        doc["tilt_deg"]      = roundf(tilt * 1000) / 1000.0f;
        doc["tilt_rate"]     = roundf(tilt_rate * 1000) / 1000.0f;

        JsonArray m = doc.createNestedArray("moisture");
        for (int i = 0; i < 6; i++) m.add(roundf(moisture[i] * 10) / 10.0f);

        doc["pore_pressure"] = roundf(pore * 10) / 10.0f;
        doc["rainfall_1hr"]  = roundf(rainfall_mm * 10) / 10.0f;
        doc["battery_pct"]   = (int)battery;
        doc["rssi"]          = rssi;

        // ── Transmit ──────────────────────────────────────────────────────
        transmit_packet(doc);

        // ── Local safety check (fallback if LoRa gateway is down) ────────
        // Simple threshold: if both tilt > 0.5 deg AND moisture > 90%,
        // pulse the local buzzer as a standalone warning.
        bool local_danger = (tilt > 0.5f) &&
                            (moisture[0] > 90.0f) &&
                            (pore > 3.0f);
        if (local_danger) {
            Serial.println("[WARN] LOCAL DANGER THRESHOLD EXCEEDED");
            // In production: trigger local buzzer or relay
            // digitalWrite(BUZZER_PIN, HIGH);
        }
    }

    delay(100);
}
