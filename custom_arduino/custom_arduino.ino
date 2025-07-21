#include <Servo.h>

// Servo setup
const int NUM_BINS = 4;
const int SERVO_PINS[NUM_BINS] = {9, 10, 11, 12}; // Update as needed
Servo servos[NUM_BINS];
const int OPEN_ANGLE = 90;
const int CLOSE_ANGLE = 0;

// Ultrasonic sensor setup
const int TRIG_PIN = 7;
const int ECHO_PIN = 8;

void setup() {
  Serial.begin(57600);

  // Attach servos and set to closed
  for (int i = 0; i < NUM_BINS; i++) {
    servos[i].attach(SERVO_PINS[i]);
    servos[i].write(CLOSE_ANGLE);
  }

  // Ultrasonic sensor pins
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
}

void loop() {
  // --- Handle serial commands for servos ---
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.startsWith("OPEN:")) {
      int bin = cmd.substring(5).toInt();
      if (bin >= 0 && bin < NUM_BINS) {
        servos[bin].write(OPEN_ANGLE);
        Serial.print("OPENED:");
        Serial.println(bin);
      }
    } else if (cmd.startsWith("CLOSE:")) {
      int bin = cmd.substring(6).toInt();
      if (bin >= 0 && bin < NUM_BINS) {
        servos[bin].write(CLOSE_ANGLE);
        Serial.print("CLOSED:");
        Serial.println(bin);
      }
    }
  }

  // --- Read ultrasonic sensor and send distance ---
  long duration, distance;
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  duration = pulseIn(ECHO_PIN, HIGH, 30000); // 30ms timeout
  distance = duration * 0.034 / 2;

  // Only send if distance is reasonable (e.g., < 20cm)
  if (distance > 0 && distance < 200) {
    Serial.print("DIST:");
    Serial.println(distance);
  }

  delay(100); // 10Hz update rate
}