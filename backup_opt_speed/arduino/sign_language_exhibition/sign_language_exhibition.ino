/*
  =============================================================================
  AI-Based Sign Language Recognition & Voice Communication System
  Exhibition Display & Hardware Actuation Firmware for Arduino UNO
  =============================================================================

  Hardware Components:
    1. Arduino UNO Rev3
    2. 16x2 Character LCD with PCF8574 I2C Backpack (Default Address: 0x27 or 0x3F)
    3. Active 5V Buzzer (Digital Pin 8)
    4. Emergency Red LED (Digital Pin 9 - via 220 Ohm resistor)
    5. Status Green LED (Digital Pin 10 - via 220 Ohm resistor)

  I2C Pin Connections (Arduino UNO):
    - LCD SDA  -> Arduino Analog Pin A4 (or dedicated SDA pin)
    - LCD SCL  -> Arduino Analog Pin A5 (or dedicated SCL pin)
    - LCD VCC  -> 5V
    - LCD GND  -> GND

  Serial Protocol:
    - Baud Rate: 9600 bps
    - Command Format: "G:<GESTURE_NAME>\n"
    - Examples:
        "G:HELLO\n"
        "G:YES\n"
        "G:NO\n"
        "G:HELP\n"
        "G:STOP\n"
  =============================================================================
*/

#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// --- PIN DEFINITIONS ---
const int PIN_BUZZER    = 8;   // Active Buzzer
const int PIN_RED_LED   = 9;   // Emergency Alert Red LED
const int PIN_GREEN_LED = 10;  // Normal Status Green LED

// --- I2C LCD CONFIGURATION ---
// Most common I2C backpacks use 0x27. If text doesn't show, change to 0x3F.
LiquidCrystal_I2C lcd(0x27, 16, 2);

// --- SERIAL BUFFER CONFIGURATION ---
const int SERIAL_BUFFER_SIZE = 32;
char serialBuffer[SERIAL_BUFFER_SIZE];
int bufferIndex = 0;

// --- STATE MANAGEMENT & NON-BLOCKING TIMERS ---
enum SystemState {
  STATE_STANDBY,
  STATE_NORMAL_SIGN,
  STATE_EMERGENCY_ALERT
};

SystemState currentState = STATE_STANDBY;
unsigned long stateStartTime = 0;
unsigned long lastBlinkTime = 0;
bool emergencyBlinkState = false;
int emergencyBeepCount = 0;
char lastRecognizedGesture[16] = "";

// Timeouts (milliseconds)
const unsigned long DISPLAY_HOLD_TIME = 6000;    // Return to standby after 6 sec of no signs
const unsigned long GREEN_LED_HOLD_TIME = 1500;  // Green LED light duration
const unsigned long EMERGENCY_BLINK_INTERVAL = 150; // Rapid alert strobing

// --- FUNCTION PROTOTYPES ---
void handleSerialInput();
void processCommand(const char* cmd);
void triggerNormalSign(const char* signName);
void triggerEmergencyAlert(const char* signName);
void updateHardwareOutputs();
void displayStandbyScreen();
void startupSelfTest();

void setup() {
  // Initialize Serial Interface at 115200 baud for fast responsive communication
  Serial.begin(115200);
  while (!Serial) {
    ; // Wait for serial port to connect (native USB boards)
  }

  // Initialize GPIO Pins
  pinMode(PIN_BUZZER, OUTPUT);
  pinMode(PIN_RED_LED, OUTPUT);
  pinMode(PIN_GREEN_LED, OUTPUT);

  digitalWrite(PIN_BUZZER, LOW);
  digitalWrite(PIN_RED_LED, LOW);
  digitalWrite(PIN_GREEN_LED, LOW);

  // Initialize I2C LCD Display
  Wire.begin();
  lcd.init();
  lcd.backlight();

  // Run Exhibition Boot Self-Test
  startupSelfTest();

  // Ready for live operation
  displayStandbyScreen();
  Serial.println("ARDUINO_READY");
}

void loop() {
  // 1. Check for incoming serial commands from Python
  handleSerialInput();

  // 2. Update non-blocking LEDs, Buzzer, and Screen Timeouts
  updateHardwareOutputs();
}

/**
 * Reads incoming serial characters line by line.
 */
void handleSerialInput() {
  while (Serial.available() > 0) {
    char inChar = (char)Serial.read();

    if (inChar == '\n' || inChar == '\r') {
      if (bufferIndex > 0) {
        serialBuffer[bufferIndex] = '\0'; // Null-terminate string
        processCommand(serialBuffer);
        bufferIndex = 0; // Reset buffer
      }
    } else {
      if (bufferIndex < SERIAL_BUFFER_SIZE - 1) {
        serialBuffer[bufferIndex++] = inChar;
      }
    }
  }
}

/**
 * Parses and executes received commands.
 * Expected format: "G:<GESTURE_NAME>"
 */
void processCommand(const char* cmd) {
  if (strncmp(cmd, "G:", 2) == 0) {
    const char* gestureName = cmd + 2;

    // Check if this is an Emergency Gesture
    if (strcmp(gestureName, "HELP") == 0 || strcmp(gestureName, "STOP") == 0) {
      triggerEmergencyAlert(gestureName);
    } else {
      triggerNormalSign(gestureName);
    }
  } else if (strcmp(cmd, "PING") == 0) {
    Serial.println("PONG");
  } else if (strcmp(cmd, "RESET") == 0) {
    displayStandbyScreen();
  }
}

/**
 * Handles Normal Signs (HELLO, YES, NO).
 * Displays word, gives a gentle confirmation chime, lights Green LED.
 */
void triggerNormalSign(const char* signName) {
  currentState = STATE_NORMAL_SIGN;
  stateStartTime = millis();
  strncpy(lastRecognizedGesture, signName, sizeof(lastRecognizedGesture) - 1);

  // Turn off red alert if active
  digitalWrite(PIN_RED_LED, LOW);

  // Light green confirmation LED
  digitalWrite(PIN_GREEN_LED, HIGH);

  // Pleasant 60ms acknowledgement beep
  digitalWrite(PIN_BUZZER, HIGH);
  delay(60);
  digitalWrite(PIN_BUZZER, LOW);

  // Update LCD Display
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Sign: ");
  lcd.print(signName);

  lcd.setCursor(0, 1);
  lcd.print("Status: Transcribed");

  Serial.print("ACK:NORMAL:");
  Serial.println(signName);
}

/**
 * Handles Emergency Signs (HELP, STOP).
 * Triggers rapid red strobe, urgent 3-pulse buzzer alarm, and prominent LCD alert.
 */
void triggerEmergencyAlert(const char* signName) {
  currentState = STATE_EMERGENCY_ALERT;
  stateStartTime = millis();
  emergencyBeepCount = 0;
  strncpy(lastRecognizedGesture, signName, sizeof(lastRecognizedGesture) - 1);

  // Green LED OFF, Red LED initially ON
  digitalWrite(PIN_GREEN_LED, LOW);
  digitalWrite(PIN_RED_LED, HIGH);

  // Urgent Emergency LCD Display
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("! EMERGENCY !");

  lcd.setCursor(0, 1);
  lcd.print("SIGN: ");
  lcd.print(signName);
  lcd.print(" !!");

  Serial.print("ACK:EMERGENCY:");
  Serial.println(signName);
}

/**
 * Non-blocking state machine for handling buzzer cadence and display timeouts.
 */
void updateHardwareOutputs() {
  unsigned long now = millis();

  switch (currentState) {
    case STATE_STANDBY:
      // In standby, all alert actuators are idle
      digitalWrite(PIN_BUZZER, LOW);
      digitalWrite(PIN_RED_LED, LOW);
      digitalWrite(PIN_GREEN_LED, LOW);
      break;

    case STATE_NORMAL_SIGN:
      // Turn off green LED after hold time
      if (now - stateStartTime > GREEN_LED_HOLD_TIME) {
        digitalWrite(PIN_GREEN_LED, LOW);
      }
      // Revert to standby screen after display hold time
      if (now - stateStartTime > DISPLAY_HOLD_TIME) {
        displayStandbyScreen();
      }
      break;

    case STATE_EMERGENCY_ALERT:
      // Rapid Red LED flashing
      if (now - lastBlinkTime >= EMERGENCY_BLINK_INTERVAL) {
        emergencyBlinkState = !emergencyBlinkState;
        digitalWrite(PIN_RED_LED, emergencyBlinkState ? HIGH : LOW);
        lastBlinkTime = now;
      }

      // 3-Pulse Alarm Beep Sequence
      {
        unsigned long elapsed = now - stateStartTime;
        // Pulse 1: 0 to 180 ms
        // Pulse 2: 300 to 480 ms
        // Pulse 3: 600 to 780 ms
        if ((elapsed >= 0 && elapsed <= 180) ||
            (elapsed >= 300 && elapsed <= 480) ||
            (elapsed >= 600 && elapsed <= 780)) {
          digitalWrite(PIN_BUZZER, HIGH);
        } else {
          digitalWrite(PIN_BUZZER, LOW);
        }
      }

      // Revert to standby screen after display hold time
      if (now - stateStartTime > DISPLAY_HOLD_TIME) {
        displayStandbyScreen();
      }
      break;
  }
}

/**
 * Resets LCD to standby presentation screen.
 */
void displayStandbyScreen() {
  currentState = STATE_STANDBY;
  digitalWrite(PIN_BUZZER, LOW);
  digitalWrite(PIN_RED_LED, LOW);
  digitalWrite(PIN_GREEN_LED, LOW);

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("AI SIGN COMM.");
  lcd.setCursor(0, 1);
  lcd.print("WAITING FOR SIGN");
}

/**
 * Startup Hardware Diagnostic Routine.
 * Tests LCD backlight, LEDs, and Buzzer on power-up so students know
 * all exhibition hardware is fully functional.
 */
void startupSelfTest() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("AI SIGN SYSTEM");
  lcd.setCursor(0, 1);
  lcd.print("HARDWARE CHECK..");

  // Flash Green LED
  digitalWrite(PIN_GREEN_LED, HIGH);
  delay(250);
  digitalWrite(PIN_GREEN_LED, LOW);

  // Flash Red LED
  digitalWrite(PIN_RED_LED, HIGH);
  delay(250);
  digitalWrite(PIN_RED_LED, LOW);

  // Short welcome beep
  digitalWrite(PIN_BUZZER, HIGH);
  delay(80);
  digitalWrite(PIN_BUZZER, LOW);

  delay(600);
}
