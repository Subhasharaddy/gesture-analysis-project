# Troubleshooting & Debugging Guide

This reference resolves common technical issues encountered during hardware assembly, software execution, and live exhibition demonstrations.

---

## 1. Camera & Video Feed Issues

### Issue: `Could not access camera index 0` or Black Screen
- **Cause 1: Windows Camera Privacy Permissions**:
  - Open **Windows Settings** -> **Privacy & Security** -> **Camera**.
  - Ensure **Camera access** is set to **On** and **Let desktop apps access your camera** is enabled.
- **Cause 2: Another program is using the webcam**:
  - Close any running applications that might hold the webcam handle (e.g., Zoom, Microsoft Teams, Skype, OBS, or Windows Camera App).
- **Cause 3: Multiple Cameras connected (e.g., external USB webcam + internal laptop camera)**:
  - Open `config.py`.
  - Change `CAMERA_INDEX = 0` to `CAMERA_INDEX = 1` or `CAMERA_INDEX = 2`.

---

## 2. Arduino & Serial Communication Issues

### Issue: `SerialException: PermissionError / Access Denied` on COM port
- **Cause**: The Arduino IDE's **Serial Monitor** or **Serial Plotter** window is currently open.
- **Fix**: In the Arduino IDE, close the Serial Monitor window. Windows locks serial ports to a single active process; Python cannot connect while the IDE monitor is holding the port.

### Issue: Arduino COM Port Not Detected
- **Cause 1: Missing USB-to-UART driver for Arduino clone (CH340/CH341 chip)**:
  - Official Arduino boards use an ATmega16U2 chip and install drivers automatically.
  - Clone Arduino boards use the **WCH CH340G** chip. Download and install the standard **CH341SER.EXE** driver installer if the board shows up as "Unknown Device" in Windows Device Manager.
- **Cause 2: Power-only USB cable**:
  - Some cheap micro-USB or Type-B cables only have power wires and lack data lines. Use a high-quality data transfer USB cable.

### Issue: Testing Without Arduino Hardware
- In `config.py`, ensure `ENABLE_MOCK_SERIAL = True`.
- The application will automatically detect that no hardware is plugged in, print hardware telemetry commands to the terminal console, and run the computer vision and voice output without throwing errors.

---

## 3. 16×2 I2C LCD Issues

### Issue: Backlight is ON, but no text appears (or only blank solid squares)
- **Primary Cause: Contrast potentiometer not adjusted**:
  - On the back of the LCD, find the small blue potentiometer on the black I2C backpack.
  - Take a small Phillips or flathead screwdriver and gently turn the potentiometer screw until clear text characters appear with good contrast against the blue/yellow backlight.

### Issue: LCD stays completely blank or scrambled
- **Cause: I2C Address Mismatch**:
  - Most I2C backpacks have address `0x27`. Some use `0x3F` or `0x20`.
  - Open `arduino/sign_language_exhibition/sign_language_exhibition.ino`.
  - Look for line:
    ```cpp
    LiquidCrystal_I2C lcd(0x27, 16, 2);
    ```
  - Change `0x27` to `0x3F`:
    ```cpp
    LiquidCrystal_I2C lcd(0x3F, 16, 2);
    ```
  - Re-upload the sketch to the Arduino.

#### Diagnostic I2C Scanner Sketch:
If unsure of the address, upload this small sketch to find it via Arduino Serial Monitor:
```cpp
#include <Wire.h>
void setup() {
  Wire.begin();
  Serial.begin(9600);
  Serial.println("\nI2C Scanner");
}
void loop() {
  byte error, address;
  int nDevices = 0;
  for (address = 1; address < 127; address++) {
    Wire.beginTransmission(address);
    error = Wire.endTransmission();
    if (error == 0) {
      Serial.print("I2C device found at address 0x");
      if (address < 16) Serial.print("0");
      Serial.println(address, HEX);
      nDevices++;
    }
  }
  if (nDevices == 0) Serial.println("No I2C devices found\n");
  delay(5000);
}
```

---

## 4. Text-To-Speech (TTS) Audio Issues

### Issue: No sound is coming from laptop speakers
- Check Windows master volume and ensure the default audio playback device is set to your laptop speakers or connected external speaker.
- Test Windows SAPI5 voice: Press `Windows Key + R`, type `narrator`, and check if Windows built-in voice works.
- In `4_app_realtime.py`, press the **`[M]`** key on your keyboard to toggle the mute state if accidentally muted.

---

## 5. Recognition Accuracy & Environmental Factors

### Recommendations for 98%+ Accuracy:
1. **Background Contrast**: Avoid standing directly against a wall that matches skin color. A neutral, uncluttered background works best.
2. **Stable Framing**: Keep your hand approximately 60 cm from the camera. The hand should occupy roughly 25% to 40% of the screen height.
3. **Lighting**: Ensure light shines onto the face and palms from the front or ceiling, rather than directly from behind you.
