"""
Gesture & Sign Language Database Module.
Centralized repository defining supported signs and gestures with:
  - Semantic separation: 'Gesture Recognition' vs 'Sign Language Recognition'
  - Motion type: Static Posture vs Dynamic Motion (e.g. WAVE)
  - English description & Kannada Unicode translation
  - Audio speech phrases for offline text-to-speech
  - Extensibility: Supports registering custom user-trained gestures at runtime
"""

from typing import Dict, Any, List, Optional


# The 12 Primary Sign Language Classes for Exhibition
CORE_12_GESTURES = [
    "HOME",
    "NAMASTE",
    "HELLO",
    "THANK YOU",
    "YES",
    "NO",
    "HELP",
    "STOP",
    "WATER",
    "FOOD",
    "PLEASE",
    "GOOD"
]

# Central Dictionary of Supported Signs & Gestures
GESTURE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    # --- 12 PRIMARY EXHIBITION SIGN LANGUAGE CLASSES ---
    "HOME": {
        "category": "Sign Language (ISL)",
        "english": "Home / House",
        "kannada": "ಮನೆ",
        "kannada_translit": "Mane",
        "meaning": "Residence / House / Shelter",
        "speech": "Home",
        "is_dynamic": False,
        "is_emergency": False,
        "isl_status": "Official ISL Sign",
        "description": "Roof peak shape: join fingertips at an inverted V-angle like a rooftop.",
        "how_to_perform": "Touch fingertips together at an angle to create a triangular roof peak / slant."
    },
    "NAMASTE": {
        "category": "Sign Language (ISL)",
        "english": "Namaste / Respectful Greeting",
        "kannada": "ನಮಸ್ಕಾರ",
        "kannada_translit": "Namaskara",
        "meaning": "Traditional Indian greeting & sign of respect",
        "speech": "Namaste",
        "is_dynamic": False,
        "is_emergency": False,
        "isl_status": "Official ISL Sign",
        "description": "Both palms pressed flat together in prayer pose (Anjali Mudra) at chest height.",
        "how_to_perform": "Hold flat upright palms pressed together in front of camera with fingers pointing upward."
    },
    "HELLO": {
        "category": "Sign Language (ISL)",
        "english": "Hello / Greetings",
        "kannada": "ಹಲೋ / ನಮಸ್ಕಾರ",
        "kannada_translit": "Hello",
        "meaning": "Friendly greeting / Welcome gesture",
        "speech": "Hello",
        "is_dynamic": False,
        "is_emergency": False,
        "isl_status": "Official ISL Sign",
        "description": "Open upright hand with fingers held together or slightly spread.",
        "how_to_perform": "Hold open upright palm facing camera at shoulder height."
    },
    "THANK YOU": {
        "category": "Sign Language (ISL)",
        "english": "Thank You / Gratitude",
        "kannada": "ಧನ್ಯವಾದಗಳು",
        "kannada_translit": "Dhanyavadagalu",
        "meaning": "Expression of gratitude / Appreciation",
        "speech": "Thank you",
        "is_dynamic": False,
        "is_emergency": False,
        "isl_status": "Official ISL Sign",
        "description": "Flat hand held outward in offering, palm angled slightly upward.",
        "how_to_perform": "Move flat open hand gently forward from chin/chest towards the viewer."
    },
    "YES": {
        "category": "Sign Language (ISL)",
        "english": "Yes / Affirmative",
        "kannada": "ಹೌದು",
        "kannada_translit": "Haudu",
        "meaning": "Agreement / Approval / Consent",
        "speech": "Yes",
        "is_dynamic": False,
        "is_emergency": False,
        "isl_status": "Official ISL Sign",
        "description": "Closed fist held firmly forward, nodding slightly from the wrist.",
        "how_to_perform": "Make a solid fist facing the camera and nod it gently down."
    },
    "NO": {
        "category": "Sign Language (ISL)",
        "english": "No / Negative",
        "kannada": "ಇಲ್ಲ",
        "kannada_translit": "Illa",
        "meaning": "Disagreement / Refusal / Denial",
        "speech": "No",
        "is_dynamic": False,
        "is_emergency": False,
        "isl_status": "Official ISL Sign",
        "description": "Index and Middle fingers extended forward together, others curled.",
        "how_to_perform": "Extend Index and Middle fingers together forward towards the camera."
    },
    "HELP": {
        "category": "Sign Language (ISL)",
        "english": "Help / Emergency",
        "kannada": "ಸಹಾಯ",
        "kannada_translit": "Sahaya",
        "meaning": "Distress signal / Request for immediate assistance",
        "speech": "Help",
        "is_dynamic": False,
        "is_emergency": True,
        "isl_status": "Official ISL Sign",
        "description": "Closed fist with thumb tucked into palm (distress sign).",
        "how_to_perform": "Hold closed fist with thumb folded across palm toward camera."
    },
    "STOP": {
        "category": "Sign Language (ISL)",
        "english": "Stop / Halt",
        "kannada": "ನಿಲ್ಲಿಸಿ",
        "kannada_translit": "Nillisi",
        "meaning": "Command to cease motion / Halt right now",
        "speech": "Stop",
        "is_dynamic": False,
        "is_emergency": True,
        "isl_status": "Official ISL Sign",
        "description": "Flat upright open palm pushed firmly facing forward.",
        "how_to_perform": "Push open flat palm forward with fingers straight and held together."
    },
    "WATER": {
        "category": "Sign Language (ISL)",
        "english": "Water / Thirsty",
        "kannada": "ನೀರು",
        "kannada_translit": "Neeru",
        "meaning": "Request for drinking water",
        "speech": "Water",
        "is_dynamic": False,
        "is_emergency": False,
        "isl_status": "Official ISL Sign",
        "description": "'W' handshape: Index, Middle, and Ring fingers extended upright in a 'W', thumb holding pinky.",
        "how_to_perform": "Form a 'W' by holding Index, Middle, and Ring fingers up, with thumb holding pinky."
    },
    "FOOD": {
        "category": "Sign Language (ISL)",
        "english": "Food / Eat / Hungry",
        "kannada": "ಊಟ / ಆಹಾರ",
        "kannada_translit": "Oota",
        "meaning": "Request for food / Eating meal",
        "speech": "Food",
        "is_dynamic": False,
        "is_emergency": False,
        "isl_status": "Official ISL Sign",
        "description": "Flattened 'O' handshape: all 5 fingertips gathered together pointing upward/towards mouth.",
        "how_to_perform": "Pinch all fingertips and thumb tips together pointing upward towards face."
    },
    "PLEASE": {
        "category": "Sign Language (ISL)",
        "english": "Please / Request",
        "kannada": "ದಯವಿಟ್ಟು",
        "kannada_translit": "Dayavittu",
        "meaning": "Polite request / Entreaty",
        "speech": "Please",
        "is_dynamic": False,
        "is_emergency": False,
        "isl_status": "Official ISL Sign",
        "description": "Open flat hand held facing the chest or held flat in polite request.",
        "how_to_perform": "Place open flat palm facing chest / held flat towards camera in polite gesture."
    },
    "GOOD": {
        "category": "Sign Language (ISL)",
        "english": "Good / Well Done",
        "kannada": "ಒಳ್ಳೆಯದು / ಉತ್ತಮ",
        "kannada_translit": "Olleyadu",
        "meaning": "Quality is good / Positive feedback / Thumbs Up",
        "speech": "Good",
        "is_dynamic": False,
        "is_emergency": False,
        "isl_status": "Official ISL Sign",
        "description": "Thumb pointed vertically upward, all four fingers curled firmly into a fist.",
        "how_to_perform": "Point thumb straight up with other fingers folded tightly in a fist."
    },
    "I / ME": {
        "category": "Sign Language",
        "english": "I / Myself",
        "kannada": "ನಾನು",
        "kannada_translit": "Naanu",
        "meaning": "First-person pronoun / Indicating oneself",
        "speech": "I am referring to myself. Naanu.",
        "is_dynamic": False,
        "is_emergency": False,
        "description": "Index finger or thumb pointing towards user's chest."
    },
    "YOU": {
        "category": "Sign Language",
        "english": "You / Second Person",
        "kannada": "ನೀವು",
        "kannada_translit": "Neevu",
        "meaning": "Second-person pronoun / Addressing listener",
        "speech": "You. Neevu.",
        "is_dynamic": False,
        "is_emergency": False,
        "description": "Index finger pointing directly forward towards viewer."
    },
    "WELCOME": {
        "category": "Sign Language",
        "english": "Welcome / Hospitality",
        "kannada": "ಸುಸ್ವಾಗತ",
        "kannada_translit": "Suswagatha",
        "meaning": "Welcoming a guest / Hospitality",
        "speech": "Welcome! Suswagatha.",
        "is_dynamic": False,
        "is_emergency": False,
        "description": "Open hand held welcomingly with palm angled upward."
    },
    "BYE": {
        "category": "Sign Language",
        "english": "Goodbye / Farewell",
        "kannada": "ವಿದಾಯ",
        "kannada_translit": "Vidaya",
        "meaning": "Farewell greeting / Leaving",
        "speech": "Goodbye! Vidaya.",
        "is_dynamic": True,
        "is_emergency": False,
        "description": "Hand held up moving in farewell."
    },

    # --- POPULAR GESTURE RECOGNITION ---
    "THUMBS UP": {
        "category": "Gesture",
        "english": "Thumbs Up / Good",
        "kannada": "ಅಂಗೀಕಾರ / ಶ್ರೇಷ್ಠ",
        "kannada_translit": "Angikara / Shreshta",
        "meaning": "Positive approval / Good / Agreement",
        "speech": "Thumbs up! Good job.",
        "is_dynamic": False,
        "is_emergency": False,
        "description": "Thumb pointed vertically upward, all four fingers curled into fist."
    },
    "THUMBS DOWN": {
        "category": "Gesture",
        "english": "Thumbs Down / Bad",
        "kannada": "ತಿರಸ್ಕಾರ / ಕೆಟ್ಟದು",
        "kannada_translit": "Tiraskara / Kettadu",
        "meaning": "Disapproval / Bad / Negative feedback",
        "speech": "Thumbs down. Disagree.",
        "is_dynamic": False,
        "is_emergency": False,
        "description": "Thumb pointed vertically downward, all four fingers curled into fist."
    },
    "OK": {
        "category": "Gesture",
        "english": "OK / Perfect",
        "kannada": "ಸರಿ / ಪರಿಪೂರ್ಣ",
        "kannada_translit": "Sari / Paripurna",
        "meaning": "Approval / Okay / Everything is fine",
        "speech": "OK, everything is good. Sari.",
        "is_dynamic": False,
        "is_emergency": False,
        "description": "Thumb tip and index fingertip touching to form an 'O', other fingers extended."
    },
    "PEACE": {
        "category": "Gesture",
        "english": "Peace / Victory",
        "kannada": "ಶಾಂತಿ / ಜಯ",
        "kannada_translit": "Shanti / Jaya",
        "meaning": "Victory V-sign / Peace symbol",
        "speech": "Peace and victory! Shanti.",
        "is_dynamic": False,
        "is_emergency": False,
        "description": "Index and Middle fingers extended upright in a V-shape, others curled."
    },
    "LOVE": {
        "category": "Gesture",
        "english": "I Love You (ILY)",
        "kannada": "ಪ್ರೀತಿ",
        "kannada_translit": "Preethi",
        "meaning": "Universal 'I Love You' sign / Affection",
        "speech": "I love you. Preethi.",
        "is_dynamic": False,
        "is_emergency": False,
        "description": "Thumb, Index finger, and Pinky finger extended; Middle and Ring curled."
    },
    "GOOD": {
        "category": "Gesture",
        "english": "Good / Positive",
        "kannada": "ಒಳ್ಳೆಯದು",
        "kannada_translit": "Olleya",
        "meaning": "Quality is good / Satisfactory",
        "speech": "Very good! Olleyadu.",
        "is_dynamic": False,
        "is_emergency": False,
        "description": "Thumb upright gesture expressing positive status."
    },
    "BAD": {
        "category": "Gesture",
        "english": "Bad / Poor",
        "kannada": "ಕೆಟ್ಟದ್ದು",
        "kannada_translit": "Kettaddu",
        "meaning": "Poor quality / Negative reaction",
        "speech": "Bad, not acceptable. Kettaddu.",
        "is_dynamic": False,
        "is_emergency": False,
        "description": "Downward thumb expressing negative status."
    },
    "POINT": {
        "category": "Gesture",
        "english": "Pointing / Direction",
        "kannada": "ನಿರ್ದೇಶನ",
        "kannada_translit": "Nirdeshana",
        "meaning": "Indicating a direction or object",
        "speech": "Pointing to target. Nirdeshana.",
        "is_dynamic": False,
        "is_emergency": False,
        "description": "Index finger extended outward, other fingers folded."
    },
    "FIST": {
        "category": "Gesture",
        "english": "Fist / Determination",
        "kannada": "ಮುಷ್ಟಿ",
        "kannada_translit": "Mushti",
        "meaning": "Power / Strength / Solid fist",
        "speech": "Fist gesture. Mushti.",
        "is_dynamic": False,
        "is_emergency": False,
        "description": "All five fingers curled tight into palm."
    },
    "OPEN PALM": {
        "category": "Gesture",
        "english": "Open Hand / Calm",
        "kannada": "ತೆರೆದ ಕೈ",
        "kannada_translit": "Tereda Kai",
        "meaning": "Open flat hand / Calm and relaxed",
        "speech": "Open palm hand.",
        "is_dynamic": False,
        "is_emergency": False,
        "description": "All five fingers relaxed and spread open facing camera."
    },
    "WAVE": {
        "category": "Dynamic Gesture",
        "english": "Hand Wave / Waving",
        "kannada": "ಕೈ ಬೀಸುವುದು",
        "kannada_translit": "Kai Beesuvudu",
        "meaning": "Motion greeting or goodbye gesture",
        "speech": "Hello there! Waving hand.",
        "is_dynamic": True,
        "is_emergency": False,
        "description": "Open hand oscillating left-right across multiple frames."
    }
}


def get_all_gesture_names() -> List[str]:
    """Returns list of all supported gesture names."""
    return list(GESTURE_DEFINITIONS.keys())


def get_gesture_info(gesture_name: str) -> Dict[str, Any]:
    """Retrieves metadata dictionary for a given gesture."""
    key = str(gesture_name).strip().upper()
    if key in GESTURE_DEFINITIONS:
        return GESTURE_DEFINITIONS[key]
    # Check case-insensitive
    for g, info in GESTURE_DEFINITIONS.items():
        if g.upper() == key:
            return info
    # Fallback for unknown
    return {
        "category": "Unknown",
        "english": "Unknown Gesture",
        "kannada": "ಗುರುತಿಸಲಾಗದ ಸಂಕೇತ",
        "kannada_translit": "Gurutisalagada",
        "meaning": "Gesture not recognized",
        "speech": "",
        "is_dynamic": False,
        "is_emergency": False,
        "description": "Unclassified hand position."
    }


def register_custom_gesture(
    name: str,
    english: str,
    kannada: str = "",
    meaning: str = "",
    speech: str = "",
    category: str = "Custom Gesture"
):
    """Dynamically registers a newly trained user gesture into the database."""
    name_clean = name.strip().upper()
    GESTURE_DEFINITIONS[name_clean] = {
        "category": category,
        "english": english or name_clean,
        "kannada": kannada or name_clean,
        "kannada_translit": "",
        "meaning": meaning or f"Custom gesture: {name_clean}",
        "speech": speech or f"Recognized {name_clean}",
        "is_dynamic": False,
        "is_emergency": False,
        "description": f"Custom user-trained gesture '{name_clean}'"
    }


def build_speech_map() -> Dict[str, str]:
    """Generates the speech map for TextToSpeechWorker."""
    return {g: info["speech"] for g, info in GESTURE_DEFINITIONS.items() if info["speech"]}


# Convenience Alias
GESTURE_DATABASE = GESTURE_DEFINITIONS


def get_all_supported_gestures() -> List[str]:
    """Returns list of recognized gesture names excluding system states."""
    return [k for k in GESTURE_DEFINITIONS.keys() if k not in ("STANDBY", "UNKNOWN GESTURE")]


def get_gesture_info(name: str) -> Dict[str, Any]:
    """Safely retrieves metadata for any gesture name."""
    clean = (name or "").strip().upper()
    return GESTURE_DEFINITIONS.get(clean, GESTURE_DEFINITIONS.get("UNKNOWN GESTURE", {}))

