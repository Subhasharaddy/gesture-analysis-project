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

# Canonical 12-Class Numerical ID to Gesture Name Mapping (Alphabetical order from sign_label_encoder)
CLASS_ID_TO_GESTURE_MAP_12: Dict[int, str] = {
    0: "FOOD",
    1: "GOOD",
    2: "HELLO",
    3: "HELP",
    4: "HOME",
    5: "NAMASTE",
    6: "NO",
    7: "PLEASE",
    8: "STOP",
    9: "THANK YOU",
    10: "WATER",
    11: "YES"
}

# Canonical 42-Class Numerical ID to Gesture Name Mapping
CLASS_ID_TO_GESTURE_MAP: Dict[int, str] = {
    0: 'BAD',
    1: 'BYE',
    2: 'CALL FOR HELP',
    3: 'COME',
    4: 'DOCTOR',
    5: 'EMERGENCY',
    6: 'FIST',
    7: 'FOOD',
    8: 'FRIEND',
    9: 'GO',
    10: 'GOOD',
    11: 'GOOD MORNING',
    12: 'GOOD NIGHT',
    13: 'HELLO',
    14: 'HELP',
    15: 'HOME',
    16: 'HOSPITAL',
    17: 'I / ME',
    18: 'I AM FINE',
    19: 'LOVE',
    20: 'NAMASTE',
    21: 'NO',
    22: 'OK',
    23: 'OPEN PALM',
    24: 'PEACE',
    25: 'PHONE',
    26: 'PLEASE',
    27: 'POINT',
    28: 'POLICE',
    29: 'SCHOOL',
    30: 'SORRY',
    31: 'STOP',
    32: 'THANK YOU',
    33: 'THANKS',
    34: 'THUMBS DOWN',
    35: 'THUMBS UP',
    36: 'WAIT',
    37: 'WATER',
    38: 'WAVE',
    39: 'WELCOME',
    40: 'YES',
    41: 'YOU'
}

GESTURE_TO_CLASS_ID_MAP: Dict[str, int] = {v: k for k, v in CLASS_ID_TO_GESTURE_MAP.items()}

def class_id_to_gesture_name(class_id: Any, num_classes: Optional[int] = None) -> str:
    """Converts a model class index (int/float/str) to the canonical gesture name."""
    try:
        cid = int(class_id)
        if num_classes == 12:
            if cid in CLASS_ID_TO_GESTURE_MAP_12:
                return CLASS_ID_TO_GESTURE_MAP_12[cid]
        if cid in CLASS_ID_TO_GESTURE_MAP:
            return CLASS_ID_TO_GESTURE_MAP[cid]
        if cid in CLASS_ID_TO_GESTURE_MAP_12:
            return CLASS_ID_TO_GESTURE_MAP_12[cid]
    except (ValueError, TypeError):
        pass
    return str(class_id).strip().upper()

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
    },
    "CALL FOR HELP": {
        "category": "Emergency Sign",
        "english": "Call For Help",
        "kannada": "ಸಹಾಯಕ್ಕಾಗಿ ಕರೆ ಮಾಡಿ",
        "kannada_translit": "Sahayakkagi Kare Madi",
        "meaning": "Urgent request to summon external help",
        "speech": "Emergency! Please call for help right now!",
        "is_dynamic": False,
        "is_emergency": True,
        "description": "Thumb and pinky extended to ear like a phone, urgent distress sign."
    },
    "COME": {
        "category": "Conversational Sign",
        "english": "Come / Arrive",
        "kannada": "ಬನ್ನಿ",
        "kannada_translit": "Banni",
        "meaning": "Beckoning someone to approach or come closer",
        "speech": "Please come over here.",
        "is_dynamic": False,
        "is_emergency": False,
        "description": "Palm facing upward with index finger beckoning inward."
    },
    "DOCTOR": {
        "category": "Medical Sign",
        "english": "Doctor / Medical Professional",
        "kannada": "ವೈದ್ಯರು",
        "kannada_translit": "Vaidyaru",
        "meaning": "Healthcare professional / Physician",
        "speech": "I need to consult a doctor.",
        "is_dynamic": False,
        "is_emergency": False,
        "description": "Three middle fingertips tapping wrist pulse location."
    },
    "EMERGENCY": {
        "category": "Emergency Sign",
        "english": "Emergency / Critical Alert",
        "kannada": "ತುರ್ತು ಪರಿಸ್ಥಿತಿ",
        "kannada_translit": "Thurtu Paristhithi",
        "meaning": "Critical situation requiring immediate intervention",
        "speech": "Critical emergency alert! Urgent assistance required!",
        "is_dynamic": False,
        "is_emergency": True,
        "description": "Upright open palm waving rapidly across chest with wide eyes."
    },
    "FRIEND": {
        "category": "Social Sign",
        "english": "Friend / Companion",
        "kannada": "ಸ್ನೇಹಿತ",
        "kannada_translit": "Snehitha",
        "meaning": "Companion / Friendship bond",
        "speech": "You are my good friend.",
        "is_dynamic": False,
        "is_emergency": False,
        "description": "Both index fingers hooked together or interlocking hand clasp."
    },
    "GO": {
        "category": "Directional Sign",
        "english": "Go / Leave",
        "kannada": "ಹೋಗಿ",
        "kannada_translit": "Hogi",
        "meaning": "Directing someone to proceed or move away",
        "speech": "You may go ahead.",
        "is_dynamic": False,
        "is_emergency": False,
        "description": "Index finger pointing forward and sweeping away from body."
    },
    "GOOD MORNING": {
        "category": "Greeting",
        "english": "Good Morning",
        "kannada": "ಶುಭೋದಯ",
        "kannada_translit": "Shubhodaya",
        "meaning": "Morning greeting / Salutation",
        "speech": "Good morning! Have a wonderful day.",
        "is_dynamic": False,
        "is_emergency": False,
        "description": "Sign for 'Good' followed by sun rising gesture."
    },
    "GOOD NIGHT": {
        "category": "Greeting",
        "english": "Good Night",
        "kannada": "ಶುಭ ರಾತ್ರಿ",
        "kannada_translit": "Shubha Rathri",
        "meaning": "Night parting / Bedtime blessing",
        "speech": "Good night, sweet dreams.",
        "is_dynamic": False,
        "is_emergency": False,
        "description": "Sign for 'Good' followed by hand draping downward like setting night."
    },
    "HOSPITAL": {
        "category": "Medical Sign",
        "english": "Hospital / Clinic",
        "kannada": "ಆಸ್ಪತ್ರೆ",
        "kannada_translit": "Aaspathre",
        "meaning": "Medical treatment facility / Clinic",
        "speech": "Please take me to the hospital.",
        "is_dynamic": False,
        "is_emergency": False,
        "description": "Index finger drawing a cross '+' shape on opposite upper shoulder/arm."
    },
    "I AM FINE": {
        "category": "Conversational Sign",
        "english": "I Am Fine / Well",
        "kannada": "ನಾನು ಚೆನ್ನಾಗಿದ್ದೇನೆ",
        "kannada_translit": "Naanu Chennagiddene",
        "meaning": "Reassurance of well-being / Satisfactory health",
        "speech": "I am doing totally fine, thank you.",
        "is_dynamic": False,
        "is_emergency": False,
        "description": "Open hand with thumb touching chest and moving outward with a smile."
    },
    "PHONE": {
        "category": "Communication Sign",
        "english": "Phone / Telephone Call",
        "kannada": "ದೂರವಾಣಿ",
        "kannada_translit": "Dooravani",
        "meaning": "Telephone call / Mobile device",
        "speech": "Please make a phone call.",
        "is_dynamic": False,
        "is_emergency": False,
        "description": "Thumb at ear and pinky at mouth mimicking phone receiver."
    },
    "POLICE": {
        "category": "Emergency Sign",
        "english": "Police / Security",
        "kannada": "ಪೊಲೀಸ್",
        "kannada_translit": "Police",
        "meaning": "Law enforcement / Security officer",
        "speech": "Alert! Please call the police immediately!",
        "is_dynamic": False,
        "is_emergency": True,
        "description": "Two fingers tapping upper chest like police badge or peaked cap."
    },
    "SCHOOL": {
        "category": "Institutional Sign",
        "english": "School / Classroom",
        "kannada": "ಶಾಲೆ",
        "kannada_translit": "Shaale",
        "meaning": "Educational institution / Study place",
        "speech": "I am going to school.",
        "is_dynamic": False,
        "is_emergency": False,
        "description": "Open flat hands clapping twice horizontally."
    },
    "SORRY": {
        "category": "Conversational Sign",
        "english": "Sorry / Apology",
        "kannada": "ಕ್ಷಮಿಸಿ",
        "kannada_translit": "Kshamisi",
        "meaning": "Apology / Regret expression",
        "speech": "I am really sorry.",
        "is_dynamic": False,
        "is_emergency": False,
        "description": "Fist rubbing in circular motion over heart/chest."
    },
    "THANKS": {
        "category": "Conversational Sign",
        "english": "Thanks / Appreciation",
        "kannada": "ಧನ್ಯವಾದ",
        "kannada_translit": "Dhanyavada",
        "meaning": "Informal expression of thanks",
        "speech": "Many thanks to you.",
        "is_dynamic": False,
        "is_emergency": False,
        "description": "Fingertips touching lips/chin and moving forward toward person."
    },
    "WAIT": {
        "category": "Instructional Sign",
        "english": "Wait / Pause",
        "kannada": "ಕಾಯಿರಿ",
        "kannada_translit": "Kaayiri",
        "meaning": "Hold on / Stand by / Do not rush",
        "speech": "Please wait a moment.",
        "is_dynamic": False,
        "is_emergency": False,
        "description": "Palm facing outward with fingers wiggling gently or held firm."
    },
    "EAT": {
        "category": "Sign Language (ISL)",
        "english": "Eat / Food",
        "kannada": "ಊಟ",
        "kannada_translit": "Oota",
        "meaning": "Eating meal",
        "speech": "Eat",
        "is_dynamic": False,
        "is_emergency": False,
        "description": "Same as FOOD gesture. Flattened 'O' handshape."
    },
    "SLEEP": {
        "category": "Greeting",
        "english": "Sleep",
        "kannada": "ನಿದ್ದೆ",
        "kannada_translit": "Nidde",
        "meaning": "Going to sleep",
        "speech": "Sleep.",
        "is_dynamic": False,
        "is_emergency": False,
        "description": "Same as GOOD NIGHT."
    },
    "COME HERE": {
        "category": "Conversational Sign",
        "english": "Come Here",
        "kannada": "ಇಲ್ಲಿ ಬಾ",
        "kannada_translit": "Illi baa",
        "meaning": "Beckoning",
        "speech": "Come here.",
        "is_dynamic": False,
        "is_emergency": False,
        "description": "Same as COME gesture."
    },
    "DANGER": {
        "category": "Emergency Sign",
        "english": "Danger",
        "kannada": "ಅಪಾಯ",
        "kannada_translit": "Apaaya",
        "meaning": "Critical warning",
        "speech": "Danger!",
        "is_dynamic": False,
        "is_emergency": True,
        "description": "Same as EMERGENCY."
    },
    "VICTORY": {
        "category": "Gesture",
        "english": "Victory",
        "kannada": "ಜಯ",
        "kannada_translit": "Jaya",
        "meaning": "Winning",
        "speech": "Victory!",
        "is_dynamic": False,
        "is_emergency": False,
        "description": "Same as PEACE gesture."
    },
    "ROCK ON": {
        "category": "Gesture",
        "english": "Rock On",
        "kannada": "ರಾಕ್ ಆನ್",
        "kannada_translit": "Rock on",
        "meaning": "Rock on symbol",
        "speech": "Rock on!",
        "is_dynamic": False,
        "is_emergency": False,
        "description": "Index and pinky fingers extended, middle and ring fingers curled, thumb over middle fingers."
    },
    "CLOSED FIST": {
        "category": "Gesture",
        "english": "Closed Fist",
        "kannada": "ಮುಷ್ಟಿ",
        "kannada_translit": "Mushti",
        "meaning": "Closed Fist",
        "speech": "Fist.",
        "is_dynamic": False,
        "is_emergency": False,
        "description": "Same as FIST."
    },
    "INDEX FINGER POINT": {
        "category": "Gesture",
        "english": "Index Finger Point",
        "kannada": "ನಿರ್ದೇಶನ",
        "kannada_translit": "Nirdeshana",
        "meaning": "Pointing",
        "speech": "Point.",
        "is_dynamic": False,
        "is_emergency": False,
        "description": "Same as POINT."
    },
    "PEACE SIGN": {
        "category": "Gesture",
        "english": "Peace Sign",
        "kannada": "ಶಾಂತಿ",
        "kannada_translit": "Shanti",
        "meaning": "Peace",
        "speech": "Peace.",
        "is_dynamic": False,
        "is_emergency": False,
        "description": "Same as PEACE."
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

