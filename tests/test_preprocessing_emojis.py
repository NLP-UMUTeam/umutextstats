from umutextstats.preprocessing.normalizer import default_normalizer


def normalize(text):
    normalizer = default_normalizer(replace_emojis=True)
    return normalizer.normalize(text)


# =========================
# Basic emoji replacement
# =========================

def test_single_emoji():
    assert normalize("Hola 🙂") == "hola /slightly_smiling_face/"


def test_multiple_emojis():
    assert normalize("Genial 😄🔥") == (
        "genial /grinning_face_with_smiling_eyes/ /fire/"
    )


# =========================
# Inside sentence
# =========================

def test_emoji_in_sentence():
    assert normalize("Esto me gusta ❤️ mucho") == (
        "esto me gusta /red_heart/ mucho"
    )


# =========================
# Disabled by default
# =========================

def test_emojis_disabled_by_default():
    normalizer = default_normalizer()
    assert normalizer.normalize("Hola 🙂") == "hola 🙂"


# =========================
# Interaction with punctuation
# =========================

def test_emoji_with_punctuation():
    assert normalize("Bien!!! 😄") == (
        "bien! /grinning_face_with_smiling_eyes/"
    )


# =========================
# Edge cases
# =========================

def test_only_emoji():
    assert normalize("🙂") == "/slightly_smiling_face/"


def test_emoji_with_spaces():
    assert normalize("  🙂  ") == "/slightly_smiling_face/"


def test_emoji_and_hashtag():
    assert normalize("#Hola 🙂") == "hola /slightly_smiling_face/"
    
    
def test_emoji_attached_to_word():
    assert normalize("Hola🙂") == "hola /slightly_smiling_face/"


def test_emoji_with_skin_tone():
    assert normalize("Bien 👍🏽") == "bien /thumbs_up_medium_skin_tone/"


def test_heart_emoji():
    assert normalize("Me gusta ❤️") == "me gusta /red_heart/"


def test_emoji_sequence_family():
    assert normalize("Familia 👨‍👩‍👧‍👦") == "familia /family_man_woman_girl_boy/"