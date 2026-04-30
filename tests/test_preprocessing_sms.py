from umutextstats.preprocessing.normalizer import default_normalizer


SMS_REPLACEMENTS = {
    "q": "que",
    "xq": "porque",
    "tb": "también",
    "tmb": "también",
    "k": "que",
}


def normalize(text):
    normalizer = default_normalizer(
        expand_sms=True,
        sms_replacements=SMS_REPLACEMENTS,
    )
    return normalizer.normalize(text)


# =========================
# Casos básicos
# =========================

def test_expand_simple_sms_word():
    assert normalize("q haces") == "que haces"


def test_expand_multiple_sms_words():
    assert normalize("q tal tb vienes") == "que tal también vienes"


def test_expand_sms_case_insensitive():
    assert normalize("Q haces") == "que haces"


# =========================
# Contexto
# =========================

def test_expand_sms_inside_sentence():
    assert normalize("no voy xq estoy cansado") == "no voy porque estoy cansado"


def test_expand_sms_with_punctuation():
    assert normalize("q?") == "que?"


# =========================
# Edge cases
# =========================

def test_sms_does_not_replace_inside_words():
    assert normalize("queso") == "queso"


def test_sms_with_uppercase_mixed():
    assert normalize("Q Tal") == "que tal"


def test_sms_with_symbols():
    assert normalize("@user q haces") == "que haces"


def test_sms_disabled_by_default():
    normalizer = default_normalizer()
    assert normalizer.normalize("q haces") == "q haces"


# =========================
# Casos reales
# =========================

def test_sms_real_sentence():
    text = "q haces? tb vienes o q?"
    expected = "que haces? también vienes o que?"
    assert normalize(text) == expected


def test_sms_with_urls_and_mentions():
    text = "q haces http://test.com @user"
    assert normalize(text) == "que haces"