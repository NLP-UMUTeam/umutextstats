from umutextstats.preprocessing.normalizer import default_normalizer


def normalize(text):
    normalizer = default_normalizer()
    return normalizer.normalize(text)


# =========================
# Lowercase
# =========================

def test_lowercase():
    assert normalize("HOLA MUNDO") == "hola mundo"


# =========================
# Trim + spaces
# =========================

def test_trim_spaces():
    assert normalize("   hola mundo   ") == "hola mundo"


def test_multiple_spaces():
    assert normalize("hola    mundo") == "hola mundo"


# =========================
# Line breaks
# =========================

def test_line_breaks():
    text = "hola\nmundo"
    assert normalize(text) == "hola. mundo"


# =========================
# HTML
# =========================

def test_remove_html():
    text = "<p>hola mundo</p>"
    assert normalize(text) == "hola mundo"


# =========================
# URLs
# =========================

def test_remove_urls():
    text = "hola http://test.com mundo"
    assert normalize(text) == "hola mundo"


# =========================
# Mentions
# =========================

def test_remove_mentions():
    text = "hola @user mundo"
    assert normalize(text) == "hola mundo"


# =========================
# Hashtags (keep word)
# =========================

def test_remove_hashtag_symbol():
    text = "hola #mundo"
    assert normalize(text) == "hola mundo"


# =========================
# Repeated chars
# =========================

def test_remove_repeated_chars():
    text = "holaaaa"
    assert normalize(text) == "hola"


# =========================
# Combined case
# =========================

def test_combined_cleaning():
    text = "  HOLA!!!   http://test.com  @user \n #Mundo  "
    result = normalize(text)

    assert result == "hola! mundo"