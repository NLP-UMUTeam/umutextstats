from umutextstats.preprocessing.normalizer import default_normalizer


def normalize(text):
    return default_normalizer().normalize(text)


def test_remove_mention():
    assert normalize("Hola @usuario") == "hola"


def test_remove_multiple_mentions():
    assert normalize("@uno hola @dos mundo") == "hola mundo"


def test_keep_hashtag_text():
    assert normalize("Me gusta #Python") == "me gusta python"


def test_keep_hashtag_with_accent():
    assert normalize("Viva #España") == "viva españa"


def test_url_is_removed():
    assert normalize("Lee esto https://example.com ahora") == "lee esto ahora"


def test_www_url_is_removed():
    assert normalize("Lee www.example.com ahora") == "lee ahora"


def test_url_with_query_is_removed():
    assert normalize("Mira https://example.com?a=1&b=2") == "mira"


def test_retweet_marker_basic():
    assert normalize("RT @usuario: Hola mundo") == "rt: hola mundo"


def test_retweet_marker_basic():
    assert normalize("RT @usuario: Hola mundo") == "hola mundo"
    
    
def test_mentions_do_not_leave_extra_colon():
    assert normalize("@usuario: Hola mundo") == "hola mundo"


def test_hashtag_with_punctuation():
    assert normalize("Grande #España!") == "grande españa!"
    