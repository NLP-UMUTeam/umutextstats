from umutextstats.preprocessing.normalizer import default_normalizer


def normalize(text):
    normalizer = default_normalizer()
    return normalizer.normalize(text)


def test_html_tags_are_removed():
    assert normalize("<p>Hola mundo</p>") == "hola mundo"


def test_html_nested_tags_are_removed():
    assert normalize("<p>Hola <strong>mundo</strong></p>") == "hola mundo"


def test_html_tags_do_not_join_words():
    assert normalize("Hola<strong>mundo</strong>") == "hola mundo"


def test_html_entities_are_decoded():
    assert normalize("Tom &amp; Jerry") == "tom & jerry"


def test_html_accents_entities_are_decoded():
    assert normalize("Informaci&oacute;n p&uacute;blica") == "información pública"


def test_html_br_creates_space():
    assert normalize("Hola<br>mundo") == "hola mundo"


def test_html_complex_news_fragment():
    text = "<div><h1>Título</h1><p>Primera <em>noticia</em>.</p></div>"
    assert normalize(text) == "título primera noticia."