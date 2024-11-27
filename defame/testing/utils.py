import pytest
from defame.utils.parsing import is_image_url

@pytest.mark.parametrize("url,expected", [
    ("https://media.cnn.com/api/v1/images/stellar/prod/ap22087057359494.jpg?c=16x9&q=h_653,w_1160,c_fill/f_webp", True),
    ("https://edition.cnn.com/2024/10/30/asia/north-korea-icbm-test-intl-hnk/index.html", False),
    ("https://img.zeit.de/politik/ausland/2024-10/georgien-wahl-stimmauszaehlung-regierungspartei-bild/wide__1000x562__desktop__scale_2", True),
    ("https://upload.wikimedia.org/wikipedia/commons/8/8d/President_Barack_Obama.jpg", True),
    ("https://de.wikipedia.org/wiki/Datei:President_Barack_Obama.jpg", False),  # this is the image's article view
    ("https://bingekulture.com/wp-content/uploads/2021/08/cropped-cropped-logo.fw-removebg-preview.png?w=48", False),  # this URL redirects to a webpage
    ("https://www.popularmechanics.com/_assets/design-tokens/fre/static/icons/play.db7c035.svg?primary=%2523ffffff%20%22Play%22", False),  # this is a vector graphic
])
def test_is_image_url(url, expected):
    assert is_image_url(url) == expected
