import pytest, sys
from textExtractor import TextExtractor

models_image='./example_images/models.png'

@pytest.fixture
def extractor():
    return TextExtractor()

def test_empty(extractor):
    extractor.pathToImage = models_image
    extractor.contourExample(300,300,40,40)
    assert len(extractor.words) == 1
    extractor.characterExtraction()
    assert len(extractor.characters) == 6
