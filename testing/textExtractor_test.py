import pytest, sys
from textExtractor import TextExtractor

models_image='./example_images/models.png'
textFragment_image='./example_images/textFragment.png'
maxsize=(64,64)

def test_expect_throw_on_bad_filepath():
    with pytest.raises(SystemExit):
        t = TextExtractor(maxsize)
        t.readFromFilename("/tmp/veryrandomname23235253")

def test_single_word():
    extractor = TextExtractor(maxsize)
    extractor.readFromFilename(models_image)
    extractor.wordExtraction(300,300,40,40, displayImages=True)
    assert len(extractor.words) == 1
    extractor.characterExtraction(displayImages=True, verbose=True)
    assert len(extractor.charactersFromWord) == len(extractor.words)
    assert len(extractor.charactersFromWord[0]) == 6

def test_text_fragment():
    extractor = TextExtractor(maxsize)
    extractor.readFromFilename(textFragment_image)
    extractor.wordExtraction(300,300,40,40, displayImages=True)
    assert len(extractor.words) > 1
    extractor.characterExtraction(displayImages=True, verbose=True)
    assert len(extractor.charactersFromWord) == len(extractor.words)
