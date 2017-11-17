import pytest

from TextExtractor import TextExtractor

models_image = 'models.png'
textFragment_image = 'textFragment.png'
maxsize = (64, 64)


def test_expect_throw_on_bad_filepath():
    with pytest.raises(SystemExit):
        t = TextExtractor(maxsize)
        t.read_from_filename("/tmp/veryrandomname23235253")


def test_single_word():
    extractor = TextExtractor(maxsize)
    extractor.read_from_filename(models_image)
    extractor.word_extraction(300, 300, 40, 40, display_images=True)
    assert len(extractor.words) == 1
    extractor.character_extraction(display_images=True, verbose=True)
    assert len(extractor.characters_from_word) == len(extractor.words)
    assert len(extractor.characters_from_word[0]) == 6


def test_text_fragment():
    extractor = TextExtractor(maxsize)
    extractor.read_from_filename(textFragment_image)
    extractor.word_extraction(300, 300, 40, 40, display_images=True)
    assert len(extractor.words) > 1
    extractor.character_extraction(display_images=True, verbose=True)
    assert len(extractor.characters_from_word) == len(extractor.words)
