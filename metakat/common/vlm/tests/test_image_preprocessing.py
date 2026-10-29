import pytest

from metakat.common.vlm.image_preprocessing import DEFAULT_MAX_PIXELS, ImagePreprocessing, smallest_text_height


def _alto(tmp_path, lines, page='WIDTH="1000" HEIGHT="2000"', unit="pixel"):
    path = tmp_path / "page.xml"
    path.write_text(
        '<alto xmlns="http://www.loc.gov/standards/alto/ns-v2#">'
        f'<Description><MeasurementUnit>{unit}</MeasurementUnit></Description>'
        f'<Layout><Page {page}><PrintSpace><TextBlock>{"".join(lines)}</TextBlock></PrintSpace></Page></Layout></alto>')
    return path


def _line(*words, height=None):
    strings = "".join(
        f'<String CONTENT="{content}"' + (f' HEIGHT="{h}"' if h is not None else "")
        + (f' WC="{wc}"' if wc is not None else "") + "/>"
        for content, h, wc in words)
    return f'<TextLine{"" if height is None else f" HEIGHT={chr(34)}{height}{chr(34)}"}>{strings}</TextLine>'


def test_max_pixels_always_applies_and_cannot_be_null():
    assert ImagePreprocessing().scaled_size(5064, 6280) == ImagePreprocessing(max_pixels=DEFAULT_MAX_PIXELS).scaled_size(5064, 6280)
    width, height = ImagePreprocessing().scaled_size(5064, 6280)
    assert width * height <= DEFAULT_MAX_PIXELS and abs(width / height - 5064 / 6280) < 0.001
    with pytest.raises(ValueError, match="cannot be null"):
        ImagePreprocessing(max_pixels=None)


def test_an_image_within_every_limit_is_not_scaled():
    assert ImagePreprocessing().scaled_size(2000, 3000) == (2000, 3000)


def test_the_image_fits_all_the_maxes():
    assert ImagePreprocessing(max_w=1000, max_h=1000).scaled_size(4000, 2000) == (1000, 500)
    assert ImagePreprocessing(max_w=1000, max_h=400).scaled_size(4000, 2000) == (800, 400)


def test_text_height_scales_down_but_never_up():
    assert ImagePreprocessing(text_height=20).scaled_size(4000, 2000, text_height=40) == (2000, 1000)
    assert ImagePreprocessing(text_height=20).scaled_size(4000, 2000, text_height=10) == (4000, 2000)
    assert ImagePreprocessing(text_height=20).scaled_size(4000, 2000) == (4000, 2000)


def test_min_pixels_limits_text_height_but_not_the_maxes_and_never_scales_up():
    assert ImagePreprocessing(text_height=10, min_pixels=2_000_000).scaled_size(4000, 2000, text_height=40) == (2000, 1000)
    assert ImagePreprocessing(text_height=10, min_pixels=2_000_000, max_w=1000).scaled_size(4000, 2000, 40) == (1000, 500)
    assert ImagePreprocessing(text_height=10, min_pixels=20_000_000).scaled_size(4000, 2000, 40) == (4000, 2000)


@pytest.mark.parametrize("config, match", [
    ({"max_pixels": 0}, "positive"),
    ({"text_height": True}, "positive"),
    ({"max_w": 1.5}, "positive"),
    ({"min_pixels": 30_000_000}, "exceeds"),
])
def test_invalid_values_are_rejected(config, match):
    with pytest.raises(ValueError, match=match):
        ImagePreprocessing(**config)


def test_the_smallest_text_line_is_measured_in_image_pixels(tmp_path):
    alto = _alto(tmp_path, [
        _line(("Kytice", 40, 0.95), ("z", 38, 0.9), ("pověstí", 42, 0.95)),
        _line(("Praha", 20, 0.9), ("1890", 24, 0.99), ("Otto", 22, 0.9)),
    ])
    # The page is 1000 x 2000 in the ALTO and 2000 x 4000 in the image.
    assert smallest_text_height(alto, (2000, 4000)) == 44


def test_noise_lines_are_not_text(tmp_path):
    alto = _alto(tmp_path, [
        _line(("Kytice", 40, 0.95)),
        _line(("wvw", 5, 0.1)),  # paper texture read as text
        _line(("a", 6, 0.99), ("-", 6, 0.99)),  # a speck
    ])
    assert smallest_text_height(alto, (1000, 2000)) == 40


def test_a_line_without_word_boxes_or_confidences_is_measured_by_its_own_box(tmp_path):
    alto = _alto(tmp_path, [_line(("Kytice", None, None), height=30)])
    assert smallest_text_height(alto, (1000, 2000)) == 30


def test_a_page_without_text_or_convertible_units_has_no_text_height(tmp_path):
    assert smallest_text_height(_alto(tmp_path, [_line(("wvw", 5, 0.1))]), (1000, 2000)) is None
    assert smallest_text_height(_alto(tmp_path, [_line(("Kytice", 40, 0.9))], page="", unit="mm10"), (1000, 2000)) is None
