from pathlib import Path
from xml.etree import ElementTree as ET

from metakat.common.models import AltoRefs, BoundingBox, PageDimensions
from metakat.page_number.engines.core.alto_recovery import AltoPageNumberRecovery
from metakat.page_number.engines.core.postprocessing import PageNumberPostprocessInput


def write_alto(path: Path, lines):
    root = ET.Element("alto", xmlns="http://www.loc.gov/standards/alto/ns-v4#")
    page = ET.SubElement(root, "Page", WIDTH="1000", HEIGHT="1000")
    block = ET.SubElement(page, "TextBlock", ID="block")
    for i, (texts, y) in enumerate(lines):
        line = ET.SubElement(block, "TextLine", ID=f"line{i}")
        for j, text in enumerate(texts):
            ET.SubElement(
                line,
                "String",
                CONTENT=text,
                HPOS=str(480 + 20 * j),
                VPOS=str(y),
                WIDTH="20",
                HEIGHT="20",
                ID=f"w{i}-{j}",
            )
    ET.ElementTree(root).write(path, encoding="unicode")
    return path


def test_recovery_scans_only_number_lines_inside_windows_and_keeps_refs(tmp_path):
    path = write_alto(
        tmp_path / "p.xml",
        [
            (["Some", "prose", "123"], 950),
            (["-", "1", "2", "-"], 950),
            (["99"], 400),
            (["1", "/", "2"], 950),
            (["xiv"], 50),
        ],
    )
    page = PageNumberPostprocessInput("p", PageDimensions(1000, 1000))
    results = AltoPageNumberRecovery({"p": str(path)})(
        page, (BoundingBox(400, 900, 200, 100),)
    )
    assert len(results) == 1
    evidence = results[0]
    assert evidence.text == "- 1 2 -"
    assert evidence.value == 12
    assert evidence.bbox == BoundingBox(480, 950, 80, 20)
    assert evidence.alto == AltoRefs(
        ("block",), ("line1",), ("w1-0", "w1-1", "w1-2", "w1-3")
    )


def test_recovery_handles_roman_numerals(tmp_path):
    path = write_alto(tmp_path / "p.xml", [(["xiv"], 50)])
    page = PageNumberPostprocessInput("p", PageDimensions(1000, 1000))
    results = AltoPageNumberRecovery({"p": str(path)})(
        page, (BoundingBox(400, 0, 200, 100),)
    )
    assert results[0].output_text() == "xiv"


def test_number_can_share_a_line_with_a_distant_running_header(tmp_path):
    path = write_alto(tmp_path / "p.xml", [(["Journal", "42"], 50)])
    tree = ET.parse(path)
    words = list(
        tree.getroot().iter("{http://www.loc.gov/standards/alto/ns-v4#}String")
    )
    words[0].set("HPOS", "100")
    tree.write(path, encoding="unicode")
    page = PageNumberPostprocessInput("p", PageDimensions(1000, 1000))
    results = AltoPageNumberRecovery({"p": str(path)})(
        page, (BoundingBox(400, 0, 200, 100),)
    )
    assert [e.value for e in results] == [42]
    assert results[0].alto.words == ("w0-1",)
