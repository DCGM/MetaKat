from pathlib import Path

from PIL import Image

from metakat.process_batch import init_io
from metakat.schemas.base_objects import MetakatPage, MetakatPageDimensions


def _write_alto(path: Path, width: int, height: int) -> None:
    path.write_text(
        f'''<alto xmlns="http://www.loc.gov/standards/alto/ns-v2#">
  <Layout>
    <Page ID="page" WIDTH="{width}" HEIGHT="{height}"/>
  </Layout>
</alto>
''',
        encoding="utf-8",
    )


def test_init_io_loads_image_and_alto_dimensions(tmp_path):
    Image.new("RGB", (320, 480)).save(tmp_path / "page.jpg")
    _write_alto(tmp_path / "page.xml", 320, 480)

    metakat_io, _ = init_io(
        str(tmp_path),
        ordered_image_filenames=["page.jpg"],
    )

    page = next(
        element
        for element in metakat_io.elements
        if isinstance(element, MetakatPage)
    )
    assert page.imageDim == MetakatPageDimensions(width=320, height=480)
    assert page.altoDim == MetakatPageDimensions(width=320, height=480)
    serialized_page = page.model_dump(mode="json")
    assert serialized_page["imageDim"] == {"width": 320.0, "height": 480.0}
    assert serialized_page["altoDim"] == {"width": 320.0, "height": 480.0}


def test_init_io_refreshes_dimensions_on_existing_pages(tmp_path):
    Image.new("RGB", (640, 960)).save(tmp_path / "page.jpg")
    _write_alto(tmp_path / "page.xml", 640, 960)

    initial, _ = init_io(
        str(tmp_path),
        ordered_image_filenames=["page.jpg"],
    )
    page = next(
        element
        for element in initial.elements
        if isinstance(element, MetakatPage)
    )
    page_id = page.id
    page.imageDim = MetakatPageDimensions(width=1, height=1)
    page.altoDim = MetakatPageDimensions(width=1, height=1)
    input_json = tmp_path / "input.json"
    input_json.write_text(initial.model_dump_json(), encoding="utf-8")

    refreshed, _ = init_io(
        str(tmp_path),
        metakat_data=initial.model_dump(mode="json"),
        ordered_image_filenames=["page.jpg"],
    )

    refreshed_page = next(
        element
        for element in refreshed.elements
        if isinstance(element, MetakatPage)
    )
    assert refreshed_page.id == page_id
    assert refreshed_page.imageDim == MetakatPageDimensions(width=640, height=960)
    assert refreshed_page.altoDim == MetakatPageDimensions(width=640, height=960)
