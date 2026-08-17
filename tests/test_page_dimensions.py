import tempfile
import unittest
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


class PageDimensionsTest(unittest.TestCase):
    def test_init_io_loads_image_and_alto_dimensions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            batch_dir = Path(temporary_directory)
            Image.new("RGB", (320, 480)).save(batch_dir / "page.jpg")
            _write_alto(batch_dir / "page.xml", 320, 480)

            metakat_io, _ = init_io(
                str(batch_dir),
                ordered_image_filenames=["page.jpg"],
            )

        page = next(
            element
            for element in metakat_io.elements
            if isinstance(element, MetakatPage)
        )
        self.assertEqual(
            page.imageDim,
            MetakatPageDimensions(width=320, height=480),
        )
        self.assertEqual(
            page.altoDim,
            MetakatPageDimensions(width=320, height=480),
        )
        serialized_page = page.model_dump(mode="json")
        self.assertEqual(
            serialized_page["imageDim"],
            {"width": 320.0, "height": 480.0},
        )
        self.assertEqual(
            serialized_page["altoDim"],
            {"width": 320.0, "height": 480.0},
        )

    def test_init_io_refreshes_dimensions_on_existing_pages(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            batch_dir = Path(temporary_directory)
            Image.new("RGB", (640, 960)).save(batch_dir / "page.jpg")
            _write_alto(batch_dir / "page.xml", 640, 960)

            initial, _ = init_io(
                str(batch_dir),
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
            input_json = batch_dir / "input.json"
            input_json.write_text(initial.model_dump_json(), encoding="utf-8")

            refreshed, _ = init_io(
                str(batch_dir),
                metakat_data=initial.model_dump(mode="json"),
                ordered_image_filenames=["page.jpg"],
            )

        refreshed_page = next(
            element
            for element in refreshed.elements
            if isinstance(element, MetakatPage)
        )
        self.assertEqual(refreshed_page.id, page_id)
        self.assertEqual(
            refreshed_page.imageDim,
            MetakatPageDimensions(width=640, height=960),
        )
        self.assertEqual(
            refreshed_page.altoDim,
            MetakatPageDimensions(width=640, height=960),
        )


if __name__ == "__main__":
    unittest.main()
