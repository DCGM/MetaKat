from metakat.io_parsers.parser_proarc_json import parse_proarc_json
from metakat.schemas.base_objects import ProarcIO


def _mods(title=None, part_number=None, date_issued=None):
    body = ""
    if title is not None:
        body += f"<mods:titleInfo><mods:title>{title}</mods:title></mods:titleInfo>"
    if part_number is not None:
        body += f"<mods:originInfo eventType='publication'><mods:dateIssued>{date_issued or ''}</mods:dateIssued></mods:originInfo>"
        body += f"<mods:titleInfo><mods:partNumber>{part_number}</mods:partNumber></mods:titleInfo>"
    return f"""<mods:modsCollection xmlns:mods="http://www.loc.gov/mods/v3">
      <mods:mods version="3.5">{body}</mods:mods>
    </mods:modsCollection>"""


def test_returns_proarc_io_with_parsed_fields_filled_in():
    data = {
        "type": "periodical",
        "objects": [
            {"pid": "uuid:1", "model": "title", "metadata": _mods(title="Estetika")},
            {"pid": "uuid:2", "model": "volume", "metadata": _mods(part_number="38", date_issued="2002")},
        ],
    }
    package = parse_proarc_json(data)

    assert isinstance(package, ProarcIO)
    assert package.type == "periodical"

    title_obj, volume_obj = package.objects
    assert title_obj.pid == "uuid:1"
    assert title_obj.title == "Estetika"

    assert volume_obj.pid == "uuid:2"
    assert volume_obj.partNumber == "38"
    assert volume_obj.dateIssued == ["2002"]


def test_raw_metadata_string_is_preserved_alongside_parsed_fields():
    data = {
        "type": "monograph",
        "objects": [
            {"pid": "uuid:1", "model": "volume", "metadata": _mods(title="Some Book")},
        ],
    }
    package = parse_proarc_json(data)
    obj = package.objects[0]
    assert obj.title == "Some Book"
    assert "<mods:title>Some Book</mods:title>" in obj.metadata
