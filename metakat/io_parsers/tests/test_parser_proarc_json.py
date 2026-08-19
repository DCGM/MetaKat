from uuid import uuid4

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
    title_uuid, volume_uuid = uuid4(), uuid4()
    data = {
        "type": "periodical",
        "objects": [
            {"pid": f"uuid:{title_uuid}", "model": "title", "metadata": _mods(title="Estetika")},
            {"pid": f"uuid:{volume_uuid}", "model": "volume", "metadata": _mods(part_number="38", date_issued="2002")},
        ],
    }
    package = parse_proarc_json(data)

    assert isinstance(package, ProarcIO)
    assert package.type == "periodical"

    title_obj, volume_obj = package.objects
    assert title_obj.pid == f"uuid:{title_uuid}"
    assert title_obj.id == title_uuid
    assert title_obj.title == ["Estetika"]

    assert volume_obj.pid == f"uuid:{volume_uuid}"
    assert volume_obj.id == volume_uuid
    assert volume_obj.partNumber == ["38"]
    assert volume_obj.dateIssued == ["2002"]


def test_raw_metadata_string_is_preserved_alongside_parsed_fields():
    object_uuid = uuid4()
    data = {
        "type": "monograph",
        "objects": [
            {"pid": f"uuid:{object_uuid}", "model": "volume", "metadata": _mods(title="Some Book")},
        ],
    }
    package = parse_proarc_json(data)
    obj = package.objects[0]
    assert obj.id == object_uuid
    assert obj.title == ["Some Book"]
    assert "<mods:title>Some Book</mods:title>" in obj.metadata
