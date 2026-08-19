from metakat.io_parsers.parser_proarc_json import parse_proarc_package
from metakat.schemas.base_objects import DocumentType, HierarchyType, MetakatIssue, MetakatTitle, MetakatVolume


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


def test_periodical_builds_title_and_issue_with_parent_link():
    data = {
        "type": "periodical",
        "objects": [
            {"pid": "uuid:1", "model": "title", "metadata": _mods(title="Estetika")},
            {"pid": "uuid:2", "model": "volume", "metadata": _mods(part_number="38", date_issued="2002")},
        ],
    }
    io = parse_proarc_package(data)
    assert io.elements[0].type == DocumentType.TITLE.value
    assert isinstance(io.elements[0], MetakatTitle)
    assert io.elements[0].hierarchy == HierarchyType.PERIODICAL.value

    issue = io.elements[1]
    assert isinstance(issue, MetakatIssue)
    assert issue.type == DocumentType.ISSUE.value
    assert issue.parent_id == io.elements[0].id
    assert issue.partNumber[0] == "38"
    assert issue.dateIssued[0] == "2002"


def test_monograph_without_title_is_plain_monograph_volume():
    data = {
        "type": "monograph",
        "objects": [
            {"pid": "uuid:1", "model": "volume", "metadata": _mods(title="Some Book")},
        ],
    }
    io = parse_proarc_package(data)
    assert len(io.elements) == 1
    volume = io.elements[0]
    assert isinstance(volume, MetakatVolume)
    assert volume.hierarchy == HierarchyType.MONOGRAPH.value
    assert volume.parent_id is None
    assert volume.title[0] == "Some Book"


def test_monograph_with_title_is_multipart():
    data = {
        "type": "monograph",
        "objects": [
            {"pid": "uuid:1", "model": "title", "metadata": _mods(title="Amerika")},
            {"pid": "uuid:2", "model": "volume", "metadata": _mods(part_number="1")},
            {"pid": "uuid:3", "model": "volume", "metadata": _mods(part_number="2")},
        ],
    }
    io = parse_proarc_package(data)
    title = next(e for e in io.elements if e.type == DocumentType.TITLE.value)
    volumes = [e for e in io.elements if e.type == DocumentType.VOLUME.value]
    assert title.hierarchy == HierarchyType.MULTIPART.value
    assert len(volumes) == 2
    for volume in volumes:
        assert volume.hierarchy == HierarchyType.MULTIPART.value
        assert volume.parent_id == title.id
