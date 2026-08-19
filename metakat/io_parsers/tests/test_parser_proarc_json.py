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


# Reading is best effort: the parser is the only gate for ProArc input, and a
# ProArc document that cannot be read must never stop the batch. It returns
# None instead of raising, and never returns a package with nothing in it.


def _package(*objects, package_type="monograph"):
    return {"type": package_type, "objects": list(objects)}


def _object(pid, metadata=None, model="volume"):
    return {
        "pid": pid,
        "model": model,
        "metadata": _mods(title="Some Book") if metadata is None else metadata,
    }


def test_document_that_is_not_a_proarc_json_reads_as_none(caplog):
    assert parse_proarc_json({"foo": "bar"}) is None
    assert "cannot be attempted" in caplog.text


def test_pid_rejected_by_the_schema_makes_the_whole_document_malformed(caplog):
    # The ^uuid: pattern is enforced by ProarcIO itself, so a pid that fails it
    # is a validation error for the document, not a per-record problem.
    assert parse_proarc_json(_package(_object("nonsense"))) is None
    assert "cannot be attempted" in caplog.text


def test_object_whose_pid_is_not_a_uuid_is_dropped(caplog):
    # The pattern only guarantees the prefix; the remainder still has to be a
    # real UUID, and without an id the record cannot be placed in a hierarchy.
    good = uuid4()
    package = parse_proarc_json(
        _package(_object(f"uuid:{good}"), _object("uuid:not-a-uuid"))
    )

    assert [obj.id for obj in package.objects] == [good]
    assert "carries no valid UUID" in caplog.text
    assert "Read 1 of 2" in caplog.text


def test_document_whose_every_object_is_unusable_reads_as_none(caplog):
    assert parse_proarc_json(_package(_object("uuid:not-a-uuid"))) is None
    assert "No usable record could be read" in caplog.text


def test_document_without_objects_reads_as_none(caplog):
    # Valid, but it offers nothing, so the engines must not receive it.
    assert parse_proarc_json(_package()) is None
    assert "No usable record could be read" in caplog.text


def test_object_with_unreadable_mods_is_kept_with_its_identity(caplog):
    object_uuid = uuid4()
    package = parse_proarc_json(
        _package(_object(f"uuid:{object_uuid}", metadata="not xml at all"))
    )

    obj = package.objects[0]
    assert obj.id == object_uuid
    assert obj.title is None
    assert "Could not read the MODS metadata" in caplog.text


def test_object_with_empty_mods_is_kept_with_its_identity():
    object_uuid = uuid4()
    package = parse_proarc_json(_package(_object(f"uuid:{object_uuid}", metadata="")))

    assert package.objects[0].id == object_uuid
    assert package.objects[0].title is None


def test_readable_records_survive_an_unreadable_neighbour():
    good, broken = uuid4(), uuid4()
    package = parse_proarc_json(
        _package(
            _object(f"uuid:{good}", metadata=_mods(title="Kytice")),
            _object(f"uuid:{broken}", metadata="<mods><unclosed>"),
        )
    )

    assert [obj.id for obj in package.objects] == [good, broken]
    assert package.objects[0].title == ["Kytice"]
    assert package.objects[1].title is None
