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


def test_pid_without_the_uuid_prefix_discards_the_document(caplog):
    assert parse_proarc_json(_package(_object("nonsense"))) is None
    assert "carries no valid UUID" in caplog.text


def test_one_unidentifiable_object_discards_the_whole_document(caplog):
    # A hierarchy with a record missing still looks like a complete one to the
    # engines downstream, so a package is all-or-nothing on identity: the
    # readable neighbour must not survive on its own.
    package = parse_proarc_json(
        _package(_object(f"uuid:{uuid4()}"), _object("uuid:not-a-uuid"))
    )

    assert package is None
    assert "carries no valid UUID" in caplog.text
    assert "discarding the whole document" in caplog.text


def test_document_whose_only_object_is_unusable_reads_as_none(caplog):
    assert parse_proarc_json(_package(_object("uuid:not-a-uuid"))) is None
    assert "carries no valid UUID" in caplog.text


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
    # Unlike identity, a record's catalog fields are not all-or-nothing: an
    # unparseable MODS costs that one record its fields, nothing more.
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


def test_aligned_groups_keep_their_none_placeholders():
    # An index-aligned group holds one entry per source block, so a block with
    # no value for a field occupies its index as None. The model has to accept
    # that, otherwise validating the finished document would reject exactly
    # what the MODS parser produces.
    object_uuid = uuid4()
    metadata = (
        '<mods:modsCollection xmlns:mods="http://www.loc.gov/mods/v3">'
        '<mods:mods version="3.5">'
        '<mods:titleInfo usage="primary"><mods:title>Kytice</mods:title>'
        "</mods:titleInfo>"
        "<mods:titleInfo><mods:partNumber>2</mods:partNumber></mods:titleInfo>"
        "</mods:mods></mods:modsCollection>"
    )

    package = parse_proarc_json(_package(_object(f"uuid:{object_uuid}", metadata)))

    obj = package.objects[0]
    assert obj.title == ["Kytice", None]
    assert obj.partNumber == [None, "2"]


def test_the_parsed_package_is_what_pydantic_validated():
    # The document is assembled first and validated once, so a package coming
    # out of here round-trips through its own model. Filling fields in after
    # validation would not: pydantic does not check assignment, so the model
    # would never have seen them.
    object_uuid = uuid4()
    package = parse_proarc_json(
        _package(_object(f"uuid:{object_uuid}", _mods(title="Kytice")))
    )

    revalidated = ProarcIO.model_validate(package.model_dump(mode="json"))

    assert revalidated == package
    assert revalidated.objects[0].id == object_uuid
