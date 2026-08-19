from metakat.io_parsers.parser_mods import parse_mods


def _wrap(mods_body: str, version: str = "3.5") -> str:
    return f"""<mods:modsCollection xmlns:mods="http://www.loc.gov/mods/v3">
  <mods:mods version="{version}">
    {mods_body}
  </mods:mods>
</mods:modsCollection>"""


def test_primary_title_wins_over_alternative():
    xml = _wrap("""
    <mods:titleInfo type="alternative">
      <mods:title>Alt Title</mods:title>
    </mods:titleInfo>
    <mods:titleInfo>
      <mods:title>Estetika</mods:title>
      <mods:subTitle>casopis pro estetiku</mods:subTitle>
    </mods:titleInfo>
    """)
    result = parse_mods(xml)
    assert result["title"] == "Estetika"
    assert result["subTitle"] == "casopis pro estetiku"


def test_roles_are_mapped_including_photographer_and_unmapped_role_is_dropped():
    xml = _wrap("""
    <mods:name type="personal" usage="primary">
      <mods:namePart type="family">Novak</mods:namePart>
      <mods:namePart type="given">Jan</mods:namePart>
      <mods:role><mods:roleTerm authority="marcrelator" type="code">aut</mods:roleTerm></mods:role>
    </mods:name>
    <mods:name type="personal">
      <mods:namePart type="family">Svoboda</mods:namePart>
      <mods:role><mods:roleTerm authority="marcrelator" type="code">pht</mods:roleTerm></mods:role>
    </mods:name>
    <mods:name type="personal">
      <mods:namePart type="family">Kubat</mods:namePart>
      <mods:role><mods:roleTerm authority="marcrelator" type="code">oth</mods:roleTerm></mods:role>
    </mods:name>
    """)
    result = parse_mods(xml)
    assert result["author"] == ["Novak Jan"]
    assert result["photographer"] == ["Svoboda"]
    assert result["editor"] is None
    assert result["translator"] is None


def test_manufacture_origin_info_kept_separate_from_publication():
    xml = _wrap("""
    <mods:originInfo eventType="publication">
      <mods:place><mods:placeTerm type="text">Praha</mods:placeTerm></mods:place>
      <mods:publisher>Academia</mods:publisher>
      <mods:dateIssued>1964</mods:dateIssued>
    </mods:originInfo>
    <mods:originInfo eventType="manufacture">
      <mods:place><mods:placeTerm type="text">Brno</mods:placeTerm></mods:place>
      <mods:publisher>Tiskarna XY</mods:publisher>
      <mods:edition>Manufacture-only edition</mods:edition>
    </mods:originInfo>
    """)
    result = parse_mods(xml)
    assert result["placeTerm"] == ["Praha"]
    assert result["dateIssued"] == ["1964"]
    assert result["publisher"] == ["Academia"]
    assert result["edition"] == [None]
    assert result["manufacturePublisher"] == ["Tiskarna XY"]
    assert result["manufacturePlaceTerm"] == ["Brno"]


def test_multi_era_origin_info_stays_index_aligned_with_none_for_missing_values():
    xml = _wrap("""
    <mods:originInfo eventType="publication">
      <mods:place><mods:placeTerm type="text">Praha</mods:placeTerm></mods:place>
      <mods:publisher>Academia</mods:publisher>
      <mods:dateIssued>1964-19xx</mods:dateIssued>
    </mods:originInfo>
    <mods:originInfo eventType="publication">
      <mods:dateIssued>2008-</mods:dateIssued>
      <mods:edition>New series</mods:edition>
    </mods:originInfo>
    <mods:originInfo>
      <mods:issuance>continuing</mods:issuance>
    </mods:originInfo>
    """)
    result = parse_mods(xml)
    assert result["publisher"] == ["Academia", None]
    assert result["placeTerm"] == ["Praha", None]
    assert result["dateIssued"] == ["1964-19xx", "2008-"]
    assert result["edition"] == [None, "New series"]


def test_series_name_and_number_from_related_item():
    xml = _wrap("""
    <mods:relatedItem type="series">
      <mods:titleInfo>
        <mods:title>Edice XY</mods:title>
        <mods:partNumber>12</mods:partNumber>
      </mods:titleInfo>
    </mods:relatedItem>
    """)
    result = parse_mods(xml)
    assert result["seriesName"] == ["Edice XY"]
    assert result["seriesNumber"] == ["12"]


def test_placeholder_values_are_skipped():
    xml = _wrap("""
    <mods:originInfo eventType="publication">
      <mods:publisher>bez nakladatele</mods:publisher>
      <mods:place><mods:placeTerm type="text">neuveden</mods:placeTerm></mods:place>
    </mods:originInfo>
    """)
    result = parse_mods(xml)
    assert result["publisher"] is None
    assert result["placeTerm"] is None


def test_empty_document_returns_all_none():
    xml = _wrap("")
    result = parse_mods(xml)
    assert all(value is None for value in result.values())
