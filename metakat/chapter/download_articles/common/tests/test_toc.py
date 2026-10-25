from metakat.chapter.download_articles.common.toc import Page, TocEntry, Word, alto_words, page_label, start_pages, \
    toc_entries


def _line(y, *parts):
    """Words of one line: (x, text) pairs, each word 20 px high and 12 px per character."""
    return [Word(x, y, x + 12 * len(text), y + 20, text) for x, text in parts]


def test_entries_with_right_aligned_numbers_leaders_and_titles_over_two_lines():
    words = (_line(0, (100, "CONTENTS"))
             + _line(40, (100, "Ion"), (150, "transport"), (290, "systems"), (900, "11"))
             + _line(80, (100, "Prenatal"), (210, "development"), (370, "......"), (450, "25."))
             + _line(120, (100, "Heterogeneity"), (270, "of"), (310, "the"))
             + _line(160, (100, "myocardium"), (900, "31")))
    assert toc_entries(words) == [TocEntry("11", "CONTENTS Ion transport systems"),
                                  TocEntry("25", "Prenatal development"),
                                  TocEntry("31", "Heterogeneity of the myocardium")]


def test_entries_of_two_columns_and_page_ranges():
    words = (_line(0, (100, "Rybářské"), (210, "kursy,"), (300, "str."), (350, "70."),
                   (600, "Ryby"), (660, "zápasící"), (770, "166—170"))
             + _line(40, (100, "Ročník"), (200, "1897"), (600, "Medvědi,"), (710, "str."), (760, "34")))
    assert [(e.number, e.text) for e in toc_entries(words)] == [
        ("70", "Rybářské kursy, str"), ("166", "Ryby zápasící"), ("1897", "Ročník"), ("34", "Medvědi, str")]


def test_alto_words_read_any_alto_version():
    alto = """<alto xmlns="http://www.loc.gov/standards/alto/ns-v2#"><Layout><Page><PrintSpace><TextBlock>
    <TextLine><String CONTENT="Indiáni" HPOS="10" VPOS="20" WIDTH="100" HEIGHT="30"/><SP/>
    <String CONTENT="194" HPOS="900" VPOS="22" WIDTH="40" HEIGHT="28"/></TextLine></TextBlock></PrintSpace></Page>
    </Layout></alto>""".encode()
    words = alto_words(alto)
    assert [(w.text, w.x0, w.y1) for w in words] == [("Indiáni", 10, 50), ("194", 900, 50)]
    assert toc_entries(words) == [TocEntry("194", "Indiáni")]


def test_page_labels_drop_brackets_and_parentheses():
    assert [page_label(n) for n in ("[169]", "(11)", "12.", " 5a ", None, "")] == ["169", "11", "12", "5a", None, None]


def test_start_pages_prefer_the_contents_issue_and_skip_ambiguous_numbers():
    # Two issues paginated from 1, a contents page at the end of each, and a volume index.
    pages = [Page(f"i1p{n}", str(n), "issue1", "NormalPage") for n in range(1, 6)]
    pages += [Page("i1toc", "(6)", "issue1", "TableOfContents")]
    pages += [Page(f"i2p{n}", f"({n})" if n == 3 else str(n), "issue2", None) for n in range(1, 6)]
    pages += [Page("i2toc", "6", "issue2", "tableOfContents"), Page("index", "7", "volume", "TableOfContents")]
    contents = [
        (pages[5], [TocEntry("1", "A"), TocEntry("4", "B"), TocEntry("6", "the contents itself")]),
        (pages[11], [TocEntry("3", "C"), TocEntry("99", "no such page")]),
        (pages[12], [TocEntry("2", "ambiguous in the volume"), TocEntry("4", "B again")]),
    ]
    starts = start_pages(contents, pages)
    assert [(s.page.pid, [e.text for e in s.entries], s.span) for s in starts] == [
        ("i1p1", ["A"], 3), ("i1p4", ["B"], 5), ("i2p3", ["C"], None)]
    assert starts[0].contents_pages == ["i1toc"]
