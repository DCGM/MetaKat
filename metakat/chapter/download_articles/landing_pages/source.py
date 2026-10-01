from metakat.chapter.download_articles.common.crossref import CrossrefSource


class LandingPagesSource(CrossrefSource):
    """Czech journals in Crossref without PDF links, whose article pages name the PDF for Google Scholar.

    Journals of the government list of peer-reviewed periodicals (RVVI 2008, 2015), ERIH PLUS and
    OpenAlex (2026-10) that no other library covers. Articles are catalogued from Crossref by ISSN and
    downloaded from the ``citation_pdf_url`` of their landing page.
    """

    name = "landing_pages"
    landing_pdf = True
    exclude_libraries = ("dml.cz", "journals.muni.cz", "ojs.cvut.cz", "cuni", "agriculturejournals.cz", "upol",
                         "journal_sites")
    issns = (
        ("0001-7213", "1801-7576"),  # Acta Veterinaria Brno
        ("1804-1930",),  # Agris on-line Papers in Economics and Informatics
        ("1803-2451", "1805-9538"),  # Beskydy
        ("2570-7337", "2570-7345"),  # Bulletin Mineralogie Petrologie
        ("1211-6831",),  # Castellologica bohemica (dspace.zcu.cz)
        ("0862-5409",),  # Divadelní revue
        ("0926-3837", "1210-115X", "1212-0014", "2571-421X"),  # Geografie
        ("2336-2197",),  # International Journal of Business and Management
        ("1804-980X",),  # International Journal of Social Sciences
        ("2336-2022",),  # International Journal of Teaching and Education
        ("1211-8109",),  # Kuděj (dspace.zcu.cz)
        ("1804-526X", "1805-8868"),  # Lifelong Learning
        ("1803-8115",),  # Medsoft
        ("1804-753X", "1804-7548"),  # MEMO - Časopis pro orální historii (dspace.zcu.cz)
        ("1211-1635", "2336-5609"),  # Společnost pro církevní právo
    )
