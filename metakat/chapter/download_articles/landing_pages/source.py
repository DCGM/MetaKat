import re

from metakat.chapter.download_articles.common.crossref import CrossrefSource


class LandingPagesSource(CrossrefSource):
    """Czech journals in Crossref without PDF links, whose article pages name the PDF for Google Scholar.

    Journals of the government list of peer-reviewed periodicals (RVVI 2008, 2015), ERIH PLUS and
    OpenAlex (2026-10) that no other library covers. Articles are catalogued from Crossref by ISSN and
    downloaded from the ``citation_pdf_url`` of their landing page, or from the landing page itself
    when it is the PDF.
    """

    name = "landing_pages"
    landing_pdf = True
    # Article records of the UJEP repository link the file on the journal's site; the WordPress sites
    # of Enigma Corporation link their own file among cited works.
    landing_pdf_link = re.compile(
        r"https?://(?:ab\.ff\.ujep\.cz/files|(?:eppd13|eujem|ephd)\.cz/wp-content/uploads)/[^\"'\s<>]+\.pdf", re.I)
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
        # Registered with the PDF itself as the landing page
        ("1803-408X", "2571-0273"),  # Acta Facultatis Philosophicae Universitatis Ostraviensis Studia Germanistica
        ("2464-6741",),  # Akademická psychologie (sociosphera.com)
        ("2464-675X",),  # Aktuální pedagogika (sociosphera.com)
        ("2464-580X",),  # Ekonomické trendy (sociosphera.com)
        ("2695-0243",),  # European Scientific e-Journal (eiid.eu)
        ("2464-6768",),  # Filologické vědomosti (sociosphera.com)
        ("0862-495X", "1802-5307"),  # Klinická onkologie
        ("2787-9496",),  # Klironomy Journal (eiid.eu)
        ("1803-8174", "2571-0257"),  # Ostrava Journal of English Philology
        ("2336-2642",),  # Paradigmata poznání (sociosphera.com)
        ("1803-9278", "1805-9023"),  # Psychologie a její kontexty
        ("2464-6776",),  # Sociologie člověka (sociosphera.com)
        ("1803-6406", "2571-0265"),  # Studia Romanistica
        ("1803-5663", "2571-0281"),  # Studia Slavica
        ("2788-0699",),  # Tuculart Student Scientific (eiid.eu)
        # PDF linked from the landing page (landing_pdf_link)
        ("1802-6419", "2570-916X"),  # Aussiger Beiträge (UJEP repository)
        ("2533-4794", "2533-4808"),  # European Journal of Economics and Management (Enigma Corporation)
        ("2336-5439", "2336-5447"),  # European Political and Law Discourse (Enigma Corporation)
        ("2533-4816", "2533-4824"),  # The European philosophical and historical discourse (Enigma Corporation)
    )
