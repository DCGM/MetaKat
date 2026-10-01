from metakat.chapter.download_articles.common.ojs import OjsSource


class OjsSitesSource(OjsSource):
    """Czech journals on their own Open Journal Systems, whose Crossref records lack PDF links.

    Found among the Czech journals of the Directory of Open Access Journals (2026-10). GeoScience
    Engineering (geoscience.cz) is left out: its OAI-PMH answers with server errors.
    """

    name = "ojs_sites"
    oai_urls = (
        "https://www.eriesjournal.com/index.php/eries/oai",   # Journal on Efficiency and Responsibility in Education and Science
        "https://acm.kme.zcu.cz/acm/oai",                     # Applied and Computational Mechanics
        "https://www.tape.academy/index.php/tape/oai",        # Theology and Philosophy of Education
        "http://www.ijates.org/index.php/ijates/oai",         # Int. Journal of Advances in Telecommunications, ...
        "http://www.antropoweb.cz/webzin/index.php/webzin/oai",  # AntropoWebzin
        "https://aimt.cz/index.php/aimt/oai",                 # Advances in Military Technology
        "https://stuter.fsv.cuni.cz/stuter/oai",              # AUC Studia Territorialia
    )
