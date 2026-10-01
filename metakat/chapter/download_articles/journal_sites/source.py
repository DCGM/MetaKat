from metakat.chapter.download_articles.common.crossref import CrossrefSource


class JournalSitesSource(CrossrefSource):
    """Czech journals published on their own sites, found in Crossref by their ISSNs.

    The Czech journals of the Directory of Open Access Journals (2026-10) that are neither in the
    libraries and platforms above nor without PDF links in Crossref. Articles whose DOI another
    library holds are left to it.
    """

    name = "journal_sites"
    exclude_libraries = ("dml.cz", "journals.muni.cz", "ojs.cvut.cz", "cuni", "agriculturejournals.cz", "upol")
    issns = (
        ("2571-0613", "1803-9782"),  # ACC Journal
        ("2336-6346", "1802-0364"),  # Acta Fakulty filozofické Západočeské univerzity v Plzni
        ("1805-4951",),  # Acta Informatica Pragensia
        ("2464-8310", "1211-8516"),  # Acta Universitatis Agriculturae et Silviculturae Mendelianae Brunensis
        ("1804-3119", "1336-1376"),  # Advances in Electrical and Electronic Engineering
        ("2788-2233", "1803-6058"),  # American and British Studies Annual
        ("1573-8264",),  # Biologia Plantarum
        ("2570-7434", "1805-3777"),  # Business & IT
        ("1805-482X", "1802-548X"),  # Central European Journal of International & Security Studies
        ("2336-3517",),  # Central European Journal of Nursing and Midwifery
        ("2464-479X",),  # Central European Journal of Politics
        ("2570-9429", "0323-1844"),  # Czech Journal of International Relations
        ("1821-2506",),  # DETUROPE
        ("1211-0442",),  # E-Logos
        ("1804-8358", "1801-0865"),  # Echo des Etudes Romanes
        ("2694-7161", "2336-6494"),  # European Journal of Business Science and Technology
        ("1804-0969",),  # Filosofie Dnes
        ("1802-5420",),  # GeoScience Engineering
        ("2336-680X", "1804-4913"),  # Historia Scholastica
        ("2336-2960",),  # International Journal of Entrepreneurial Knowledge
        ("1214-0287", "1214-021X"),  # Journal of Applied Biomedicine
        ("1804-1728", "1804-171X"),  # Journal of Competitiveness
        ("1804-7122", "1212-4117"),  # Kontakt
        ("2570-8619",),  # Kvasný průmysl
        ("2570-6179",),  # Listy klinické logopedie
        ("2571-3701", "1803-3814"),  # Mendel
        ("2570-7558", "2336-3274"),  # Modern Africa
        ("1802-7199", "1214-6463"),  # Obrana a strategie
        ("1801-674X",),  # Perner's Contacts
        ("1805-9600", "1210-2512"),  # Radioengineering
        ("2788-3809", "1210-8545"),  # Romano Džaniben
        ("1805-8825",),  # Sociální pedagogika
        ("2571-0621", "1802-2502"),  # Theatrum Historiae
        ("1804-0993",),  # Transactions of the VŠB - Technical University of Ostrava, Mechanical Series
        ("1804-4824",),  # Transactions of the VŠB - Technical University of Ostrava, Civil Engineering Series
        ("2336-2995", "1210-3292"),  # Vojenské rozhledy
        ("1805-4471", "1213-0613"),  # Česká stomatologie a praktické zubní lékařství
    )
