from metakat.chapter.download_articles.common.crossref import CrossrefSource


class JournalSitesSource(CrossrefSource):
    """Czech journals published on their own sites, found in Crossref by their ISSNs.

    The Czech journals of the Directory of Open Access Journals (2026-10), then those of the government
    list of peer-reviewed periodicals, ERIH PLUS and OpenAlex, that are neither in the libraries and
    platforms above nor without PDF links in Crossref. Articles whose DOI another library holds are
    left to it.
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
        # Journals of the government list of peer-reviewed periodicals (RVVI 2008, 2015), ERIH PLUS (2023)
        # and OpenAlex (country CZ) with PDF links in Crossref (2026-10)
        ("1212-415X", "2533-7610"),  # Acta academica karviniensia
        ("1804-2732",),  # Acta Carpathica Occidentalis
        ("0001-5415", "2570-981X"),  # Acta chirurgiae orthopaedicae et traumatologiae Cechoslovaca
        ("0862-8548",),  # Acta musealia
        ("1803-960X",),  # Acta Musei Beskidensis
        ("0572-3043", "1804-2112"),  # Acta Oeconomica Pragensia
        ("1805-8787",),  # Acta Salus Vitae
        ("1212-3285", "2336-4297"),  # Acta Universitatis Bohemiae Meridionalis
        ("1214-2158", "1805-4412"),  # Anesteziologie a intenzivní medicína
        ("2570-5903", "2570-5911"),  # Biological Markers in Fundamental and Clinical Medicine (collection of abstracts)
        ("1212-4923", "2336-1956"),  # Cargo Journal
        ("1805-0948",),  # Caritas et Veritas
        ("1805-4854", "1805-4862"),  # Central European Business Review
        ("1210-7778", "1803-1048"),  # Central European Journal of Public Health
        ("1802-4866",),  # Central European Journal of Public Policy
        ("2336-3312", "2336-369X"),  # Central European Papers
        ("0010-8650", "1803-7712"),  # Cor et Vasa
        ("2336-7148",),  # Czech Journal of Civil Engineering
        ("1211-8729",),  # Czech Urology
        ("1802-2960", "1803-5337"),  # Dermatologie pro praxi
        ("2570-7612",),  # DIAGNOSTIKA A PORADENSTVÍ v pomáhajících profesích
        ("1212-3609", "2336-5064"),  # E+M Ekonomie a Management
        ("1805-1944", "2695-1622"),  # e-Monumentica
        ("2695-0936",),  # EduPort
        ("1212-3951", "1805-9481"),  # Ekonomická revue - Central European Review of Economic Issues
        ("2571-1040",),  # Entecho
        ("1802-2197", "1805-4846"),  # European Financial and Accounting Journal
        ("1804-5839", "1804-9699"),  # European Journal of Business and Economics
        ("1804-5804", "1804-9702"),  # European Medical Health and Pharmaceutical Journal
        ("1213-5097",),  # Fontes Nissae
        ("1804-7874", "1804-803X"),  # Gastroenterologie a hepatologie
        ("2788-0931",),  # Health & Caring
        ("2788-0702", "2788-0710"),  # Historia Aperta
        ("0862-397X", "2570-9267"),  # Iluminace
        ("1804-9796",),  # International Journal of Economic Sciences
        ("2533-4077",),  # International Journal of Public Administration Management and Economic Development
        ("1212-7299", "1803-5256"),  # Interní medicína pro praxi
        ("1213-807X", "1803-5302"),  # Intervenční a akutní kardiologie
        ("1804-1868", "1804-7181"),  # Journal of Nursing Social Studies Public Health and Rehabilitation
        ("1804-5650",),  # Journal of Tourism and Services
        ("1213-6964", "1213-6972", "1213-6980"),  # Journal of WSCG
        ("1210-7921",),  # Klinická biochemie a metabolismus
        ("1212-7973", "1803-5353"),  # Klinická farmakologie a farmacie
        ("1336-6939",),  # Malacologica Bohemoslovaca
        ("1213-2489", "2787-9402"),  # MANUFACTURING TECHNOLOGY
        ("1805-3610", "1805-3629"),  # Mathematics for Applications
        ("1214-8687", "1803-5310"),  # Medicína pro praxi
        ("0372-7025", "2571-113X"),  # Military Medical Science Letters
        ("1213-1814", "1803-5280"),  # Neurologie pro praxi
        ("1801-0938", "1804-6290"),  # New Perspectives on Political Economy
        ("1803-0785",),  # Oceňování
        ("1802-4475", "1803-5345"),  # Onkologie
        ("2533-7106",),  # Online Journal of Primary and Preschool Education
        ("1801-5964",),  # Open European Journal on Variable stars
        ("1805-790X", "2694-720X"),  # Opera Historica
        ("2570-785X", "2571-0702"),  # Ošetřovatelské perspektivy
        ("0862-1586",),  # Památky středních Čech
        ("1213-0494", "1803-5264"),  # Pediatrie pro praxi
        ("1210-0455", "2336-730X"),  # Prague Economic Papers
        ("1801-2434", "1803-5329"),  # Praktické lékárenství
        ("2695-0693", "2695-0707"),  # Proceedings of CBU in Economics and Business
        ("2695-0731", "2695-074X"),  # Proceedings of CBU in Medicine and Pharmacy
        ("2695-0758", "2695-0766"),  # Proceedings of CBU in Natural Sciences and ICT
        ("2695-0715", "2695-0723"),  # Proceedings of CBU in Social Sciences
        ("1212-1487",),  # Průzkumy památek
        ("1211-3174", "1805-9430"),  # Scientia Agriculturae Bohemica
        ("1211-555X", "1804-8048"),  # Scientific Papers of the University of Pardubice Series D Faculty of Economics and Administration
        ("1804-4158", "1804-9710"),  # Social and Natural Sciences Journal
        ("2464-5877", "2464-5885"),  # Social Pathology and Prevention
        ("1804-6797", "1804-6800"),  # Socio-Economic and Humanities Studies
        ("1211-443X", "2788-2764"),  # Soudní inženýrství
        ("3029-8342",),  # Současné problémy v kolejových vozidlech
        ("0231-6056",),  # Staletá Praha
        ("1213-2101", "2571-0710"),  # Studia Kinanthropologica
        ("1801-1764", "1805-3238"),  # TRANSACTIONS of the VŠB – Technical University of Ostrava Safety Engineering Series
        ("1802-8527", "2336-6508"),  # Trends Economics and Management
        ("1805-0603", "2788-0079"),  # Trendy v podnikání
        ("1213-1768", "1803-5299"),  # Urologie pro praxi
        ("0042-773X", "1801-7592"),  # Vnitřní lékařství
        ("0322-8916", "1805-6555"),  # Vodohospodářské technicko-ekonomické informace
        ("2695-1584", "2695-1592"),  # Věda a perspektivy
        ("1210-5538",),  # Zprávy památkové péče
        ("0069-2328", "1805-4501"),  # Česko-slovenská pediatrie
        ("1210-7816", "1805-4439"),  # Česká a slovenská farmacie
        ("1210-7883",),  # Česká radiologie
        ("1802-2200", "1805-4838"),  # Český finanční a účetní časopis
    )
