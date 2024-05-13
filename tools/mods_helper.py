import xml.etree.ElementTree as ET


def get_year_from_doc_mods(mods_path):
    namespaces = {'mods': "http://www.loc.gov/mods/v3",
                  'mets': "http://www.loc.gov/METS/"}
    tree = ET.parse(mods_path)
    root = tree.getroot()
    date_element = root.findall(f".//mods:namePart[@type='date']", namespaces=namespaces)
    date = None
    if len(date_element) != 0:
        date = date_element[0].text
    if date is None:
        issued_element = root.findall(f".//mods:dateIssued", namespaces=namespaces)
        if len(issued_element) != 0:
            date = issued_element[0].text
    if date is None:
        return None
    # [] - for dates like [1938]
    # ? - for dates like 1938?
    start_end_year = [x.strip('[]?') for x in date.split('-')]
    try:
        int(start_end_year[0])
    except ValueError:
        return None
    try:
        int(start_end_year[1])
    except (ValueError, IndexError):
        return int(start_end_year[0]), int(start_end_year[0])
    return int(start_end_year[0]), int(start_end_year[1])


page_type_classes = ('TitlePage,Table,TableOfContents,Index,Jacket,FrontEndSheet,FrontCover,BackEndSheet,BackCover,'
                     'Blank,SheetMusic,Advertisement,Map,FrontJacket,FlyLeaf,ListOfIllustrations,Illustration,Spine,'
                     'CalibrationTable,Cover,Edge,ListOfTables,FrontEndPaper,BackEndPaper,ListOfMaps,Bibliography,'
                     'CustomInclude,Frontispiece,Errata,FragmentsOfBookbinding,BackEndPaper,FrontEndPaper,Preface,'
                     'Abstract,Dedication,Imprimatur,Impressum,Obituary,Appendix,NormalPage')
page_type_classes = page_type_classes.split(',')
page_type_classes = {x.lower(): x for x in page_type_classes}


def get_page_type_from_page_mods(mods_path):
    namespaces = {'mods': "http://www.loc.gov/mods/v3",
                  'mets': "http://www.loc.gov/METS/"}
    tree = ET.parse(mods_path)
    root = tree.getroot()
    page_type_element = root.findall(f".//mods:part[@type]", namespaces=namespaces)
    page_type = page_type_element[0].get('type')
    return page_type_classes[page_type.lower()]
