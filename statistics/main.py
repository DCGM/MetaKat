import argparse
import sys

from collections import defaultdict
from lxml import etree
from natsort import natsorted


def parseargs():
    print(' '.join(sys.argv))
    parser = argparse.ArgumentParser()

    parser.add_argument('--mets', action='append', type=str)

    parser.add_argument('--page', action='store_true')
    parser.add_argument('--title', action='store_true')
    parser.add_argument('--issue', action='store_true')
    parser.add_argument('--article', action='store_true')
    parser.add_argument('--chapter', action='store_true')


    args = parser.parse_args()
    return args


def main():
    args = parseargs()

    if args.page:
        get_page_statistics(args.mets)
    if args.title:
        get_title_statistics(args.mets)
    if args.issue:
        get_issue_statistics(args.mets)
    if args.article:
        get_article_statistics(args.mets)
    if args.chapter:
        get_chapter_statistics(args.mets)


def get_page_statistics(mets):
    namespaces = {'mods': "http://www.loc.gov/mods/v3"}

    stats_page_types = defaultdict(lambda: 0)
    stats_page_notes = defaultdict(lambda: 0)
    stats_page_numbers = defaultdict(lambda: 0)
    pages = 0

    print()
    print()

    for met in mets:
        met_stats_page_types = defaultdict(lambda: 0)
        met_stats_page_notes = defaultdict(lambda: 0)
        met_stats_page_numbers = defaultdict(lambda: 0)
        met_pages = 0

        tree = etree.parse(met)
        mods_page_elements = tree.xpath("//mods:mods[contains(@ID, 'PAGE')]",
                                        namespaces=namespaces)

        for mods_page_element in mods_page_elements:
            pages += 1
            met_pages += 1

            page_type = mods_page_element.xpath("mods:genre/@type",
                                                namespaces=namespaces)
            if page_type:
                page_type = lower_first_char(page_type[0])
                stats_page_types[page_type] += 1
                met_stats_page_types[page_type] += 1

            page_note = mods_page_element.xpath("mods:note/text()",
                                                namespaces=namespaces)
            if page_note:
                page_note = page_note[0]
                stats_page_notes[page_note] += 1
                met_stats_page_notes[page_note] += 1

            page_number = mods_page_element.xpath("mods:part/mods:detail[@type='pageNumber']/mods:number/text()",
                                                  namespaces=namespaces)
            if page_number:
                page_number = page_number[0]
                if page_number.startswith('['):
                    stats_page_numbers['invalid_page_number'] += 1
                    met_stats_page_numbers['invalid_page_number'] += 1
                else:
                    try:
                        int(page_number)
                    except:
                        print(met)
                    stats_page_numbers['valid_page_number'] += 1
                    met_stats_page_numbers['valid_page_number'] += 1

        print()
        print("MET FILE:", met)
        print("ELEMENTS:", met_pages)
        print()
        print()
        print("TYPE:")
        print()
        print(print_dict(met_stats_page_types))
        print()
        print("NOTE:")
        print()
        print(print_dict(met_stats_page_notes))
        print()
        print("NUMBER:")
        print()
        print(print_dict(met_stats_page_numbers))
        print()
        print()

    print()
    print()
    print()
    print('ALL STATS')
    print()
    print()

    print("METS:", len(mets))
    print("ELEMENTS:", pages)

    print()
    print("TYPE:")
    print()
    print_dict(stats_page_types)

    print()
    print("NOTE:")
    print()
    print_dict(stats_page_notes)

    print()
    print("NUMBER:")
    print()
    print_dict(stats_page_numbers)

    return stats_page_types, stats_page_notes, stats_page_numbers


def get_title_statistics(mets):
    return get_statistics(mets,
                          'TITLE',
                          ['title', 'subTitle', 'publisher'])


def get_issue_statistics(mets):
    return get_statistics(mets,
                          'ISSUE',
                          ['title', 'partNumber', 'placeTerm', 'publisher', 'dateIssued', 'languageTerm'])


def get_article_statistics(mets):
    return get_statistics(mets,
                          'ART',
                          ['startPageNumber',
                           'startPageIndex',
                           'title', 'subTitle', 'languageTerm',
                           'familyName',
                           'givenName'],
                          ["mods:part[@type='pageNumber']/mods:extent[@unit='pages']/mods:start/text()",
                           "mods:part[@type='pageIndex']/mods:extent[@unit='pages']/mods:start/text()",
                           None, None, None,
                           "mods:name/mods:namePart[@type='family']/text()",
                           "mods:name/mods:namePart[@type='given']/text()"])

def get_chapter_statistics(mets):
    return get_statistics(mets,
                          'CHAPTER',
                          ['title'])


def get_statistics(mets, main_element_id, tags, xpaths=None):
    namespaces = {'mods': "http://www.loc.gov/mods/v3"}

    stats = {}
    for tag in tags:
        stats[tag] = 0
    elements = 0

    print()
    print()

    for met in mets:
        met_stats = {}
        for tag in tags:
            met_stats[tag] = 0
        met_elements = 0

        tree = etree.parse(met)
        main_elements = tree.xpath(f"//mods:mods[contains(@ID, '{main_element_id}')]",
                                   namespaces=namespaces)

        for main_element in main_elements:
            elements += 1
            met_elements += 1

            for i, tag in enumerate(tags):
                if xpaths is None or xpaths[i] is None:
                    xpath = f"//mods:{tag}/text()"
                else:
                    xpath = xpaths[i]
                element_content = main_element.xpath(xpath,
                                                     namespaces=namespaces)
                if element_content and element_content[0]:
                    stats[tag] += 1
                    met_stats[tag] += 1

        print()
        print("MET FILE:", met)
        print("ELEMENTS:", met_elements)
        print()
        print()
        print_dict(met_stats)
        print()
        print()

    print()
    print()
    print()
    print('ALL STATS')
    print()
    print()

    print("METS:", len(mets))
    print("ELEMENTS:", elements)

    print()
    print_dict(stats)

    return stats


def print_dict(d):
    for k, v in natsorted(d.items(), key=lambda x: x[0].lower()):
        print("{}: {}".format(k, v))


def print_lxml_element(e):
    print(etree.tostring(e, pretty_print=True))


def lower_first_char(s):
    if s:
        return s[:1].lower() + s[1:]
    return ''


if __name__ == '__main__':
    main()

