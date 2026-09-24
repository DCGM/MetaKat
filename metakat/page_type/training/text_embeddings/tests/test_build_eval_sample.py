import csv

from metakat.page_type.training.text_embeddings.build_eval_sample import (
    doc_from_path,
    main,
    page_id_from_annotated_name,
)

ID = '00000000-0000-0000-0000-{:012d}'
CSV_HEADER = ('library,periodical_id,volume_id,item_id,page_id,page_type,page_placement,order,number,'
              'date_of_issue,access,image_path\n')


def test_doc_from_path_uses_the_archive_directory_of_images_and_zips():
    assert doc_from_path('/mnt/x/nkp/abc.images/page.jpg') == 'nkp/abc'
    assert doc_from_path('/mnt/x/nkp/abc.page_xml.zip') == 'nkp/abc'


def test_page_id_from_annotated_name_accepts_uuid_names_only():
    page_id = ID.format(1)
    assert page_id_from_annotated_name(f'uuid:{page_id}.jpg') == page_id
    assert page_id_from_annotated_name(f'{page_id}.jpg') == page_id
    assert page_id_from_annotated_name('09708630-07e8-11de-b09a-000d606f5dc6.4384300413.jpg.tif.jp2') is None


def _csv_row(library, doc, page, label, order):
    return f'{library},,,{doc},{page},{label},,{order},,,public,/mnt/x/{library}/{doc}.images/{page}.jpg\n'


def test_sample_splits_caps_and_annotations(tmp_path):
    rows = []
    progress = []
    # doc-a: 15 TitlePage pages, doc-b: 3 Index pages, doc-c: one annotated page and one unlabelled page
    pages = ([('nkp', 'doc-a', ID.format(i), 'TitlePage', i) for i in range(15)]
             + [('nkp', 'doc-b', ID.format(100 + i), 'Index', i) for i in range(3)]
             + [('nkp', 'doc-c', ID.format(200), 'NormalPage', 0),
                ('nkp', 'doc-c', ID.format(201), 'not_found', 1)])
    for library, doc, page, label, order in pages:
        rows.append(_csv_row(library, doc, page, label, order))
        progress.append(f'{library}\t{page}\t/mnt/x/{library}/{doc}.page_xml.zip\tdone\t\n')
    rows.append(_csv_row('nkp', 'doc-d', ID.format(300), 'Index', 0))  # no text: not in progress
    rows.append('broken,row\n')
    progress.append(f'mzk\t{ID.format(400)}\t/mnt/x/mzk/doc-e.page_xml.zip\tdone\t\n')  # annotated, not in CSV

    (tmp_path / 'pages.csv').write_text(CSV_HEADER + ''.join(rows), encoding='utf-8')
    (tmp_path / 'progress.tsv').write_text(''.join(progress), encoding='utf-8')
    (tmp_path / 'annotated').write_text(f'uuid:{ID.format(200)}.jpg TableOfContents\n'
                                        f'{ID.format(400)}.jpg Index\n'
                                        f'legacy.123.jpg.tif.jp2 Blank\n', encoding='utf-8')
    output = tmp_path / 'sample.tsv'
    main(['--labels-csv', str(tmp_path / 'pages.csv'), '--annotated', str(tmp_path / 'annotated'),
          '--extract-progress', str(tmp_path / 'progress.tsv'), '--output', str(output),
          '--per-doc-cap', '10', '--per-class-cap', '100', '--test-fraction', '0'])

    with output.open(encoding='utf-8') as f:
        sample = {row['page_id']: row for row in csv.DictReader(f, delimiter='\t')}

    title_pages = [row for row in sample.values() if row['archive_label'] == 'TitlePage']
    assert len(title_pages) == 10  # per-document cap
    assert all(row['split'] == 'train' and row['doc_pages'] == '15' for row in title_pages)
    assert ID.format(300) not in sample  # no text

    annotated_in_csv = sample[ID.format(200)]
    assert (annotated_in_csv['annotated_label'], annotated_in_csv['archive_label'],
            annotated_in_csv['split']) == ('TableOfContents', 'NormalPage', 'annotated')
    annotated_outside_csv = sample[ID.format(400)]
    assert (annotated_outside_csv['key'], annotated_outside_csv['doc'], annotated_outside_csv['order'],
            annotated_outside_csv['split']) == (f'mzk_{ID.format(400)}', 'mzk/doc-e', '', 'annotated')

    unlabelled = sample[ID.format(201)]
    assert (unlabelled['archive_label'], unlabelled['split']) == ('', 'background')
    assert all(row['split'] == 'train' for row in sample.values() if row['archive_label'] == 'Index'
               and row['doc'] == 'nkp/doc-b')


def test_documents_with_annotated_pages_go_to_the_test_side(tmp_path):
    rows = [_csv_row('nkp', 'doc-a', ID.format(i), 'TitlePage', i) for i in range(3)]
    progress = [f'nkp\t{ID.format(i)}\t/mnt/x/nkp/doc-a.page_xml.zip\tdone\t\n' for i in range(3)]
    (tmp_path / 'pages.csv').write_text(CSV_HEADER + ''.join(rows), encoding='utf-8')
    (tmp_path / 'progress.tsv').write_text(''.join(progress), encoding='utf-8')
    (tmp_path / 'annotated').write_text(f'uuid:{ID.format(0)}.jpg TitlePage\n', encoding='utf-8')
    output = tmp_path / 'sample.tsv'
    main(['--labels-csv', str(tmp_path / 'pages.csv'), '--annotated', str(tmp_path / 'annotated'),
          '--extract-progress', str(tmp_path / 'progress.tsv'), '--output', str(output),
          '--test-fraction', '0'])

    with output.open(encoding='utf-8') as f:
        splits = {row['page_id']: row['split'] for row in csv.DictReader(f, delimiter='\t')}
    assert splits == {ID.format(0): 'annotated', ID.format(1): 'test', ID.format(2): 'test'}
