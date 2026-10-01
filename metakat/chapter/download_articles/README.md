# Article title page sampler

Collects first pages of journal articles (title, subtitle, authors, abstract, keywords...) from
digital libraries, together with what the library says about each article. The first goal is to
cover every journal a library offers; then each journal is sampled over its whole run, one article
per few years, since journals change their title page layout over time.

## Layout

```
common/          shared code: catalog model, HTTP, OAI-PMH, selection, storage, first page
                 extraction, previews and the command line
dml_cz/          Czech Digital Mathematics Library
knav/            Kramerius of the Library of the Czech Academy of Sciences
muni_digilib/    Digital Library of the Faculty of Arts, Masaryk University
```

Every library folder holds its `source.py` (a `common.source.Source`) and its tests; new libraries
are registered in `__main__.py`.

## Usage

```bash
python -m metakat.chapter.download_articles catalog --source dml_cz
python -m metakat.chapter.download_articles select  --source dml_cz --period 5
python -m metakat.chapter.download_articles fetch   --source dml_cz
python -m metakat.chapter.download_articles preview --source dml_cz
```

1. `catalog` harvests the library's catalog into `catalog.jsonl`.
2. `select` picks items not stored yet and writes them to `selection.tsv` and `selection.html`. A
   journal is its id together with its title, so every title of a renamed journal is covered.
   - `--period N` gives every journal its first and last year and one item per N years counted
     from its first year. Periods that already have a stored item are left alone; an empty one
     gets an item from the year closest to its middle. Items the library can be served without a
     request (e.g. downloaded before) are preferred: a period takes such a year over a slightly
     more central one.
   - `--per-journal N` instead picks N items per journal spread over its years (one pick is its
     median year); `--min-year-gap` keeps them that many years apart from every stored item.
3. `fetch` stores every selected item: the PDF, its title page and a metadata JSON. PDFs found in
   `--pdf-dir` (under the name the library serves them with), or kept from earlier downloads, are
   used first; libraries that allow it are downloaded from otherwise. Items the library refuses
   are recorded in `unavailable.tsv` and not selected again.
4. `preview` renders phone-sized contact sheets into `previews/`: `overview_NN.jpg` with one title
   page per journal and `timelines/` with every stored title page of a journal in year order.

Requests are never parallel and at least `--delay` seconds apart (default 1 s for `catalog`, 2 s
for `fetch`); a server asking to slow down (429/503) is waited for as long as it asks.

Each library gets its own folder under `--root` (default
`/mnt/kolosus/data/smart_digiline/articles`):

```
<library>/catalog.jsonl         every catalog item
<library>/selection.tsv|.html   the last selection
<library>/unavailable.tsv       items the library refused
<library>/pdf/<item_id>.pdf     the article PDF as served (none for libraries serving page images)
<library>/images/<item_id>.*    its title page
<library>/metadata/<item_id>.json
<library>/previews/
```

The metadata JSON holds the catalog item (journal, volume, issue, year, title, authors, type,
languages, rights, source URLs and the harvested record) and how the image was obtained.

## Title page resolution

Images are kept at the resolution the library provides. When the title page is one upright scan
covering most of the page, the scan is stored as extracted, without the page margins around it
(`"method": "embedded"`). Any other scanned page is rendered at the highest resolution of its
images, and a born-digital page without images at 300 dpi (`"method": "rendered"`). Page images
served by the library are stored as served (`"method": "page image"`). The `dpi` field records the
resolution where it is known.

## Libraries

| `--source` | Library | Notes |
|---|---|---|
| `dml_cz` | [Czech Digital Mathematics Library](https://dml.cz/) | OAI-PMH in the EuDML JATS format (`eudml-article2`): journal, volume, issue, pages, keywords, MSC codes, translated titles and the article PDF. Harvesting without a set misses about half of the journals, so every set is harvested on its own. Proceedings and book collections are not published in the article format and are not harvested; neither is *Rozhledy matematicko-fyzikální*, whose set returns no records. Every PDF starts with a DML-CZ cover sheet, so the title page is page 2. The newest volumes are for registered users only (a moving wall): from the first refused year of a journal its later years are not selected. In old volumes an article may start in the middle of a page, below the end of the previous one. |
| `knav` | [Digital Library of the Czech Academy of Sciences](https://kramerius.lib.cas.cz/) | Every article of every periodical from the Kramerius search index, with its volume and issue numbers. Born-digital articles are one PDF each; scanned articles have no file of their own, and their title page is the image of the first page they are on. PDFs downloaded earlier into `title.automatic_pick/knav` are reused and preferred, and the logs of those downloads mark articles refused before (403), so neither costs a request. Contents pages, indexes and similar parts catalogued as articles are never picked. |
| `muni_digilib` | [Digital Library of the Faculty of Arts, Masaryk University](https://digilib.phil.muni.cz/) | OAI-PMH (oai_dc) covers 11 of the 54 journals; records name the journal and year only, no volume or issue. Files are behind a Cloudflare Turnstile human check, so they are downloaded by hand from `selection.html` and passed to `fetch --pdf-dir`. The original scan (`-source.pdf`) is preferred where one exists. |
