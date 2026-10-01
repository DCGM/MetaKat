# Article title page sampler

Collects first pages of journal articles (title, subtitle, authors, abstract, keywords...) from
digital libraries, together with what the library says about each article. The first goal is to
cover every journal a library offers; more samples of a journal can be added later from issues
far enough apart in time.

## Usage

```bash
python -m metakat.chapter.download_articles catalog --source muni_digilib
python -m metakat.chapter.download_articles select  --source muni_digilib --per-journal 1
python -m metakat.chapter.download_articles select  --source dml_cz --period 5
python -m metakat.chapter.download_articles fetch   --source muni_digilib --pdf-dir ~/Downloads/muni
```

1. `catalog` harvests the library's catalog into `catalog.jsonl`.
2. `select` picks `--per-journal` items from every journal that are not stored yet and writes
   them to `selection.tsv` and `selection.html`. A journal is its id together with its title, so
   every title of a renamed journal is covered. Items are spread over the journal's years (one
   pick is its median year); `--min-year-gap` keeps new picks that many years away from every
   item of the journal already stored, so repeated runs add samples from other periods.
   With `--period N` the selection instead gives every journal its first and last year and one
   item per N years counted from its first year; periods that already have a stored item are
   left alone, empty ones get an item from the year closest to the period's middle.
3. `fetch` stores every selected item: the PDF, its first page and a metadata JSON. PDFs found in
   `--pdf-dir` (under the name the library serves them with) are used first; libraries that allow
   it are downloaded from otherwise.

Each library gets its own folder under `--root` (default
`/mnt/kolosus/data/smart_digiline/articles`):

```
<library>/catalog.jsonl         every catalog item
<library>/selection.tsv|.html   the last selection
<library>/pdf/<item_id>.pdf     the article PDF as served
<library>/images/<item_id>.*    its first page
<library>/metadata/<item_id>.json
```

The metadata JSON holds the catalog item (journal, volume, issue, year, title, authors, type,
languages, rights, source URLs and the harvested record) and how the image was obtained.

## First page resolution

Images are kept at the resolution the library provides. When the first page is one upright scan
covering the page, the scan is stored byte for byte (`"method": "embedded"`). Any other scanned
page is rendered at the highest resolution of its images, and a born-digital page without images
at 300 dpi (`"method": "rendered"`). The `dpi` field records the resolution either way.

## Sources

| `--source` | Library | Notes |
|---|---|---|
| `dml_cz` | [Czech Digital Mathematics Library](https://dml.cz/) | OAI-PMH in the EuDML JATS format (`eudml-article2`): journal, volume, issue, pages, keywords, MSC codes, translated titles and the article PDF, downloaded directly. Harvesting without a set misses about half of the journals, so every set is harvested on its own. Proceedings and book collections are not published in the article format and are not harvested; neither is *Rozhledy matematicko-fyzikální*, whose set returns no records. In old volumes an article may start in the middle of a page, below the end of the previous one. |
| `muni_digilib` | [Digital Library of the Faculty of Arts, Masaryk University](https://digilib.phil.muni.cz/) | OAI-PMH (oai_dc) covers 11 of the 54 journals; records name the journal and year only, no volume or issue. Files are behind a Cloudflare Turnstile human check, so they are downloaded by hand from `selection.html` and passed to `fetch --pdf-dir`. The original scan (`-source.pdf`) is preferred where one exists. |
