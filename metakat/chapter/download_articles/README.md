# Article title page sampler

Collects first pages of journal articles (title, subtitle, authors, abstract, keywords...) from
digital libraries, together with what the library says about each article. The first goal is to
cover every journal a library offers; then each journal is sampled over its whole run, one article
per few years, since journals change their title page layout over time.

## Layout

```
common/          shared code: catalog model, HTTP, OAI-PMH, Kramerius, Open Journal Systems, Crossref,
                 selection, storage, first page extraction, previews, review and the command line
agriculturejournals/  journals of the Czech Academy of Agricultural Sciences (Crossref)
cuni/            journals of Charles University: Karolinum, ojs.cuni.cz, faculty sites (Crossref)
cvut_journals/   journals of the Czech Technical University (OJS)
dml_cz/          Czech Digital Mathematics Library
journal_sites/   single-journal sites found in Crossref by ISSN
knav/            Kramerius of the Library of the Czech Academy of Sciences
landing_pages/   journals found in Crossref by ISSN, PDFs named by their article pages
muni_digilib/    Digital Library of the Faculty of Arts, Masaryk University
muni_journals/   journals of Masaryk University (OJS)
mzk/             Kramerius of the Moravian Library
national_museum/ periodicals of the National Museum in Prague (publikace.nm.cz)
nkp/             Kramerius of the National Library of the Czech Republic
ojs_sites/       single-journal Open Journal Systems sites
upol/            journals of Palacký University Olomouc (Crossref)
```

Every library folder holds its `source.py` (a `common.source.Source`) and its tests; new libraries
are registered in `__main__.py`.

## Install

```bash
pip install -e ".[vis]"     # Python 3.12; add ,dev for the tests
```

`vis` is MetaKat's base install with PyMuPDF, numpy and OpenCV (the review window), and no
torch. Fetching a KNAV volume whose contents page has no ALTO also needs `easyocr`.

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
   are recorded in `unavailable.tsv` and neither requested nor selected again.
4. `preview` renders phone-sized contact sheets into `previews/`: `overview_NN.jpg` with one title
   page per journal and `timelines/` with every stored title page of a journal in year order.

5. `common/review.py` goes through the journals of library folders one by one, to decide which
   journals to trust:

   ```bash
   python -m metakat.chapter.download_articles.common.review /mnt/kolosus/data/smart_digiline/articles/{knav,dml.cz,nkp,journals.muni.cz,ojs.cvut.cz,ojs_sites,cuni,agriculturejournals.cz,upol,journal_sites,landing_pages,national_museum}
   ```

   A window (OpenCV) first shows every stored title page of a journal in year order, labelled with
   year, volume/issue and article title, with each pick's verdict in its corner (green approved, red
   rejected, grey none); clicking a page opens the picks from that one on, in order, to be judged
   like any pick. `y` approves the journal and goes through
   its picks one by one, enlarged (those without a verdict, or all again when every one has one);
   `n` rejects the journal together with all its picks and goes to the next journal; `u` clears the
   journal and its picks. On a pick, `y`/`n` approves or rejects it (after the last one the journal sheet
   is shown again), `u` clears it, `j` leaves the
   rest of the picks for later. Everywhere, ←/→ (or `,`/`.`) go to the previous/next journal or
   pick, space skips, Enter goes on to the next library folder, Esc on a pick returns to its journal
   sheet and `q` (or Esc on a sheet) quits; with everything
   reviewed the window stays open for checking. Verdicts are written after every key: journals into
   `<library>/review.csv`
   (journal id and title, samples, first and last year, `approved`/`rejected`, time) and picks into
   `<library>/review_items.csv` (item id, journal, year, volume, issue, title, image, verdict and
   `by`: `item` for a pick's own verdict, `journal` when it was rejected with its journal). The next
   session continues at the first journal without a verdict, or inside an approved journal at its
   first pick without one; `--all` goes through everything again. Journal sheets are cached in
   `<library>/previews/journals/`.

   Apart from its verdict, a pick can be marked as a review of someone's work (a book, an exhibition,
   tests): `r` on a pick marks or unmarks it (on a guess it first confirms it, so a wrong guess takes `r` twice), shown as a triangle in the top left corner of its tile
   (blue). Marks can be guessed beforehand (`review_by` = `auto`, pale blue, with `review_note` telling
   what the guess rests on); a session also stops at every pick with a guess not yet looked at, and a
   verdict given to the pick confirms its mark as shown (`review_by` = `item`). The columns `review`
   (`yes`/`no`), `review_by` and `review_note` are in `review_items.csv`. The guesses come from
   `common/guess_reviews.py` (`--write` stores them): review sections and types in the metadata, KNAV
   genres from Kramerius, the record title, and the OCR text of the title page (`<library>/txt/`, one
   text line per line) when it exists: an ISBN, a review heading or a citation with a page count near
   the top.

6. The review goes in rounds, until only approved picks are left. A finished round is closed, its
   rejected picks are replaced and the next review shows the approved picks and the new ones:

   ```bash
   python -m metakat.chapter.download_articles.common.review --close-round /mnt/kolosus/data/smart_digiline/articles/{knav,dml.cz,...}
   python -m metakat.chapter.download_articles select --source knav --replace
   python -m metakat.chapter.download_articles fetch  --source knav
   python -m metakat.chapter.download_articles.common.review /mnt/kolosus/data/smart_digiline/articles/{knav,dml.cz,...}
   ```

   `--close-round` writes the round's number into the `round` column of every verdict given so far;
   such verdicts are final, and reviews no longer show the rejected journals and picks of closed
   rounds, nor journals left with no pick (`--closed` shows them again). Picks without a verdict stay
   open. `select --replace` picks a replacement for every pick rejected on its own (not with its
   journal) in a closed round that has no stored replacement yet: an item of the same journal from the
   same year, or else the closest year (the earlier one on a tie), never one stored before, with the
   library's type preference. Items whose title names a part that is rarely an article (title leaf,
   cover, instructions for authors, summary, editorial, preface, obituary; `Source.is_unlikely`) are
   taken only when nothing else is left, and an item with the rejected one's title only when its year
   has no other. The pairs go to `<library>/replacements.tsv`; `fetch` then stores the selection as
   usual. A replacement that the library refuses is replaced by the next `select --replace`. A verdict
   changed in a later round belongs to that round. When a closed round left nothing to replace and
   nothing open, the review shows the final selection: only approved picks, all reviewed.

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
<library>/toc/                  page lists and contents page words (KNAV volumes without article records)
<library>/review.csv            journal verdicts from common/review.py
<library>/review_items.csv      verdicts of the single picks
<library>/replacements.tsv      replacements picked for rejected picks (select --replace)
```

The metadata JSON holds the catalog item (journal, volume, issue, year, title, authors, type,
languages, rights, source URLs and the harvested record) and how the image was obtained.

## Title page resolution

Images are kept at the resolution the library provides. When the title page is one upright scan
covering most of the page, the scan is stored as extracted, without the page margins around it
(`"method": "embedded"`). Any other page is rendered at the highest resolution of its images
(scans cropped to the text block or cut in strips), but at least 300 dpi, for born-digital pages
with or without a coarse figure, and at most 600 dpi, since a small sharp photo or logo on a
born-digital page would ask for 1000–2000 dpi (`"method": "rendered"`). Page images
served by the library are stored as served (`"method": "page image"`). The `dpi` field records the
resolution where it is known.

## Libraries

| `--source` | Library | Notes |
|---|---|---|
| `dml_cz` | [Czech Digital Mathematics Library](https://dml.cz/) | OAI-PMH in the EuDML JATS format (`eudml-article2`): journal, volume, issue, pages, keywords, MSC codes, translated titles and the article PDF. Harvesting without a set misses about half of the journals, so every set is harvested on its own. Proceedings and book collections are not published in the article format and are not harvested; neither is *Rozhledy matematicko-fyzikální*, whose set returns no records. Every PDF starts with a DML-CZ cover sheet, so the title page is page 2. The newest volumes are for registered users only (a moving wall): from the first refused year of a journal its later years are not selected. In old volumes an article may start in the middle of a page, below the end of the previous one. |
| `knav` | [Digital Library of the Czech Academy of Sciences](https://kramerius.lib.cas.cz/) | Every article of every periodical from the Kramerius search index, with its volume and issue numbers, and the articles of volumes without article records found through their contents pages ([below](#knav-volumes-without-article-records)). Born-digital articles are one PDF each; scanned articles have no file of their own, and their title page is the image of the first page they are on. PDFs downloaded earlier into `title.automatic_pick/knav` are reused and preferred, and the logs of those downloads mark articles refused before (403), so neither costs a request. The newest issues of many journals and some whole journals are licensed, so a refusal works as a moving wall: the journal's later years are taken only from earlier downloads. Contents pages, indexes and similar parts catalogued as articles are never picked. |
| `muni_digilib` | [Digital Library of the Faculty of Arts, Masaryk University](https://digilib.phil.muni.cz/) | OAI-PMH (oai_dc) covers 20 of the 54 journals (2026-10); records name the journal and year only, no volume or issue. The Faculty's journals on OJS (journals.phil.muni.cz) keep their files here too; the 14 that OAI-PMH lacks are taken from the `journals.muni.cz` catalog (catalog `muni_journals` first). Files are behind a Cloudflare Turnstile human check, so they are downloaded by hand: `select` writes `selection.html` with one list per journal, and the OJS links of the picked articles are first resolved to the file in the library (one request each to journals.phil.muni.cz), so that the saved files keep the name `fetch --pdf-dir` looks for. The original scan (`-source.pdf`) is preferred where one exists. |

## Articles in Kramerius

Kramerius has an article record only where a library described a periodical at article level:
`model:article` (KNAV, MZK, ČBVK, a few in KFBZ and ZČM) or, in NKP, `model:internalpart` under an
issue. Most periodicals of the Czech Digital Library (ČDK) have issues and pages only, with no article
boundaries; they are not sampled. `root.model` is missing from many ČDK records, so articles are
found by `own_model_path:periodical*`.

Kramerius does not tell newspapers from journals, and the field for genre is filled for few
periodicals, so `common/kramerius.py` counts the volumes and issues of each periodical: one with 40 or
more issues per volume (dailies, weeklies), or catalogued as a newspaper, is left out of the catalog.
Journals have 1–25 issues per volume.

Every library is read from its own Kramerius (`common.kramerius.KrameriusSource`); the ČDK search
was used only to find which libraries hold articles. Libraries share digitised periodicals under the
same pids, so each library leaves the articles already in the catalog of a library sampled before it
to that library, in the order KNAV, MZK, NKP; catalog them in this order. KFBZ, ZČM, NTK and
SVKKL serve no or a couple of articles to the public and have no folder. ČBVK was sampled and dropped
(2026-10): its articles are mostly regional newsletters and bulletins rather than journal articles.
Only public articles are selected.

Items under the out-of-commerce or on-site licences (`dnnto`, `dnntt`, `onsite` in `licenses.facet`)
are never selected, even when the library serves some of their pages anonymously: the licences allow
registered users to read them, not to download them. Title pages stored before this rule are deleted
and recorded in `unavailable.tsv` as "restricted licence ...; not to be used" (19 in KNAV, 2026-10);
such records start no moving wall.

| `--source` | Library | Notes |
|---|---|---|
| `mzk` | [Moravian Library](https://www.mzk.cz/) | Articles not held by KNAV: about 15,000 in 99 journals and magazines (2026-10), every one under the out-of-commerce or on-site licence (`dnnto`, `dnntt`, `onsite`) and refused to anonymous users (403), so nothing is selected. The public MZK articles are the KNAV copies and Lidové noviny. |
| `nkp` | [National Library](https://www.nkp.cz/) | Internal parts of issues; nearly all are in newspapers (left out) or licensed military journals. The public ones are in 9 short-lived periodicals, mostly Pilsen magazines of 1884–1910. |

### KNAV volumes without article records

The KNAV list of periodicals (sheet "Časopisy bez metadat článků") names periodicals whose articles
were not catalogued in some or all volumes: scanned as issues and pages only (Vesmír before 1998,
Organon F before 2009, Umění, Byzantinoslavica, ...). `knav/unsegmented_periodicals.tsv` lists them.
`catalog` lists their volumes and adds every volume of a year without article records as one item
of type `volume`; `select` picks volumes by period like articles. `fetch` finds the volume's
article through its contents pages (`common/toc.py`) and stores that page:

1. The volume's pages and issues are listed, in reading order.
2. The words of every page typed `TableOfContents` (where no page is, of the first and last 8 pages,
   keeping those with 5 or more entries matching pages of the volume) are read with their boxes from the page's ALTO,
   or, where KNAV has none, by OCR of the page image ([EasyOCR](https://github.com/JaidedAI/EasyOCR),
   an optional dependency: `pip install easyocr`).
3. Words are joined into lines and a line is split at wide gaps (columns of an index, a title and
   its right-aligned number); a segment ending in a number is an entry starting on that page.
4. An entry's number is looked up among the printed page numbers (`[12]` and `(12)`, unprinted
   numbers, count as `12`) of the contents page's own issue first, then of the whole volume; an
   entry matching no page or several (pagination restarting every issue) is dropped, and so are
   matches on contents, blank or advertisement pages.

5. Of the start pages found, the one with the most pages before the next start is the volume's
   article: a main article rather than a review or a short note. Entries numbered like sections of
   a longer text ("1.4 The impact ...") are left out.

The stored item keeps the volume's metadata, with the issue of the page found; it has no title, and
`record` holds the page (`first_page`, `page_number`), the entries' text as read (`toc_entries`),
the contents pages and how many start pages were found. A volume without a matching entry is
recorded as unavailable; since KNAV did not refuse it, it starts no moving wall. Annual indexes (Vesmír) list every short note too, so their entries cover most
pages. Page lists and contents words are kept in `knav/toc/`, so KNAV is asked and OCR run once.

## Journals outside Kramerius

Czech journals published by universities and societies keep their articles on their own sites. Two
kinds of interface serve most of them with the same code:

- **Open Journal Systems** (`common/ojs.py`): OAI-PMH gives every article with its journal and section
  (the record's set `journal:section`), the citation ("Religio; Vol 12 No 1 (2004); 5-26", in older
  installations "AntropoWebzin 1/2013") and its galleys, downloaded from
  `.../article/download/<article>/<galley>`. The year is taken from the citation only, since the
  record's date is often the date of upload; articles whose citation has no year are not picked.
  Items of review, news, editorial and similar sections are picked only from years without a research
  article. A galley that redirects to another site (a digital library behind a human check, a paid
  database) is not followed and the article is recorded as refused with the address it pointed to; a
  journal refused that way twice, with nothing stored, is not selected any more.
- **Crossref** (`common/crossref.py`): publishers without OAI-PMH register their DOIs with the URL of
  the article PDF (for similarity checking). A link of unspecified type counts when its address ends
  in `.pdf` or it is meant for similarity checking (`.../pdf`, `dl/123`); the download keeps only
  answers that are PDFs. All journals of a publisher are harvested by its DOI
  prefix, single journals by their ISSNs. A journal is identified by its ISSNs; its title is the most
  frequent spelling of `container-title`, so a renamed journal is still two journals. Crossref only
  knows the years since the journal registers DOIs, mostly from 2010 on. Only one process should query
  Crossref at a time: it answers parallel anonymous requests with 429. Some deposited links are dead
  (old De Gruyter addresses, hosts that no longer exist); those articles are recorded as refused.

The journals were found in the Directory of Open Access Journals (Czech journals, 2026-10): of 169, 34
are in the libraries above and about 115 on the platforms and sites below. About 20 small journals on
WordPress, Drupal and similar sites have neither OAI-PMH nor PDF links in Crossref and are not
harvested. The registers of Czech journals (the government lists of peer-reviewed periodicals RVVI 2008 and
2015, ERIH PLUS 2023, OpenAlex) add 86 journals with PDF links in Crossref (`journal_sites`) and,
of 105 without, 6 on OJS (`ojs_sites`), 34 whose article pages name or are the PDF
(`landing_pages`) and 10 of the National Museum (`national_museum`). The medical journals on the
prolekare.cz platform (Care Comm, the Czech Medical Association) are left out: their articles are
shown only after declaring oneself a healthcare professional. About 25 others are on sites of their
own without a link to the PDF, and about 15 are gone or refuse requests.

| `--source` | Publisher | Notes |
|---|---|---|
| `muni_journals` | [Masaryk University](https://journals.muni.cz/) (OJS) | 26,540 articles in 49 journals (2026-10). The 22 Faculty of Arts journals (journals.phil.muni.cz, e.g. Religio, Theatralia, Opera Slavica) redirect their files to the Faculty's digital library, which serves people only; they are refused, and their PDF addresses are in `unavailable.tsv` for downloading by hand (see `muni_digilib`). Czech Journal of Political Science keeps older volumes in CEEOL. Citation lists, proceedings and book series are not harvested. |
| `cvut_journals` | [Czech Technical University](https://ojs.cvut.cz/) (OJS) | 4,925 articles in 10 journals. Applications of Structural Fire Engineering (conference) cites no year and is not picked. |
| `ojs_sites` | single-journal OJS sites | 3,670 articles in 13 journals whose Crossref records lack PDF links (JERES, Applied and Computational Mechanics, Theology and Philosophy of Education, IJATES, AntropoWebzin, Advances in Military Technology, AUC Studia Territorialia; from the registers: Česká a slovenská psychiatrie, Rozhledy v chirurgii, Akustika, Law, Business and Sustainability Herald, Global Prosperity, Middle European Scientific Bulletin). Česká a slovenská psychiatrie and Rozhledy v chirurgii serve files only after a login. |
| `cuni` | [Charles University](https://karolinum.cz/) (Crossref, prefix 10.14712) | 14,004 articles in 46 journals: Karolinum Press (the AUC series, Orbis Scholae, ...), ojs.cuni.cz and the faculties' journal sites; 12,268 with a PDF link. |
| `agriculturejournals` | [Czech Academy of Agricultural Sciences](https://www.agriculturejournals.cz/) (Crossref, prefix 10.17221) | 14,005 articles in 11 journals, 1999–2026. The sites answer requests 2 s apart with 429 now and then, so requests are 5 s apart. |
| `upol` | [Palacký University Olomouc](https://www.upol.cz/) (Crossref, prefix 10.5507) | 7,120 articles in 24 journals, 2000–2026. |
| `journal_sites` | single-journal sites (Crossref, by ISSN) | The other Czech journals with PDF links in Crossref: 37 from DOAJ and 86 from the government list of peer-reviewed periodicals (RVVI 2008 and 2015), ERIH PLUS and OpenAlex, of universities, institutes, museums, societies and medical publishers (Solen, Galen, Care Comm); 122 journals (129 titles, counting renames), 49,219 articles, 47,042 with a PDF link. Articles whose DOI one of the libraries above holds are left to it. The register comparison is in `registers/czech_journals.tsv` under `--root`. |
| `landing_pages` | single-journal sites (Crossref, by ISSN, PDF from the article page) | Register journals without PDF links in Crossref whose article pages name the PDF in `citation_pdf_url`, are the PDF, or link the site's own file (`landing_pdf_link`): 34 journals such as Geografie (from 1960), Acta Veterinaria Brno (from 1978), Agris on-line, Lifelong Learning, Klinická onkologie, the DSpace of the University of West Bohemia (Castellologica bohemica, Kuděj, MEMO), the University of Ostrava (Studia Slavica, Studia Romanistica, ...), sociosphera.com, Aussiger Beiträge (UJEP repository) and the three journals of Enigma Corporation; 8,321 articles. A pick costs two requests, the page and the PDF. |
| `national_museum` | [National Museum](https://publikace.nm.cz/periodicke-publikace) | 7,159 articles in 14 periodicals (Acta Musei Nationalis Pragae – Historia and Historia litterarum, Annals of the Náprstek Museum, Lynx, Fossil Imprint, Journal of the National Museum, Numismatické listy, Muzeum, ...), from the issue pages; the article page links the PDF. Older volumes are listed with abstracts only; such articles are refused but start no moving wall. European Journal of Taxonomy is published elsewhere and left out. |
