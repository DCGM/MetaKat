# Proposed revision of the MetaKat metadata schema

## What this is

Background for a decision, not an implementation plan. It describes how the
metadata model MetaKat produces would change from what is on `main` today, and
why each change is proposed.

**Nothing here is settled.** The point of writing it down is to get the model
confirmed — or corrected — before any of the surrounding code is reimplemented
against it. Section 7 lists the questions that need an answer from the archive
side; everything else is context for those questions.

Throughout, "DMF" means *Definice metadatových formátů* — DMF pro digitalizaci
monografických dokumentů v. 2.3 and DMF pro digitalizaci periodik v. 2.2 — and
"the mapping" means the `metada_mapping.xlsx` spreadsheet.

---

## 1. Where we start

On `main`, MetaKat describes a document with seven independent classes: title,
volume, issue, page, supplement, chapter, article. Each carries its own flat
list of fields, and three properties of that arrangement are what this proposal
addresses.

**A field holds one value.** A title page listing two publishers, or a book
printed in Praha *and* Brno, has to lose one of them.

**Values that belong together are not connected.** "Praha : Odeon, 1902" and
"Brno : Barvič, 1908" become four unrelated entries — three places, two
publishers, two dates in separate lists — with nothing recording which place
goes with which publisher.

**The levels drifted apart.** A volume could have an illustrator and a
photographer but no *redaktor*; an issue had a *redaktor* and nothing else; a
supplement had only an author. None of that comes from the standard — the DMF
places no role restriction on any level.

---

## 2. Three principles behind the change

**MODS has one element for every level.** A title, a volume, an issue and a
supplement are not different record types: they are the same `<mods>` element,
distinguished by its `ID` and its `<genre>`. The DMF's per-level tables differ
almost entirely in which children are *mandatory*, not in which children exist.
Seven divergent classes were a MetaKat invention.

**The container is what carries the binding.** This is the DMF's own rule, in
its description of `<originInfo>`:

> …v případě, že je v jednom poli 260/264 uvedeno opakované podpole $a nebo $b,
> je možné příslušné subelementy opakovat v rámci jednoho `<originInfo>` nebo se
> **zopakuje celý `<originInfo>` tak, aby se neztratily vzájemné vazby mezi
> subelementy** (např. mezi konkrétním místem vydání a vydavatelem).

Both serialisations are permitted, which matters: a record that cannot work out
the bindings repeats the subelements inside one container and is still valid;
one that can repeats the whole container and is more precise.

**Only what can be read from a scan.** MetaKat extracts from images. Signatures,
URN:NBN, Konspekt and czenas headings, UDC, authority-file numbers and the whole
of `<recordInfo>` dominate the catalogue records but originate in the catalogue
or the digitisation workflow, so none of them is a candidate. Section 8 lists
what was excluded on this ground.

---

## 3. Every value may repeat

Each field became a list. A record with one publisher holds a one-item list; a
title page with two holds both.

This also removed an inconsistency: `publisher` was already a list while
`placeTerm`, `dateIssued` and `edition` were single values, so the schema could
hold three publishers but only one place — the very relationship the DMF asks to
be preserved was impossible to record.

---

## 4. One shared field set per hierarchy

The four bibliographic levels — title, volume, issue, supplement — now share one
field set, and chapter and article share another.

The two sets stay apart for a substantive reason: **an internal part has no
`<originInfo>` at all.** A chapter or an article inherits its imprint from the
volume or issue carrying it, so publisher, place, all the dates, edition,
frequency and the manufacture trio are inapplicable there — thirteen fields.
Merging the two would leave nearly half of them permanently empty.

Within each set, a field being available at a level does not mean it is
meaningful there. A periodical volume legitimately fills little beyond
`partNumber` and `dateIssued`. Which fields apply where is recorded in the
tables in the appendix, and is deliberately *not* enforced in code yet — those
tables are the specification such rules would be written from.

One consequence worth flagging: sharing the field set makes every role available
at every level, which is what the DMF describes, and resolves the drift
described in section 1.

---

## 5. What each value now carries, and where it was read

**Language.** A value may carry the language it is written in. This is what
parallel title pages and bilingual abstracts need: a Czech article with an
English abstract and an English parallel title currently gives no way to tell
which string is which. In the 116-record article dataset, 36 records carry two
or three languages across their titles, abstracts and keywords.

Note this is *per value*. MODS itself tags the container — `<titleInfo lang="eng"
type="translated">` covers title, subTitle, partNumber and partName together —
so recording it per value is the evidence from which those blocks are
reconstructed, not a claim that the value owns the language.

Separately, both hierarchies gained a `language` **field**, which is a different
thing: `<language><languageTerm>`, the language the document or part is *written*
in. It cannot be derived from the per-value languages — in 29 of 63 article
records the first title's language differs from the abstract's.

**Which page a value was read on.** A chapter or article is described by two
pages, and they are read separately: the table-of-contents entry that points at
it, and the part's own opening page. The field names now say which:

- **no suffix** — read on the part's own destination page;
- **`TocPage`** — read in the table-of-contents entry, including the page number
  printed on the right of that entry.

The old names had this backwards: `title`, `subTitle`, `partNumber` and
`pageNumber` were all filled from the TOC while the destination reading carried
the explicit suffix.

The two sides may disagree, and the schema does not assert that they agree. A
chapter listed in the TOC as „Počátky písma“ and headed „I. Počátky písma“ on
its own page keeps both readings.

---

## 6. Recording which values belong together

Each record carries a list of groups. A group has a type and a list of
identifiers, and nothing else:

```
type: titleInfo | originInfoPublication | originInfoManufacture
    | agent | series | reviewedWork | pageRange
members: [ids of the values that belong together]
```

The type names the MODS container the members would sit inside, so a reader
knows what a group is about without inspecting it. Two publication events on one
title page become two `originInfoPublication` groups; an author with their
affiliation and e-mail becomes one `agent` group.

Three properties are deliberate:

**A group contains, it does not interpret.** The type says which container the
members belong in and no more. A `titleInfo` group holding a title read from the
TOC and the same title read from the opening page simply holds both; deciding
they are one title is the reader's job.

**Grouping is never required.** Because the DMF permits both serialisations, a
record with no groups is still valid — the subelements are repeated inside one
container. Partial grouping, two of three publishers grouped, is a normal state.

**No group for `<subject>`.** It would bind the topics of one keyword block, but
the per-value language already separates them: of the 116 article records, all
45 that carry several title, abstract or keyword blocks give every block a
distinct language, and none repeats one.

---

## 7. Questions for the meeting

These are the points where the proposal rests on an assumption that the archive
side should confirm or correct.

**1. MODS 3.6 or 3.8.** The packages MetaKat sees today are MODS 3.6, but DMF
monografie 2.2 / periodika 2.1 (December 2024) moved the standard to 3.8, and
one change lands directly on the imprint: `<originInfo><publisher>` is replaced
by `<originInfo><agent><namePart>`. ProArc still emits 3.6. Which should MetaKat
target, and is there a date by which output must be 3.8?

**2. Printer at issue level, and the name of the field.** The mapping's row 17
assigns *tiskař* to `MODS_ISSUE`, but the DMF's issue table lists only
publication-era children of `originInfo`. Is a manufacture block expected on an
issue? Separately, the field is currently called `manufactureDateIssued` while
it serialises to `<dateOther type="manufacture">` — `manufactureDate` would match
the standard, and renaming now is cheap.

**3. Two controlled vocabularies we could not close.**
   - `form` (`physicalDescription/form`) currently admits only *print* and
     *manuscript*. The DMF does not enumerate the values, deferring to MARC
     008/23. Which of those values should MetaKat be able to produce?
   - `articleGenre` (`genre @type`) currently admits *review*, *interview*,
     *cover*, *tableOfContents* — the four the mapping names. The DMF defers the
     full list to Pravidla pro popis periodik v. 8.7. Which values matter?

**4. The date on an article.** An internal part has no `<originInfo>`, so an
article's date of issue belongs to the issue carrying it. It is nevertheless
printed on the article's own page and is present in 72 of 116 ground-truth
records. The proposal keeps it on the article as extraction evidence and
serialises it to the parent. Is that acceptable, or should it be written only to
the parent?

**5. E-mail addresses.** A corresponding author's address is printed and
extractable, but MODS models no contact data for names — `<name>` admits
`namePart`, `displayForm`, `affiliation`, `role`, `description`,
`nameIdentifier`, `alternativeName`, `etal` and nothing else. Should MetaKat
record it as a field that never exports, or drop it?

**6. The imprint of a reviewed work.** The mapping's row 30 keeps place,
publisher and year of the reviewed book in a single `<publisher>` element, which
is how a review header prints it. MetaKat follows that with one field rather
than three. Confirm that is what is wanted.

**7. Page ranges printed in the TOC.** The page number in a TOC entry is
recorded on the chapter or article and serialises to that part's own
`<part type="pageNumber">` — MODS records what the number is, not where it was
read. Confirm that is the intended treatment.

**8. Anything missing.** The fields below are what we judged both useful and
readable from a scan. If something an archive needs is absent, this is the
moment to say so.

---

## 8. Deliberately excluded

Everything in the DMF that cannot come off a scan, listed so the omission reads
as a decision rather than an oversight.

| Excluded | Because |
|---|---|
| `identifier` — uuid, urnnbn, ccnb, oclc, sysno, barcode | Minted by the workflow or assigned by the catalogue. Only ISBN and ISSN are printed on the object, and both are proposed additions. |
| `location` — physicalLocation, shelfLocator, url | Sigla and call number: a property of the copy, not of the edition. |
| `recordInfo` — all children | Metadata about the metadata record. |
| `classification` (Konspekt, UDC), `subject @authority="czenas"` | Controlled vocabularies applied by a cataloguer. Printed keywords are a different thing and are in scope. |
| `name/nameIdentifier` | National authority number. The printed name is in scope; its authority id is not. |
| `placeTerm @type="code"` (marccountry) | The "xr" form. In the sample corpus every repeated place is a code-plus-text pair for one place, never two places. |
| `issuance`, `typeOfResource` | Determined by the document hierarchy, not by reading the page. |
| `nonSort` | Zero occurrences in 143 sample records; Czech and Polish have no articles. |
| A general `note` | Of 76 notes in the sample, 26 are the statement of responsibility (now its own field), 13 are language, 4 restate page types, and the remaining 25 are cataloguer commentary spanning a whole run that no single page contains. |

---

## 9. Still proposed, not yet added

| Field | MODS | Why |
|---|---|---|
| `isbn`, `issn` | `identifier @type` | Printed in the impressum or as a barcode, and format-validatable. |
| `extent` | `physicalDescription/extent` | „176 stran.“ Read from the impressum or derived from the page count. |
| `level` | — | Nesting depth of a chapter. Needed to represent a TOC division such as „Část I: Starověk“, which the logical structure map can only express as a parent chapter. |

---

## Appendix A — bibliographic levels

Fields shared by **title (T)**, **volume (V)**, **issue (I)** and
**supplement (S)**, in declaration order. "Levels" says where the field is
meaningful, not where it is permitted — nothing is enforced.

| Field | Levels | Stored in MODS as | Note |
|---|---|---|---|
| `id` | all | — | MetaKat identity. |
| `parent_id` | V I S | — | Title is the root. A supplement hangs off a volume or an issue. |
| `page_id` | all | — | Which page the record was anchored to. MetaKat only. |
| `hierarchy` | T V | — | multipart / monograph / periodical. Not MODS. |
| `partNumber` | all | `titleInfo/partNumber` | Volume number, issue number, part of a multipart set. |
| `partName` | all | `titleInfo/partName` | Yearbooks, special and thematic issues. |
| `title` | all | `titleInfo/title` | A periodical *volume* has none — its `titleInfo` admits only `partNumber`. |
| `subTitle` | all | `titleInfo/subTitle` |  |
| `edition` | T V | `originInfo/edition` | A supplement’s `originInfo` omits it; at issue level a mutation goes through a repeated `titleInfo` instead. |
| `frequency` | T S | `originInfo/frequency` | A periodical title, or a supplement issued as its own series. |
| `statementOfResponsibility` | all | `note @type="statement of responsibility"` | **New.** The verbatim printed line, e.g. „sepsal Vincenc Blahouš“. |
| `publisher` | all | `originInfo/publisher` (3.6) / `originInfo/agent/namePart` (3.8) |  |
| `placeTerm` | all | `originInfo/place/placeTerm @type="text"` |  |
| `dateIssued` | all | `originInfo/dateIssued` |  |
| `copyrightDate` | T V S | `originInfo/copyrightDate` | **New.** „© 1967“ on the verso — often the only date a book prints. |
| `manufacturePublisher` | all | `originInfo @eventType="manufacture"/publisher` |  |
| `manufacturePlaceTerm` | all | `originInfo @eventType="manufacture"/place/placeTerm` |  |
| `manufactureDateIssued` | all | `originInfo @eventType="manufacture"/dateOther` | **New.** See open question 2 on the name. |
| `seriesName` | V | `relatedItem @type="series"/titleInfo/title` |  |
| `seriesPartNumber` | V | `relatedItem @type="series"/titleInfo/partNumber` |  |
| `seriesPartName` | V | `relatedItem @type="series"/titleInfo/partName` | **New.** From 830 $p, naming a subseries. |
| `language` | all | `language/languageTerm @type="code"` | **New.** iso639-2b. The language the document is written in. |
| `form` | all | `physicalDescription/form @authority="marcform"` | **New.** print / manuscript. See open question 3. |
| `author` | all | `name` + `role/roleTerm` "aut" |  |
| `illustrator` | all | `name` + `role/roleTerm` "ill" |  |
| `photographer` | all | `name` + `role/roleTerm` "pht" |  |
| `translator` | all | `name` + `role/roleTerm` "trl" |  |
| `editor` | all | `name` + `role/roleTerm` "edt" |  |
| `redaktor` | all | `name` + `role/roleTerm` |  |
| `affiliation` | all | `name/affiliation` | **New.** The institution a named person belongs to. |
| `email` | all | **none** | **New.** No MODS element exists. See open question 5. |
| `groups` | all | — | Detection groupings — see section 6. |

## Appendix B — chapters and articles

Fields shared by **chapter (C)** and **article (A)**, in declaration order.
Destination-page fields first, then table-of-contents fields.

| Field | Kinds | Stored in MODS as | Note |
|---|---|---|---|
| `id` | both | — | MetaKat identity. |
| `parent_id` | both | — | A volume, an issue, or another chapter. |
| `pageIndexStart` | both | `part @type="pageIndex"/extent/start` | Which scans the part occupies. A list — a part continued on later pages has several runs. |
| `pageIndexEnd` | both | `part @type="pageIndex"/extent/end` |  |
| `title` | both | `titleInfo/title` | Read on the part’s **own opening page**. |
| `subTitle` | both | `titleInfo/subTitle` | The DMF admits a perex here. |
| `abstract` | both | `abstract` | A list — parallel Czech and English abstracts are normal. |
| `keywords` | both | `subject/topic` | A list — one entry per printed term. |
| `dateIssued` | A | *parent’s* `originInfo/dateIssued` | **New.** See open question 4. |
| `reviewedWorkTitle` | A | `relatedItem/titleInfo/title` | **New.** The book under review. |
| `reviewedWorkAuthor` | A | `relatedItem/name/namePart` | **New.** |
| `reviewedWorkImprint` | A | `relatedItem/originInfo/publisher` | **New.** Place, publisher and year as one printed line. See open question 6. |
| `language` | both | `language/languageTerm @type="code"` | **New.** The language the part is written in. |
| `articleGenre` | A | `genre @type=...` | **New.** review, interview, cover, tableOfContents. See open question 3. |
| `pageIndexTocPage` | both | — | Which scan carries the table-of-contents entry. |
| `titleTocPage` | both | `titleInfo/title` | The same title as read **in the TOC entry**. |
| `subTitleTocPage` | both | `titleInfo/subTitle` |  |
| `partNumberTocPage` | both | `titleInfo/partNumber` | The chapter number (I., XI., 5.), kept out of the title. |
| `pageNumberStartTocPage` | both | `part @type="pageNumber"/extent/start` | The page number printed in the entry, usually on the right. |
| `pageNumberEndTocPage` | both | `part @type="pageNumber"/extent/end` | When the entry gives a range. |
| `author` | both | `name` + `role/roleTerm` "aut" |  |
| `illustrator` | both | `name` + `role/roleTerm` "ill" |  |
| `photographer` | both | `name` + `role/roleTerm` "pht" |  |
| `translator` | both | `name` + `role/roleTerm` "trl" |  |
| `editor` | both | `name` + `role/roleTerm` "edt" |  |
| `redaktor` | both | `name` + `role/roleTerm` |  |
| `affiliation` | both | `name/affiliation` | **New.** |
| `email` | both | **none** | **New.** No MODS element exists. |
| `groups` | both | — | Detection groupings — see section 6. |
