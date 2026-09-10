# Návrh úpravy metadatového schématu MetaKat

## K čemu tento dokument slouží

Je to podklad pro rozhodnutí, ne plán implementace. Popisuje, jak by se
metadatový model, který MetaKat produkuje, změnil oproti stavu na větvi `main`,
a proč je každá změna navržena.

**Nic z toho není uzavřené.** Smyslem sepsání je nechat si model potvrdit — nebo
opravit — dřív, než se proti němu přepíše okolní kód. Kapitola 4 shrnuje otázky,
na které potřebujeme odpověď z archivní strany; zbytek je kontext k nim.

Zkratkou „DMF“ se dále rozumí *Definice metadatových formátů* — DMF pro
digitalizaci monografických dokumentů v. 2.3 a DMF pro digitalizaci periodik
v. 2.2 — a „mapováním“ tabulka `metada_mapping.xlsx`. „Testovací sadou“ se
rozumí 116 ručně opravených záznamů článků, což je náhodný vzorek z KNAV; čísla,
která se na ni dále odvolávají, popisují tento vzorek.

Názvy polí, elementů MODS a hodnot řízených slovníků zůstávají v celém dokumentu
v původní podobě, protože odkazují na konkrétní pole v MetaKat, respektive na
konkrétní elementy MODS.

---

## 1. Výchozí stav a jeho nedostatky

Na větvi `main` popisuje MetaKat dokument sedmi samostatnými třídami: titul,
svazek, číslo, strana, příloha, kapitola, článek. Každá z nich má vlastní plochý
seznam polí a tři vlastnosti tohoto uspořádání jsou to, co návrh řeší.

**Pole nese jednu hodnotu.** Titulní list se dvěma nakladateli, nebo kniha
vytištěná v Praze *i* v Brně, o jeden údaj nutně přijde.

**Hodnoty, které patří k sobě, nejsou propojené.** Z „Praha : Odeon, 1902“ a
„Brno : Barvič, 1908“ se stanou čtyři nesouvisející položky — dvě místa, dva
nakladatelé, dvě data v oddělených seznamech — a nikde není zaznamenáno, které
místo patří ke kterému nakladateli.

**Úrovně se od sebe vzdálily.** Svazek mohl mít ilustrátora a fotografa, ale ne
redaktora; číslo mělo redaktora a nic dalšího; příloha jen autora. Nic z toho
nevychází ze standardu — DMF žádné omezení rolí na jednotlivých úrovních
nestanoví.

---

## 2. Nové uspořádání: dvě základní třídy

Čtyři bibliografické úrovně — titul, svazek, číslo, příloha — mají nyní společnou
sadu polí, a kapitola s článkem druhou.

Vychází to z toho, že **MODS má pro každou úroveň jeden a tentýž element**.
Titul, svazek, číslo a příloha nejsou různé typy záznamu: je to týž element
`<mods>`, rozlišený svým `ID` a hodnotou `<genre>`. Tabulky DMF se pro jednotlivé
úrovně liší téměř výhradně v tom, které elementy jsou *povinné*, nikoli v tom,
které vůbec existují. Sedm rozbíhajících se tříd byl vynález MetaKat.

Obě sady přitom zůstávají oddělené z věcného důvodu: **vnitřní část nemá
`<originInfo>` vůbec.** Kapitola ani článek se nevydávají samostatně,
nakladatelské údaje přebírají od svazku či čísla, které je nese. Nakladatel,
místo, všechna data, vydání, periodicita i trojice údajů o tisku jsou tam tedy
nepoužitelné — třináct polí. Sloučením by téměř polovina zůstala trvale prázdná.

Uvnitř každé sady platí, že dostupnost pole na dané úrovni neznamená, že tam
dává smysl. Ročník periodika legitimně vyplní sotva víc než `partNumber` a
`dateIssued`; sloupec „Úrovně“ v tabulkách níže říká, kde pole dává smysl.

### Dvě drobnější změny

**Z každého pole se stal seznam.** Záznam s jedním nakladatelem nese
jednoprvkový seznam, titulní list se dvěma nese oba. Tím se zároveň odstranila
nedůslednost, kdy `publisher` už seznamem byl, zatímco `placeTerm`, `dateIssued`
a `edition` byly jednotlivé hodnoty — schéma uneslo tři nakladatele, ale jen
jedno místo.

**Hodnota se zapisuje jako slovník.** Dosud to byla trojice, ve které si čtenář
musel pamatovat, co která pozice znamená:

```json
["Kytice", 0.94, "3f2a…"]
```

Nově je to slovník se čtyřmi pojmenovanými klíči:

```json
{"text": "Kytice", "confidence": 0.94, "lang": "ces", "id": "3f2a…"}
```

`text` je přečtený řetězec, `confidence` jistota, `id` identifikátor této
hodnoty — právě ten uvádějí skupiny v kapitole 3 mezi svými členy.

Klíč `lang` je nový a nese jazyk, ve kterém je hodnota zapsána. To potřebují
souběžné titulní listy a dvojjazyčné abstrakty: u českého článku s anglickým
abstraktem a anglickým souběžným názvem dnes nelze rozlišit, který řetězec je
který. V testovací sadě nese 36 ze 116 záznamů dva nebo tři jazyky napříč názvy,
abstrakty a klíčovými slovy. Odděleně od toho obě sady získaly **pole**
`language` — `<language><languageTerm>`, tedy jazyk, ve kterém je dokument nebo
vnitřní část napsána.

### Na které straně byla hodnota přečtena

Kapitolu a článek popisují dvě strany a čtou se odděleně: vlastní úvodní strana
části a záznam v obsahu, který na část odkazuje. Názvy polí to nyní říkají:

- **bez přípony** — přečteno na vlastní úvodní straně části;
- **`TocPage`** — přečteno v záznamu v obsahu, včetně čísla strany, které záznam
  uvádí.

Obě strany se mohou lišit a schéma netvrdí, že se shodují. Kapitola vedená v
obsahu jako „Počátky písma“ a nadepsaná na vlastní straně „I. Počátky písma“ si
podrží obě čtení.

### Bibliografické úrovně

Pole sdílená úrovněmi **titul**, **svazek**, **číslo** a **příloha**.

| Pole | Úrovně | Uloženo v MODS jako | Poznámka |
|---|---|---|---|
| `id` | vše | — | Identita záznamu v MetaKat. |
| `parent_id` | svazek, číslo, příloha | — | Titul je kořen. Příloha visí na svazku nebo na čísle. |
| `page_id` | vše | — | Strana, ke které je záznam ukotven. Pouze MetaKat. |
| `hierarchy` | titul, svazek | — | `multipart` / `monograph` / `periodical`. Není v MODS. |
| `partNumber` | vše | `titleInfo/partNumber` | Číslo svazku, číslo výtisku, číslo části vícesvazkové monografie. |
| `partName` | vše | `titleInfo/partName` | U ročenek, speciálních a tematických čísel. |
| `title` | vše | `titleInfo/title` | Ročník periodika vlastní název nemá — jeho `titleInfo` připouští jen `partNumber`. |
| `subTitle` | vše | `titleInfo/subTitle` |  |
| `edition` | titul, svazek | `originInfo/edition` | V `originInfo` přílohy DMF tento element neuvádí; na úrovni čísla se mutační vydání řeší opakovaným `titleInfo`. |
| `frequency` | titul, příloha | `originInfo/frequency` | Vlastnost vydávání: titul periodika, nebo příloha vycházející jako samostatná řada. |
| `statementOfResponsibility` | vše | `note @type="statement of responsibility"` | **Nové.** Údaj o odpovědnosti doslova tak, jak je vytištěn, např. „sepsal Vincenc Blahouš“. |
| `publisher` | vše | `originInfo/publisher` (3.6) / `originInfo/agent/namePart` (3.8) |  |
| `placeTerm` | vše | `originInfo/place/placeTerm @type="text"` |  |
| `dateIssued` | vše | `originInfo/dateIssued` |  |
| `copyrightDate` | titul, svazek, příloha | `originInfo/copyrightDate` | **Nové.** „© 1967“ na rubu titulního listu — často jediné datum, které kniha tiskne. |
| `manufacturePublisher` | vše | `originInfo @eventType="manufacture"/publisher` |  |
| `manufacturePlaceTerm` | vše | `originInfo @eventType="manufacture"/place/placeTerm` |  |
| `manufactureDate` | vše | `originInfo @eventType="manufacture"/dateOther` | **Nové.** Zapisuje se jako `<dateOther type="manufacture">`, proto ten název. |
| `seriesName` | svazek | `relatedItem @type="series"/titleInfo/title` |  |
| `seriesPartNumber` | svazek | `relatedItem @type="series"/titleInfo/partNumber` |  |
| `seriesPartName` | svazek | `relatedItem @type="series"/titleInfo/partName` | **Nové.** Z pole 830 $p, název podřady. |
| `language` | vše | `language/languageTerm @type="code"` | **Nové.** Kód `iso639-2b`. Jazyk, ve kterém je dokument napsán. |
| `form` | vše | `physicalDescription/form @authority="marcform"` | **Nové.** Zatím `print` / `manuscript`. Viz otázka 2. |
| `author` | vše | `name` + `role/roleTerm` `aut` |  |
| `illustrator` | vše | `name` + `role/roleTerm` `ill` |  |
| `photographer` | vše | `name` + `role/roleTerm` `pht` |  |
| `translator` | vše | `name` + `role/roleTerm` `trl` |  |
| `editor` | vše | `name` + `role/roleTerm` `edt` |  |
| `redaktor` | vše | `name` + `role/roleTerm` |  |
| `affiliation` | vše | `name/affiliation` | **Nové.** Instituce, ke které se jmenovaná osoba hlásí. |
| `email` | vše | **žádný** | **Nové.** V MODS pro tento údaj element neexistuje. Viz otázka 4. |
| `groups` | vše | — | Seskupení detekcí — viz kapitola 6. |

### Kapitoly a články

Pole sdílená **kapitolou** a **článkem**. Nejdřív pole z vlastní úvodní strany,
pak pole ze strany obsahu.

| Pole | Druh | Uloženo v MODS jako | Poznámka |
|---|---|---|---|
| `id` | obě | — | Identita záznamu v MetaKat. |
| `parent_id` | obě | — | Svazek, číslo, nebo jiná kapitola. |
| `pageIndexStart` | obě | `part @type="pageIndex"/extent/start` | Které skeny vnitřní část zabírá. Seznam — část pokračující na dalších stranách má úseků víc. |
| `pageIndexEnd` | obě | `part @type="pageIndex"/extent/end` |  |
| `title` | obě | `titleInfo/title` | Přečteno na **vlastní úvodní straně** vnitřní části. |
| `subTitle` | obě | `titleInfo/subTitle` | DMF sem výslovně připouští i perex. |
| `abstract` | obě | `abstract` | Seznam — souběžný český a anglický abstrakt je běžný. |
| `keywords` | obě | `subject/topic` | Seznam — jedna položka na každý vytištěný termín. |
| `dateIssued` | článek | `originInfo/dateIssued` **rodiče** | **Nové.** Viz otázka 3. |
| `reviewedWorkTitle` | článek | `relatedItem/titleInfo/title` | **Nové.** Recenzované dílo. |
| `reviewedWorkAuthor` | článek | `relatedItem/name/namePart` | **Nové.** |
| `reviewedWorkImprint` | článek | `relatedItem/originInfo/publisher` | **Nové.** Místo, nakladatel a rok jako jeden vytištěný řádek. Viz otázka 5. |
| `language` | obě | `language/languageTerm @type="code"` | **Nové.** Jazyk, ve kterém je vnitřní část napsána. |
| `articleGenre` | článek | `genre @type=…` | **Nové.** `review`, `interview`, `cover`, `tableOfContents`. Viz otázka 2. |
| `pageIndexTocPage` | obě | — | Který sken nese záznam v obsahu. |
| `titleTocPage` | obě | `titleInfo/title` | Tentýž název tak, jak je přečten **v záznamu v obsahu**. |
| `subTitleTocPage` | obě | `titleInfo/subTitle` |  |
| `partNumberTocPage` | obě | `titleInfo/partNumber` | Pořadové číslo kapitoly (I., XI., 5.), do názvu se neuvádí. |
| `pageNumberStartTocPage` | obě | `part @type="pageNumber"/extent/start` | Číslo strany, které uvádí záznam v obsahu. |
| `pageNumberEndTocPage` | obě | `part @type="pageNumber"/extent/end` | Pokud záznam uvádí rozsah. |
| `author` | obě | `name` + `role/roleTerm` `aut` |  |
| `illustrator` | obě | `name` + `role/roleTerm` `ill` |  |
| `photographer` | obě | `name` + `role/roleTerm` `pht` |  |
| `translator` | obě | `name` + `role/roleTerm` `trl` |  |
| `editor` | obě | `name` + `role/roleTerm` `edt` |  |
| `redaktor` | obě | `name` + `role/roleTerm` |  |
| `affiliation` | obě | `name/affiliation` | **Nové.** |
| `email` | obě | **žádný** | **Nové.** V MODS pro tento údaj element neexistuje. |
| `groups` | obě | — | Seskupení detekcí — viz kapitola 6. |

### Vědomě vynechané

Do tabulek výše se dostalo jen to, co lze přečíst ze skenu. Signatury, URN:NBN,
věcné třídění, čísla národních autorit i údaje o metadatovém záznamu v
katalogizačních záznamech převažují, ale vznikají v katalogu nebo v procesu
digitalizace. Následující přehled je uveden proto, aby jejich nepřítomnost
nevypadala jako opomenutí.

| Vynecháno | Důvod |
|---|---|
| `identifier` — `uuid`, `urnnbn`, `ccnb`, `oclc`, `sysno`, `barcode` | Vzniká v procesu digitalizace nebo je přiděluje katalog. Na dokumentu jsou vytištěny jen ISBN a ISSN, obojí je mezi navrhovanými doplňky. |
| `location` — `physicalLocation`, `shelfLocator`, `url` | Sigla a signatura: vlastnost konkrétního exempláře, ne vydání. |
| `recordInfo` — všechny subelementy | Metadata o metadatovém záznamu. |
| `classification` (Konspekt, MDT), `subject @authority="czenas"` | Řízené slovníky, které přiděluje katalogizátor. Vytištěná klíčová slova jsou něco jiného a v záběru jsou. |
| `name/nameIdentifier` | Číslo národní autority. Vytištěné jméno v záběru je, jeho autoritní identifikátor ne. |
| `placeTerm @type="code"` (`marccountry`) | Tvar „xr“. Ve vzorku je každé opakované `place` dvojicí kód a text pro jedno místo, nikdy ne dvě místa. |
| `issuance`, `typeOfResource` | Vyplývá z hierarchie dokumentu, ne ze čtení strany. |
| `nonSort` | Ve 143 vzorových záznamech nula výskytů; čeština ani polština člen nemají. |
| Obecná `note` | Ze 76 poznámek ve vzorku je 26 údaj o odpovědnosti (nyní vlastní pole), 13 jazykových, 4 opakují typ strany a zbylých 25 jsou komentáře katalogizátora k celé řadě, které žádná jednotlivá strana neobsahuje. |

### Zatím jen navržené

| Pole | MODS | Proč |
|---|---|---|
| `isbn`, `issn` | `identifier @type` | Vytištěno v tiráži nebo jako čárový kód a lze ověřit kontrolní číslicí. |
| `extent` | `physicalDescription/extent` | „176 stran.“ Přečteno z tiráže, nebo dopočítáno z počtu stran. |
| `level` | — | Hloubka zanoření kapitoly. Potřebná k zachycení oddílu obsahu typu „Část I: Starověk“, který logická strukturální mapa umí vyjádřit jen jako nadřazenou kapitolu. |

---

## 3. Vázání detekcí do skupin

Že hodnoty patří k sobě, říká v MODS kontejnerový element. Je to vlastní pravidlo
DMF, uvedené v popisu elementu `<originInfo>`:

> …v případě, že je v jednom poli 260/264 uvedeno opakované podpole $a nebo $b,
> je možné příslušné subelementy opakovat v rámci jednoho `<originInfo>` nebo se
> **zopakuje celý `<originInfo>` tak, aby se neztratily vzájemné vazby mezi
> subelementy** (např. mezi konkrétním místem vydání a vydavatelem).

Každý záznam proto nese seznam skupin. Skupina má typ a seznam identifikátorů,
nic víc:

```
type: titleInfo | originInfoPublication | originInfoManufacture
    | agent | series | reviewedWork | pageRange
members: [identifikátory hodnot, které patří k sobě]
```

Typ pojmenovává kontejnerový element MODS, do kterého by členové patřili, takže
čtenář ví, o co ve skupině jde, aniž by ji musel rozebírat. Dvě vydavatelské
události na jednom titulním listu jsou dvě skupiny `originInfoPublication`;
autor se svou afiliací a e-mailem je jedna skupina `agent`.

Dvě vlastnosti jsou záměrné:

**Skupina obsahuje, nevykládá.** Typ říká, do kterého kontejneru členové patří,
a nic dalšího. Skupina `titleInfo`, která drží název přečtený v obsahu a týž
název přečtený na úvodní straně, prostě drží oba; rozhodnout, že jde o jeden
název, je věc čtenáře.

**Seskupení není nikdy podmínkou.** Protože DMF připouští obě varianty zápisu,
záznam bez jediné skupiny je platný — subelementy se zopakují uvnitř jednoho
kontejneru. Částečné seskupení, kdy jsou svázáni dva nakladatelé ze tří, je
normální stav.

---

## 4. Otázky k projednání

Body, ve kterých návrh stojí na předpokladu, který by měla archivní strana
potvrdit nebo opravit.

**1. MODS 3.6, nebo 3.8.** Balíčky, které dnes MetaKat vidí, jsou v MODS 3.6,
ale DMF monografie 2.2 / periodika 2.1 (prosinec 2024) přechází na 3.8 a jedna
ze změn dopadá přímo na nakladatelské údaje: `<originInfo><publisher>` je
nahrazen `<originInfo><agent><namePart>`. ProArc stále vydává 3.6. Na kterou
verzi má MetaKat mířit a existuje termín, do kdy musí být výstup v 3.8?

**2. Odvozené údaje: chceme je, a jak je zapisovat?** Vedle údajů, které se ze
strany opisují, umí MetaKat z textu odvozovat i údaje jiného druhu — jazyk,
písmo, žánr, tematické zařazení a podobně. Nejsou to přepsané řetězce, ale
zatřídění, a dají se dělat na různé úrovni podrobnosti: pro stranu, pro vnitřní
část, pro číslo i pro svazek.

V návrhu jsou zatím tři, protože jen k nim se zatím našel odpovídající element
MODS:

   - `language` (`language/languageTerm`) — jazyk dokumentu nebo vnitřní části;
   - `form` (`physicalDescription/form`) — zatím `print`, `manuscript`; DMF
     hodnoty nevyjmenovává a odkazuje na pole 008/23 MARC 21;
   - `articleGenre` (`genre @type`) — `review`, `interview`, `cover`,
     `tableOfContents`; těmto čtyřem odpovídají v mapování položky „typ článku –
     recenze / rozhovor / obálka / obsah“, úplný výčet DMF odkazuje do Pravidel
     pro popis periodik v. 8.7.

Ty tři jsou ale jen to, co se podařilo namapovat, ne výčet toho, co by MetaKat
uměl dodat. Otázka je proto širší:

   - stojí archivu tento druh údajů za to, a u kterých z nich?
   - na jaké úrovni je chtít — strana, vnitřní část, číslo, svazek, titul?
   - jak je zapisovat? Dnes nesou jen hodnotu a jistotu, protože se nevážou na
     konkrétní místo na stránce: nemají text ani identifikátor, a tedy ani
     způsob, jak je zařadit do skupiny;
   - které řízené slovníky použít tam, kde je standard předepisuje, a co s tím,
     na co v MODS element není?

**3. Datum u článku.** Vnitřní část nemá `<originInfo>`, datum vydání článku
tedy patří číslu, které jej nese. Přesto bývá vytištěné na vlastní straně
článku. V testovací sadě je vyplněné u 72 ze 116 záznamů. Návrh datum ponechává
u článku jako doklad o extrakci a zapisuje je do rodičovského záznamu. Je to
přijatelné, nebo se má zapisovat výhradně k rodiči?

**4. E-mailové adresy.** Adresa korespondujícího autora bývá vytištěná a jde ji
získat, ale MODS pro kontaktní údaje u jmen element nemá — `<name>` připouští
`namePart`, `displayForm`, `affiliation`, `role`, `description`,
`nameIdentifier`, `alternativeName`, `etal` a nic jiného. Má ji MetaKat vést
jako pole, které se nikdy neexportuje, nebo ji vypustit?

**5. Nakladatelské údaje recenzovaného díla.** Mapování drží u položky
„recenzované dílo“ (úroveň `MODS_ART`) místo, nakladatele i rok recenzované
knihy v jednom elementu `<publisher>`, tedy tak, jak to tiskne záhlaví recenze:

```xml
<mods:relatedItem>
  <mods:name type="personal">
    <mods:namePart type="family">Harris</mods:namePart>
    <mods:namePart type="given">Amy</mods:namePart>
  </mods:name>
  <mods:originInfo>
    <mods:publisher>Oxford : Oxford University Press, 2023</mods:publisher>
  </mods:originInfo>
  <mods:titleInfo>
    <mods:title>Being single in Georgian England : families, households, and the unmarried</mods:title>
  </mods:titleInfo>
</mods:relatedItem>
```

V MetaKat tomu odpovídají tři pole; skupina `reviewedWork` je váže dohromady:

```json
"reviewedWorkTitle":   [{"text": "Being single in Georgian England : families, households, and the unmarried", "confidence": 0.94, "lang": "eng", "id": "a1…"}],
"reviewedWorkAuthor":  [{"text": "Amy Harris", "confidence": 0.91, "lang": null, "id": "b2…"}],
"reviewedWorkImprint": [{"text": "Oxford : Oxford University Press, 2023", "confidence": 0.88, "lang": null, "id": "c3…"}],
"groups": [{"type": "reviewedWork", "members": ["a1…", "b2…", "c3…"]}]
```

Nakladatelské údaje tedy zůstávají v jednom poli, nerozdělené na místo,
nakladatele a rok. Je to tak zamýšleno?

**6. Dvojí zápis názvu: dvě pole, nebo klíč u hodnoty?** Kapitola 2 popisuje
současné řešení — název přečtený na úvodní straně je v `title`, název přečtený
v obsahu v `titleTocPage`, totéž u podnázvu, a skupina `titleInfo` obě čtení
sváže. Druhá možnost je mít jen `title` a `subTitle` a doplnit hodnotě další
klíč, který řekne, odkud pochází:

```json
{"text": "Počátky písma", "confidence": 0.95, "lang": "ces", "id": "…", "source": "destinationPage"}
{"text": "Počátky písma", "confidence": 0.80, "lang": "ces", "id": "…", "source": "tocPage"}
```

Pro dvě pole mluví, že je rozdíl vidět už v seznamu polí a že konzument, kterého
obsah nezajímá, prostě čte `title`. Pro klíč u hodnoty mluví, že nerozmnožuje
pole — přípona `TocPage` by jinak musela přibýt u všeho, co lze přečíst na obou
stranách — a že lépe odpovídá tomu, že jde o jednu informaci přečtenou dvakrát.

Klíč `source` v ukázce je přitom nejspíš zbytečný. `id` hodnoty se mapuje na
stranu, na které byla detekována (`detection_to_page_mapping`), a
`pageIndexTocPage` říká, která strana nese záznam v obsahu — z čeho hodnota
pochází, se tedy dá zjistit i bez toho, aby se to u ní zapisovalo. Varianta
s jedním polem tak ve skutečnosti nepotřebuje žádný nový klíč, jen mapování,
které už v datech je.

Týká se to názvu, podnázvu a pořadového čísla kapitoly. To sice MetaKat dnes
čte jen z obsahu, ale bývá vytištěné i na úvodní straně části a časem se odtud
číst má, takže dvojice vznikne i tam. Čísel stran se rozhodnutí netýká — ty se
čtou pouze v obsahu, protože číslo vytištěné na vlastní straně části je vedeno
u záznamu strany.

**7. Má MetaKat vracet rovnou MODS?** Schéma je proti MODS namapované pole po
poli — sloupec „Uloženo v MODS jako“ v tabulkách v kapitole 2 je v podstatě celý
ten převod. Napsat převodník oběma směry je proto v tuto chvíli přímočaré a
otevírá to dvě možnosti:

   - **vracet vedle vlastního JSON rovnou MODS**, aby si převod nemusel psát
     každý konzument sám;
   - **doplňovat MODS, který už existuje**. To je scénář ProArc: balíček
     katalogizační záznam v MODS už nese a MetaKat by do něj přidal, co přečetl
     ze skenů.

K rozhodnutí:

   - stojí archivu MODS na výstupu za to, nebo si převod raději nechá u sebe?
   - u doplňování: co má přednost, když se katalogizační záznam a čtení ze skenu
     liší? Má MetaKat existující hodnoty přepisovat, doplňovat jen to, co
     v záznamu chybí, nebo neshody pouze hlásit?
   - v jaké verzi MODS — viz otázka 1.

**8. Chybí něco?** Pole v tabulkách v kapitole 2 jsou to, co považujeme zároveň
za užitečné a za čitelné ze skenu.
