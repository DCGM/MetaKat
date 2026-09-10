# Návrh úpravy metadatového schématu MetaKat

## K čemu tento dokument slouží

Je to podklad pro rozhodnutí, ne plán implementace. Popisuje, jak by se
metadatový model, který MetaKat produkuje, změnil oproti stavu na větvi `main`,
a proč je každá změna navržena.

**Nic z toho není uzavřené.** Smyslem sepsání je nechat si model potvrdit — nebo
opravit — dřív, než se proti němu přepíše okolní kód. Kapitola 7 shrnuje otázky,
na které potřebujeme odpověď z archivní strany; zbytek je kontext k nim.

Zkratkou „DMF“ se dále rozumí *Definice metadatových formátů* — DMF pro
digitalizaci monografických dokumentů v. 2.3 a DMF pro digitalizaci periodik
v. 2.2 — a „mapováním“ tabulka `metada_mapping.xlsx`.

Názvy polí, elementů MODS a hodnot řízených slovníků zůstávají v celém dokumentu
v původní podobě, protože odkazují na konkrétní pole v MetaKat, respektive na
konkrétní elementy MODS.

---

## 1. Výchozí stav

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

## 2. Tři principy, ze kterých návrh vychází

**MODS má pro každou úroveň jeden a tentýž element.** Titul, svazek, číslo a
příloha nejsou různé typy záznamu: je to týž element `<mods>`, rozlišený svým
`ID` a hodnotou `<genre>`. Tabulky DMF se pro jednotlivé úrovně liší téměř
výhradně v tom, které elementy jsou *povinné*, nikoli v tom, které vůbec
existují. Sedm rozbíhajících se tříd byl vynález MetaKat.

**Vazby nese kontejnerový element.** To je vlastní pravidlo DMF, uvedené v popisu
elementu `<originInfo>`:

> …v případě, že je v jednom poli 260/264 uvedeno opakované podpole $a nebo $b,
> je možné příslušné subelementy opakovat v rámci jednoho `<originInfo>` nebo se
> **zopakuje celý `<originInfo>` tak, aby se neztratily vzájemné vazby mezi
> subelementy** (např. mezi konkrétním místem vydání a vydavatelem).

Přípustné jsou obě varianty zápisu, což je podstatné: záznam, u kterého se vazby
určit nepodařilo, opakuje subelementy uvnitř jednoho kontejneru a je stále
platný; záznam, u kterého se určit podařilo, opakuje celý kontejner a je
přesnější.

**Jen to, co lze přečíst ze skenu.** MetaKat získává údaje z obrazu. Signatury,
URN:NBN, věcné třídění Konspekt a czenas, MDT, čísla národních autorit i celý
`<recordInfo>` v katalogizačních záznamech převažují, ale vznikají v katalogu
nebo v procesu digitalizace, takže o ně tu nejde. Kapitola 8 uvádí, co bylo z
tohoto důvodu vynecháno.

---

## 3. Každá hodnota se může opakovat

Z každého pole se stal seznam. Záznam s jedním nakladatelem nese jednoprvkový
seznam, titulní list se dvěma nese oba.

Tím se zároveň odstranila nedůslednost: `publisher` už seznamem byl, zatímco
`placeTerm`, `dateIssued` a `edition` byly jednotlivé hodnoty. Schéma tedy
uneslo tři nakladatele, ale jen jedno místo — právě ten vztah, o jehož zachování
DMF žádá, nešlo zaznamenat.

---

## 4. Jedna společná sada polí pro každou hierarchii

Čtyři bibliografické úrovně — titul, svazek, číslo, příloha — mají nyní společnou
sadu polí, a kapitola s článkem druhou.

Obě sady zůstávají oddělené z věcného důvodu: **vnitřní část nemá `<originInfo>`
vůbec.** Kapitola ani článek se nevydávají samostatně, nakladatelské údaje
přebírají od svazku či čísla, které je nese. Nakladatel, místo, všechna data,
vydání, periodicita i trojice údajů o tisku jsou tam tedy nepoužitelné — třináct
polí. Sloučením by téměř polovina zůstala trvale prázdná.

Uvnitř každé sady platí, že dostupnost pole na dané úrovni neznamená, že tam
dává smysl. Ročník periodika legitimně vyplní sotva víc než `partNumber` a
`dateIssued`. Které pole kam patří, zachycují tabulky v příloze — a **záměrně to
zatím není v kódu nijak vynucováno**; ty tabulky jsou zadáním, ze kterého by se
taková kontrola teprve psala.

Jeden důsledek stojí za zmínku: sdílení sady zpřístupňuje všechny role na všech
úrovních, což odpovídá DMF a řeší rozejití popsané v kapitole 1.

---

## 5. Co hodnota nese a na které straně byla přečtena

**Jazyk.** Hodnota může nést jazyk, ve kterém je zapsána. To potřebují souběžné
titulní listy a dvojjazyčné abstrakty: u českého článku s anglickým abstraktem a
anglickým souběžným názvem dnes nelze rozlišit, který řetězec je který. V datové
sadě 116 článků nese 36 záznamů dva nebo tři jazyky napříč názvy, abstrakty a
klíčovými slovy.

Jde o údaj **u jednotlivé hodnoty**. Samotný MODS značkuje kontejner —
`<titleInfo lang="eng" type="translated">` platí pro `title`, `subTitle`,
`partNumber` i `partName` dohromady — takže zápis u hodnoty je podkladem, ze
kterého se ty bloky rekonstruují, ne tvrzením, že jazyk patří hodnotě.

Odděleně od toho obě hierarchie získaly **pole** `language`, což je něco jiného:
`<language><languageTerm>`, tedy jazyk, ve kterém je dokument nebo vnitřní část
napsána. Z jazyků u jednotlivých hodnot ho odvodit nelze — u 29 z 63 článkových
záznamů se jazyk prvního názvu liší od jazyka abstraktu.

**Na které straně byla hodnota přečtena.** Kapitolu a článek popisují dvě strany
a čtou se odděleně: záznam v obsahu, který na část odkazuje, a vlastní úvodní
strana části. Názvy polí to nyní říkají:

- **bez přípony** — přečteno na vlastní úvodní straně části;
- **`TocPage`** — přečteno v záznamu v obsahu, včetně čísla strany vytištěného
  vpravo u toho záznamu.

Původní pojmenování to mělo obráceně: `title`, `subTitle`, `partNumber` i
`pageNumber` se plnily z obsahu, zatímco výslovnou příponu nesl údaj z úvodní
strany.

Obě strany se mohou lišit a schéma netvrdí, že se shodují. Kapitola vedená v
obsahu jako „Počátky písma“ a nadepsaná na vlastní straně „I. Počátky písma“ si
podrží obě čtení.

---

## 6. Zaznamenání toho, které hodnoty patří k sobě

Každý záznam nese seznam skupin. Skupina má typ a seznam identifikátorů, nic
víc:

```
type: titleInfo | originInfoPublication | originInfoManufacture
    | agent | series | reviewedWork | pageRange
members: [identifikátory hodnot, které patří k sobě]
```

Typ pojmenovává kontejnerový element MODS, do kterého by členové patřili, takže
čtenář ví, o co ve skupině jde, aniž by ji musel rozebírat. Dvě vydavatelské
události na jednom titulním listu jsou dvě skupiny `originInfoPublication`;
autor se svou afiliací a e-mailem je jedna skupina `agent`.

Tři vlastnosti jsou záměrné:

**Skupina obsahuje, nevykládá.** Typ říká, do kterého kontejneru členové patří,
a nic dalšího. Skupina `titleInfo`, která drží název přečtený v obsahu a týž
název přečtený na úvodní straně, prostě drží oba; rozhodnout, že jde o jeden
název, je věc čtenáře.

**Seskupení není nikdy podmínkou.** Protože DMF připouští obě varianty zápisu,
záznam bez jediné skupiny je platný — subelementy se zopakují uvnitř jednoho
kontejneru. Částečné seskupení, kdy jsou svázáni dva nakladatelé ze tří, je
normální stav.

**Žádná skupina pro `<subject>`.** Vázala by termíny jednoho bloku klíčových
slov, jenže jazyk u jednotlivé hodnoty je odděluje sám: ze 116 článkových
záznamů dává všech 45, které nesou několik bloků názvů, abstraktů či klíčových
slov, každému bloku odlišný jazyk a žádný se neopakuje.

---

## 7. Otázky k projednání

Body, ve kterých návrh stojí na předpokladu, který by měla archivní strana
potvrdit nebo opravit.

**1. MODS 3.6, nebo 3.8.** Balíčky, které dnes MetaKat vidí, jsou v MODS 3.6,
ale DMF monografie 2.2 / periodika 2.1 (prosinec 2024) přechází na 3.8 a jedna
ze změn dopadá přímo na nakladatelské údaje: `<originInfo><publisher>` je
nahrazen `<originInfo><agent><namePart>`. ProArc stále vydává 3.6. Na kterou
verzi má MetaKat mířit a existuje termín, do kdy musí být výstup v 3.8?

**2. Tiskař na úrovni čísla a název pole.** Řádek 17 mapování přiřazuje tiskaře
k `MODS_ISSUE`, ale tabulka DMF pro číslo uvádí u `originInfo` jen subelementy
vydavatelské události. Očekává se u čísla blok `manufacture`? Odděleně: pole se
dnes jmenuje `manufactureDateIssued`, přitom se zapisuje jako
`<dateOther type="manufacture">` — název `manufactureDate` by odpovídal
standardu a přejmenovat je teď levné.

**3. Dva řízené slovníky, které se nepodařilo uzavřít.**
   - `form` (`physicalDescription/form`) dnes připouští jen `print` a
     `manuscript`. DMF hodnoty nevyjmenovává a odkazuje na pole 008/23 MARC 21.
     Které z těch hodnot má MetaKat umět vyprodukovat?
   - `articleGenre` (`genre @type`) dnes připouští `review`, `interview`,
     `cover` a `tableOfContents` — čtyři hodnoty, které uvádí mapování. Úplný
     výčet DMF odkazuje do Pravidel pro popis periodik v. 8.7. Které hodnoty
     mají smysl?

**4. Datum u článku.** Vnitřní část nemá `<originInfo>`, datum vydání článku
tedy patří číslu, které jej nese. Přesto je vytištěné na vlastní straně článku a
je vyplněné u 72 ze 116 referenčních záznamů. Návrh je ponechává u článku jako
doklad o extrakci a zapisuje je do rodičovského záznamu. Je to přijatelné, nebo
se má zapisovat výhradně k rodiči?

**5. E-mailové adresy.** Adresa korespondujícího autora bývá vytištěná a jde ji
získat, ale MODS pro kontaktní údaje u jmen element nemá — `<name>` připouští
`namePart`, `displayForm`, `affiliation`, `role`, `description`,
`nameIdentifier`, `alternativeName`, `etal` a nic jiného. Má ji MetaKat vést
jako pole, které se nikdy neexportuje, nebo ji vypustit?

**6. Nakladatelské údaje recenzovaného díla.** Řádek 30 mapování drží místo,
nakladatele i rok recenzované knihy v jednom elementu `<publisher>`, tedy tak,
jak to tiskne záhlaví recenze. MetaKat to přebírá jedním polem místo tří. Je to
tak zamýšleno?

**7. Čísla stran vytištěná v obsahu.** Číslo strany ze záznamu v obsahu se vede
u kapitoly či článku a zapisuje se do jeho vlastního `<part type="pageNumber">` —
MODS zaznamenává, jaké to číslo je, ne kde bylo přečteno. Je to zamýšlené
zacházení?

**8. Chybí něco?** Níže uvedená pole jsou to, co považujeme zároveň za užitečné a
za čitelné ze skenu. Pokud archivu chybí něco dalšího, teď je vhodná chvíle to
říct.

---

## 8. Vědomě vynechané

Vše z DMF, co ze skenu získat nelze, uvedené proto, aby to nevypadalo jako
opomenutí.

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

---

## 9. Zatím jen navržené

| Pole | MODS | Proč |
|---|---|---|
| `isbn`, `issn` | `identifier @type` | Vytištěno v tiráži nebo jako čárový kód a lze ověřit kontrolní číslicí. |
| `extent` | `physicalDescription/extent` | „176 stran.“ Přečteno z tiráže, nebo dopočítáno z počtu stran. |
| `level` | — | Hloubka zanoření kapitoly. Potřebná k zachycení oddílu obsahu typu „Část I: Starověk“, který logická strukturální mapa umí vyjádřit jen jako nadřazenou kapitolu. |

---

## Příloha A — bibliografické úrovně

Pole sdílená úrovněmi **titul**, **svazek**, **číslo** a **příloha**, v pořadí
deklarace. Sloupec „Úrovně“ říká, kde pole dává smysl, ne kde je povoleno —
vynucováno není nic.

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
| `manufactureDateIssued` | vše | `originInfo @eventType="manufacture"/dateOther` | **Nové.** K názvu pole viz otázka 2. |
| `seriesName` | svazek | `relatedItem @type="series"/titleInfo/title` |  |
| `seriesPartNumber` | svazek | `relatedItem @type="series"/titleInfo/partNumber` |  |
| `seriesPartName` | svazek | `relatedItem @type="series"/titleInfo/partName` | **Nové.** Z pole 830 $p, název podřady. |
| `language` | vše | `language/languageTerm @type="code"` | **Nové.** Kód `iso639-2b`. Jazyk, ve kterém je dokument napsán. |
| `form` | vše | `physicalDescription/form @authority="marcform"` | **Nové.** Zatím `print` / `manuscript`. Viz otázka 3. |
| `author` | vše | `name` + `role/roleTerm` `aut` |  |
| `illustrator` | vše | `name` + `role/roleTerm` `ill` |  |
| `photographer` | vše | `name` + `role/roleTerm` `pht` |  |
| `translator` | vše | `name` + `role/roleTerm` `trl` |  |
| `editor` | vše | `name` + `role/roleTerm` `edt` |  |
| `redaktor` | vše | `name` + `role/roleTerm` |  |
| `affiliation` | vše | `name/affiliation` | **Nové.** Instituce, ke které se jmenovaná osoba hlásí. |
| `email` | vše | **žádný** | **Nové.** V MODS pro tento údaj element neexistuje. Viz otázka 5. |
| `groups` | vše | — | Seskupení detekcí — viz kapitola 6. |

## Příloha B — kapitoly a články

Pole sdílená **kapitolou** a **článkem**, v pořadí deklarace. Nejdřív pole z
vlastní úvodní strany, pak pole ze strany obsahu.

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
| `dateIssued` | článek | `originInfo/dateIssued` **rodiče** | **Nové.** Viz otázka 4. |
| `reviewedWorkTitle` | článek | `relatedItem/titleInfo/title` | **Nové.** Recenzované dílo. |
| `reviewedWorkAuthor` | článek | `relatedItem/name/namePart` | **Nové.** |
| `reviewedWorkImprint` | článek | `relatedItem/originInfo/publisher` | **Nové.** Místo, nakladatel a rok jako jeden vytištěný řádek. Viz otázka 6. |
| `language` | obě | `language/languageTerm @type="code"` | **Nové.** Jazyk, ve kterém je vnitřní část napsána. |
| `articleGenre` | článek | `genre @type=…` | **Nové.** `review`, `interview`, `cover`, `tableOfContents`. Viz otázka 3. |
| `pageIndexTocPage` | obě | — | Který sken nese záznam v obsahu. |
| `titleTocPage` | obě | `titleInfo/title` | Tentýž název tak, jak je přečten **v záznamu v obsahu**. |
| `subTitleTocPage` | obě | `titleInfo/subTitle` |  |
| `partNumberTocPage` | obě | `titleInfo/partNumber` | Pořadové číslo kapitoly (I., XI., 5.), do názvu se neuvádí. |
| `pageNumberStartTocPage` | obě | `part @type="pageNumber"/extent/start` | Číslo strany vytištěné v záznamu v obsahu, obvykle vpravo. |
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
