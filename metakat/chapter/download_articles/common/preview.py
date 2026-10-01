from __future__ import annotations

import re
import shutil
import textwrap
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from metakat.chapter.download_articles.common.models import StoredArticle
from metakat.chapter.download_articles.common.store import ArticleStore

_FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
TILE_WIDTH, TILE_HEIGHT, PAD, COLUMNS = 300, 420, 10, 4
# Journals with fewer samples share a timeline sheet, one row each.
MIN_OWN_TIMELINE = 4


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = _FONT_DIR / ("DejaVuSansCondensed-Bold.ttf" if bold else "DejaVuSansCondensed.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default(size)


def render_previews(store: ArticleStore) -> list[Path]:
    """Render phone-sized contact sheets of the stored title pages into ``<library>/previews``.

    ``overview_NN.jpg`` shows one title page per journal, nine per sheet; ``timelines/`` shows every
    stored title page of a journal in year order, labelled with year and volume/issue. Earlier
    previews are replaced.
    """
    by_journal: dict[str, list[StoredArticle]] = defaultdict(list)
    for article in store.stored():
        by_journal[article.item.journal_title or article.item.journal_id or "?"].append(article)
    for articles in by_journal.values():
        articles.sort(key=lambda a: (a.item.year or 0, a.item.item_id))
    journals = sorted(by_journal.items(), key=lambda kv: (kv[1][0].item.year or 0, kv[0]))

    out = store.dir / "previews"
    if out.exists():
        shutil.rmtree(out)
    (out / "timelines").mkdir(parents=True)
    files = _overview(store, journals, out)
    files += _timelines(store, journals, out / "timelines")
    return files


def _thumbnail(store: ArticleStore, article: StoredArticle, width: int, height: int) -> Image.Image:
    with Image.open(store.dir / article.image.file) as image:
        image = image.convert("L")
        image.thumbnail((width, height))
        return image


def _label(article: StoredArticle) -> str:
    item = article.item
    volume = (item.volume or "").lstrip("0") or "?"
    return f"{item.year or '?'}  v{volume}/{item.issue or '?'}"


def _tile(sheet, draw, store, article, x, y, width, height, font):
    image = _thumbnail(store, article, width, height)
    sheet.paste(image, (x + (width - image.width) // 2, y))
    draw.rectangle([x, y, x + width - 1, y + height - 1], outline=(170, 170, 170))
    draw.text((x + 4, y + height + 4), _label(article), font=font, fill="black")


def _overview(store, journals, out: Path) -> list[Path]:
    width, height, caption, columns = 420, 560, 84, 3
    bold, regular = _font(22, bold=True), _font(22)
    files = []
    for page_number, start in enumerate(range(0, len(journals), 9), 1):
        page = journals[start:start + 9]
        rows = (len(page) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * (width + PAD) + PAD, rows * (height + caption + PAD) + PAD), "white")
        draw = ImageDraw.Draw(sheet)
        for k, (title, articles) in enumerate(page):
            x = PAD + (k % columns) * (width + PAD)
            y = PAD + (k // columns) * (height + caption + PAD)
            article = articles[len(articles) // 2]
            image = _thumbnail(store, article, width, height)
            sheet.paste(image, (x + (width - image.width) // 2, y))
            draw.rectangle([x, y, x + width - 1, y + height - 1], outline=(180, 180, 180))
            draw.text((x, y + height + 6), textwrap.shorten(title, 36, placeholder="…"), font=bold, fill="black")
            years = [a.item.year for a in articles if a.item.year]
            span = f"{min(years)}–{max(years)}" if years else "?"
            draw.text((x, y + height + 34), f"shown {article.item.year} · {len(articles)} samples {span}",
                      font=regular, fill=(80, 80, 80))
        path = out / f"overview_{page_number:02d}.jpg"
        sheet.save(path, quality=82)
        files.append(path)
    return files


def _timelines(store, journals, out: Path) -> list[Path]:
    title_font, label_font = _font(30, bold=True), _font(24, bold=True)
    sheet_width = COLUMNS * (TILE_WIDTH + PAD) + PAD
    files = []
    own = [(t, a) for t, a in journals if len(a) >= MIN_OWN_TIMELINE]
    shared = [(t, a) for t, a in journals if len(a) < MIN_OWN_TIMELINE]

    for number, (title, articles) in enumerate(own, 1):
        rows = (len(articles) + COLUMNS - 1) // COLUMNS
        head = 50
        sheet = Image.new("RGB", (sheet_width, head + rows * (TILE_HEIGHT + 34 + PAD) + PAD), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((PAD, 8), f"{title[:60]} ({len(articles)})", font=title_font, fill="black")
        for k, article in enumerate(articles):
            _tile(sheet, draw, store, article, PAD + (k % COLUMNS) * (TILE_WIDTH + PAD),
                  head + (k // COLUMNS) * (TILE_HEIGHT + 34 + PAD), TILE_WIDTH, TILE_HEIGHT, label_font)
        path = out / f"{number:02d}_{_slug(title)}.jpg"
        sheet.save(path, quality=80)
        files.append(path)

    row_height = 44 + TILE_HEIGHT + 34 + PAD
    for group_start in range(0, len(shared), 4):
        group = shared[group_start:group_start + 4]
        sheet = Image.new("RGB", (sheet_width, len(group) * row_height + PAD), "white")
        draw = ImageDraw.Draw(sheet)
        for row, (title, articles) in enumerate(group):
            y = PAD + row * row_height
            draw.text((PAD, y), title[:62], font=label_font, fill="black")
            for k, article in enumerate(articles[:COLUMNS]):
                _tile(sheet, draw, store, article, PAD + k * (TILE_WIDTH + PAD), y + 36,
                      TILE_WIDTH, TILE_HEIGHT, label_font)
        path = out / f"{len(own) + group_start // 4 + 1:02d}_short_lived.jpg"
        sheet.save(path, quality=80)
        files.append(path)
    return files


def _slug(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", title.encode("ascii", "ignore").decode())[:40].strip("_") or "journal"
