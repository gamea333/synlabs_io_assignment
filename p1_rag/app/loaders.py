"""Turn PDF / HTML / MD files into plain text, keeping page numbers for PDFs."""
import re
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

SUPPORTED = {".pdf": "pdf", ".html": "html", ".htm": "html", ".md": "md", ".markdown": "md"}


@dataclass
class Document:
    source: str            # path relative to the corpus root, stable across machines
    doc_type: str          # pdf | html | md  (used as the metadata filter)
    title: str
    text: str
    # (char_offset, page_number) pairs marking where each PDF page starts in `text`
    page_starts: list[tuple[int, int]] = field(default_factory=list)

    def page_at(self, char_offset: int) -> int | None:
        page = None
        for start, num in self.page_starts:
            if start > char_offset:
                break
            page = num
        return page


def _clean(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)      # de-hyphenate PDF line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: Path, source: str) -> Document:
    reader = PdfReader(path)
    parts, page_starts, offset = [], [], 0
    for i, page in enumerate(reader.pages, start=1):
        page_text = _clean(page.extract_text() or "")
        if not page_text:
            continue
        page_starts.append((offset, i))
        parts.append(page_text)
        offset += len(page_text) + 2
    title = (reader.metadata.title if reader.metadata and reader.metadata.title else path.stem)
    return Document(source, "pdf", title, "\n\n".join(parts), page_starts)


def load_html(path: Path, source: str) -> Document:
    soup = BeautifulSoup(path.read_bytes(), "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else path.stem
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "sup", "table", "figure"]):
        tag.decompose()
    # Wikipedia boilerplate sections that add noise but no answers
    for sel in [".reflist", ".navbox", ".mw-references-wrap", ".hatnote", ".mw-editsection"]:
        for tag in soup.select(sel):
            tag.decompose()
    blocks = [el.get_text(" ", strip=True) for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre"])]
    text = "\n\n".join(b for b in blocks if b)
    return Document(source, "html", title, _clean(text))


def load_md(path: Path, source: str) -> Document:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    raw = re.sub(r"<!--.*?-->", "", raw, flags=re.S)       # html comments
    raw = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", raw)          # images / badges
    raw = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", raw)      # keep link text only
    raw = re.sub(r"<[^>]+>", "", raw)                        # inline html tags
    m = re.search(r"^#\s+(.+)$", raw, flags=re.M)
    title = m.group(1).strip() if m else path.stem
    return Document(source, "md", title, _clean(raw))


def load_file(path: Path, root: Path) -> Document:
    source = path.relative_to(root).as_posix()
    kind = SUPPORTED[path.suffix.lower()]
    return {"pdf": load_pdf, "html": load_html, "md": load_md}[kind](path, source)


def discover(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
