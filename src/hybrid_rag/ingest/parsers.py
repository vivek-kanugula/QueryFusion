"""Document text extraction for PDF, HTML, MD, and TXT files.

Each parser returns a list of ParsedSection(text, page_number, heading).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass
class ParsedSection:
    text: str
    page_number: int | None = None
    heading: str | None = None


@dataclass
class ParsedDocument:
    filename: str
    file_type: str
    content_hash: str
    size_bytes: int
    sections: list[ParsedSection] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(s.text for s in self.sections if s.text.strip())


def compute_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_pdf(data: bytes, filename: str) -> ParsedDocument:
    import pymupdf

    doc = pymupdf.open(stream=data, filetype="pdf")
    sections: list[ParsedSection] = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            sections.append(ParsedSection(text=text, page_number=page_num))
    doc.close()
    return ParsedDocument(
        filename=filename,
        file_type="pdf",
        content_hash=compute_hash(data),
        size_bytes=len(data),
        sections=sections,
    )


def parse_html(data: bytes, filename: str) -> ParsedDocument:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(data, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    sections: list[ParsedSection] = []
    heading_tags = {"h1", "h2", "h3", "h4", "h5", "h6"}
    current_heading: str | None = None
    parsed_tag_ids: set[int] = set()

    for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "td"]):
        # Skip child tags whose parent was already processed to prevent duplicate text extraction
        if any(id(parent) in parsed_tag_ids for parent in element.parents):
            continue
        if element.name in heading_tags:
            current_heading = element.get_text(strip=True)
        text = element.get_text(separator=" ", strip=True)
        if text:
            parsed_tag_ids.add(id(element))
            sections.append(ParsedSection(text=text, heading=current_heading))

    if not sections:
        body_text = soup.get_text(separator="\n", strip=True)
        if body_text:
            sections.append(ParsedSection(text=body_text))

    return ParsedDocument(
        filename=filename,
        file_type="html",
        content_hash=compute_hash(data),
        size_bytes=len(data),
        sections=sections,
    )


def parse_markdown(data: bytes, filename: str) -> ParsedDocument:
    from markdown_it import MarkdownIt

    text = data.decode("utf-8", errors="replace")
    md = MarkdownIt()
    tokens = md.parse(text)

    sections: list[ParsedSection] = []
    current_heading: str | None = None

    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.type == "heading_open":
            heading_parts: list[str] = []
            close_idx = i + 1
            while close_idx < len(tokens) and tokens[close_idx].type != "heading_close":
                if tokens[close_idx].type == "inline" and tokens[close_idx].content.strip():
                    heading_parts.append(tokens[close_idx].content.strip())
                close_idx += 1
            current_heading = " ".join(heading_parts) if heading_parts else None
            i = close_idx + 1
            continue
        if token.type == "inline" and token.content.strip():
            sections.append(ParsedSection(text=token.content, heading=current_heading))
        elif token.type in ("fence", "code_block") and token.content.strip():
            sections.append(ParsedSection(text=token.content, heading=current_heading))
        i += 1

    if not sections and text.strip():
        sections.append(ParsedSection(text=text))

    return ParsedDocument(
        filename=filename,
        file_type="md",
        content_hash=compute_hash(data),
        size_bytes=len(data),
        sections=sections,
    )


def parse_txt(data: bytes, filename: str) -> ParsedDocument:
    text = data.decode("utf-8", errors="replace")
    sections = [ParsedSection(text=text)] if text.strip() else []
    return ParsedDocument(
        filename=filename,
        file_type="txt",
        content_hash=compute_hash(data),
        size_bytes=len(data),
        sections=sections,
    )


PARSERS = {
    "pdf": parse_pdf,
    "html": parse_html,
    "htm": parse_html,
    "md": parse_markdown,
    "txt": parse_txt,
}


def parse_file(data: bytes, filename: str) -> ParsedDocument:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    parser = PARSERS.get(ext)
    if parser is None:
        raise ValueError(f"Unsupported file type: .{ext}")
    return parser(data, filename)
