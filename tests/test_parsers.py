"""Tests for document parsers."""

from hybrid_rag.ingest.parsers import (
    compute_hash,
    parse_file,
    parse_html,
    parse_markdown,
    parse_txt,
)


def test_compute_hash_deterministic():
    h1 = compute_hash(b"hello world")
    h2 = compute_hash(b"hello world")
    assert h1 == h2
    assert len(h1) == 64


def test_compute_hash_different():
    h1 = compute_hash(b"hello")
    h2 = compute_hash(b"world")
    assert h1 != h2


def test_parse_txt():
    data = b"Hello, this is plain text.\nSecond line."
    result = parse_txt(data, "test.txt")
    assert result.file_type == "txt"
    assert result.filename == "test.txt"
    assert len(result.sections) == 1
    assert "Hello" in result.sections[0].text
    assert result.content_hash == compute_hash(data)


def test_parse_txt_empty():
    result = parse_txt(b"", "empty.txt")
    assert len(result.sections) == 0


def test_parse_markdown_headings():
    md = b"# Title\n\nFirst paragraph.\n\n## Section\n\nSecond paragraph."
    result = parse_markdown(md, "doc.md")
    assert result.file_type == "md"
    assert len(result.sections) >= 2
    headings = [s.heading for s in result.sections if s.heading]
    assert "Title" in headings


def test_parse_html_strips_scripts():
    html = b"<html><body><script>alert(1)</script><p>Content here</p></body></html>"
    result = parse_html(html, "page.html")
    assert result.file_type == "html"
    assert any("Content here" in s.text for s in result.sections)
    assert not any("alert" in s.text for s in result.sections)


def test_parse_html_extracts_headings():
    html = b"<h1>Main Title</h1><p>Body text.</p><h2>Sub</h2><p>More text.</p>"
    result = parse_html(html, "page.html")
    headings = [s.heading for s in result.sections if s.heading]
    assert "Main Title" in headings


def test_parse_file_txt():
    result = parse_file(b"test content", "file.txt")
    assert result.file_type == "txt"


def test_parse_file_unsupported():
    try:
        parse_file(b"data", "file.xyz")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "Unsupported" in str(e)


def test_parse_html_no_duplicate_nested_text():
    """Verify parse_html skips child elements whose parents were already processed,
    preventing duplicate section extraction."""
    html = b"""
    <html><body>
        <table>
            <tr>
                <td><p>Cell 1 content</p></td>
                <td><p>Cell 2 content</p></td>
            </tr>
        </table>
        <ul>
            <li><p>List item content</p></li>
        </ul>
    </body></html>
    """
    result = parse_html(html, "table.html")
    texts = [s.text for s in result.sections]
    assert texts.count("Cell 1 content") == 1
    assert texts.count("Cell 2 content") == 1
    assert texts.count("List item content") == 1
    assert len(result.sections) == 3


def test_parse_markdown_indented_code_blocks():
    """Verify parse_markdown captures indented code blocks (4 spaces)."""
    md = b"""# Code Sample

    def hello():
        print("Hello, world!")

Follow-up paragraph.
"""
    result = parse_markdown(md, "code.md")
    texts = [s.text for s in result.sections]
    assert any('print("Hello, world!")' in t for t in texts)
    assert any("Follow-up paragraph." in t for t in texts)


def test_parse_markdown_empty_heading_does_not_skip_tokens():
    """Verify empty markdown heading does not skip subsequent tokens."""
    md = b"""# \n\n    indented_code()\n\nParagraph text after empty heading."""
    result = parse_markdown(md, "empty_heading.md")
    texts = [s.text for s in result.sections]
    assert any("indented_code()" in t for t in texts)
    assert any("Paragraph text after empty heading." in t for t in texts)
