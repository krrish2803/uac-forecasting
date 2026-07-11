#!/usr/bin/env python3
"""Convert research_paper.md to a formatted PDF using fpdf2."""

from fpdf import FPDF
import re

class PaperPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, sanitize("Predictive Forecasting of Care Load & Placement Demand: HHS UAC Program"), align="C")
            self.ln(4)
            self.set_draw_color(200, 200, 200)
            self.line(15, self.get_y(), self.w - 15, self.get_y())
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, sanitize(f"Page {self.page_no()}/{{nb}}"), align="C")

    def chapter_title(self, title, level=1):
        title = sanitize(title)
        if level == 1:
            self.set_font("Helvetica", "B", 16)
            self.set_text_color(20, 60, 120)
            self.ln(6)
            self.multi_cell(0, 9, title)
            self.set_draw_color(20, 60, 120)
            self.line(15, self.get_y() + 1, self.w - 15, self.get_y() + 1)
            self.ln(6)
        elif level == 2:
            self.set_font("Helvetica", "B", 13)
            self.set_text_color(40, 80, 140)
            self.ln(5)
            self.multi_cell(0, 8, title)
            self.ln(3)
        elif level == 3:
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(60, 60, 60)
            self.ln(3)
            self.multi_cell(0, 7, title)
            self.ln(2)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        # Handle bold
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = sanitize(text)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bold_text(self, text):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, sanitize(text))
        self.ln(1)

    def bullet_item(self, text, indent=0):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        x = self.get_x()
        self.set_x(x + 5 + indent * 5)
        # Handle bold in bullet text
        parts = re.split(r'(\*\*.*?\*\*)', text)
        bullet = "-  "
        self.set_font("Helvetica", "", 10)
        self.write(5.5, bullet)
        for part in parts:
            if part.startswith('**') and part.endswith('**'):
                self.set_font("Helvetica", "B", 10)
                self.write(5.5, sanitize(part[2:-2]))
                self.set_font("Helvetica", "", 10)
            else:
                self.write(5.5, sanitize(part))
        self.ln(6)

    def numbered_item(self, number, text, indent=0):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        x = self.get_x()
        self.set_x(x + 5 + indent * 5)
        num_str = f"{number}.  "
        self.set_font("Helvetica", "B", 10)
        w = self.get_string_width(num_str)
        self.cell(w, 5.5, num_str)
        self.set_font("Helvetica", "", 10)
        # Handle bold in text
        parts = re.split(r'(\*\*.*?\*\*)', text)
        for part in parts:
            if part.startswith('**') and part.endswith('**'):
                self.set_font("Helvetica", "B", 10)
                self.write(5.5, sanitize(part[2:-2]))
                self.set_font("Helvetica", "", 10)
            else:
                self.write(5.5, sanitize(part))
        self.ln(6)

    def draw_table(self, headers, rows, col_widths=None):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(230, 240, 250)
        self.set_text_color(20, 40, 80)
        self.set_draw_color(180, 180, 180)

        if col_widths is None:
            available = self.w - 30
            col_widths = [available / len(headers)] * len(headers)

        # Header
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, sanitize(h.strip()), border=1, fill=True, align="C")
        self.ln()

        # Rows
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(30, 30, 30)
        fill = False
        for row in rows:
            if self.get_y() > self.h - 25:
                self.add_page()
                # Redraw header
                self.set_font("Helvetica", "B", 9)
                self.set_fill_color(230, 240, 250)
                self.set_text_color(20, 40, 80)
                for i, h in enumerate(headers):
                    self.cell(col_widths[i], 7, sanitize(h.strip()), border=1, fill=True, align="C")
                self.ln()
                self.set_font("Helvetica", "", 8.5)
                self.set_text_color(30, 30, 30)
                fill = False

            if fill:
                self.set_fill_color(245, 248, 252)
            else:
                self.set_fill_color(255, 255, 255)

            for i, cell in enumerate(row):
                cell_text = sanitize(cell.strip())
                # Bold winner rows
                if cell_text.startswith('**') and cell_text.endswith('**'):
                    self.set_font("Helvetica", "B", 8.5)
                    cell_text = cell_text[2:-2]
                align = "L" if i == 0 else "C"
                self.cell(col_widths[i], 6.5, cell_text, border=1, fill=True, align=align)
                self.set_font("Helvetica", "", 8.5)
            self.ln()
            fill = not fill
        self.ln(4)


def sanitize(text):
    """Replace non-latin1 characters with ASCII equivalents."""
    replacements = {
        '\u2014': ' -- ',   # em dash
        '\u2013': '-',      # en dash
        '\u2018': "'",      # left single quote
        '\u2019': "'",      # right single quote
        '\u201c': '"',      # left double quote
        '\u201d': '"',      # right double quote
        '\u2022': '-',      # bullet
        '\u2026': '...',    # ellipsis
        '\u00b0': ' degrees', # degree sign
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # Fallback: strip any remaining non-latin1
    return text.encode('latin-1', errors='replace').decode('latin-1')


def parse_markdown(md_path):
    """Parse markdown into structured blocks."""
    with open(md_path, 'r') as f:
        lines = f.readlines()

    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip('\n')

        # Blank line
        if not line.strip():
            i += 1
            continue

        # Horizontal rule
        if line.strip() in ('---', '***', '___'):
            blocks.append(('hr', ''))
            i += 1
            continue

        # Headings
        m = re.match(r'^(#{1,3})\s+(.+)', line)
        if m:
            level = len(m.group(1))
            blocks.append(('heading', (level, m.group(2).strip())))
            i += 1
            continue

        # Table
        if '|' in line and i + 1 < len(lines) and re.match(r'^[\s|:-]+$', lines[i+1]):
            table_lines = []
            while i < len(lines) and '|' in lines[i]:
                table_lines.append(lines[i])
                i += 1
            # Parse table
            headers = [c.strip() for c in table_lines[0].split('|') if c.strip()]
            rows = []
            for tl in table_lines[2:]:  # skip header and separator
                row = [c.strip() for c in tl.split('|') if c.strip() != '']
                # re-split to handle empty leading/trailing
                parts = tl.split('|')
                row = [p.strip() for p in parts[1:-1]]  # skip first and last empty
                if row:
                    rows.append(row)
            blocks.append(('table', (headers, rows)))
            continue

        # Numbered list
        m = re.match(r'^(\d+)\.\s+(.+)', line)
        if m:
            blocks.append(('numbered', (m.group(1), m.group(2))))
            i += 1
            continue

        # Bullet list
        m = re.match(r'^[-*]\s+(.+)', line)
        if m:
            blocks.append(('bullet', m.group(1)))
            i += 1
            continue

        # Italic line (for footer)
        if line.strip().startswith('*') and line.strip().endswith('*') and not line.strip().startswith('**'):
            blocks.append(('italic', line.strip().strip('*')))
            i += 1
            continue

        # Regular paragraph
        blocks.append(('text', line))
        i += 1

    return blocks


def build_pdf(blocks, output_path):
    pdf = PaperPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Title page
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(20, 50, 100)
    pdf.ln(30)
    title = sanitize("Predictive Forecasting of Care Load & Placement Demand")
    pdf.multi_cell(0, 12, title, align="C")
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(80, 80, 80)
    subtitle = sanitize("HHS Unaccompanied Alien Children Program")
    pdf.cell(0, 10, subtitle, align="C")
    pdf.ln(15)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, sanitize("Unified Mentor Project"), align="C")
    pdf.ln(8)
    pdf.cell(0, 8, sanitize("July 2026"), align="C")
    pdf.ln(15)
    pdf.set_draw_color(20, 50, 100)
    pdf.line(60, pdf.get_y(), pdf.w - 60, pdf.get_y())
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 6, sanitize("Data source: HHS Office of Refugee Resettlement\n"
                   "Period: January 2023 - December 2025\n"
                   "1,061 daily observations | 46 engineered features | 6 models"), align="C")

    # Content pages
    pdf.add_page()

    for block_type, content in blocks:
        if block_type == 'heading':
            level, text = content
            pdf.chapter_title(text, level)

        elif block_type == 'text':
            pdf.body_text(content)

        elif block_type == 'bullet':
            pdf.bullet_item(content)

        elif block_type == 'numbered':
            num, text = content
            pdf.numbered_item(num, text)

        elif block_type == 'table':
            headers, rows = content
            n_cols = len(headers)
            available = pdf.w - 30
            if n_cols <= 4:
                col_widths = [available / n_cols] * n_cols
            elif n_cols == 5:
                col_widths = [available * 0.22, available * 0.195, available * 0.195, available * 0.195, available * 0.195]
            elif n_cols == 6:
                col_widths = [available * 0.22, available * 0.156, available * 0.156, available * 0.156, available * 0.156, available * 0.156]
            else:
                col_widths = [available / n_cols] * n_cols
            pdf.draw_table(headers, rows, col_widths)

        elif block_type == 'italic':
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(120, 120, 120)
            pdf.cell(0, 6, sanitize(content), align="C")
            pdf.ln(6)

        elif block_type == 'hr':
            pdf.set_draw_color(180, 180, 180)
            pdf.line(15, pdf.get_y(), pdf.w - 15, pdf.get_y())
            pdf.ln(4)

    pdf.output(output_path)
    return output_path


if __name__ == "__main__":
    md_path = "research_paper.md"
    out_path = "outputs/Research_Paper_HHS_UAC_Forecasting.pdf"
    blocks = parse_markdown(md_path)
    result = build_pdf(blocks, out_path)
    print(f"PDF saved: {result}")
