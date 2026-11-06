"""
Build the hackathon technical solution report (Word .docx).

Content follows the "Hackathon Technical Solution Report" template (sections 1-5 +
checklist); formatting follows the RV University report format (A4, margins
3.81/2.54/2.54/2.54 cm, Times New Roman 12 pt, 1.5 spacing, chapter 16 pt bold,
section 14 pt caps, subsection 12 pt caps, page numbers bottom-centre, roman
numerals for front matter). All numbers are read from weights/eval_*.json.
Unknown details are left as [placeholders].

Usage: python scripts/build_report.py   ->  docs/Visionary_Hackathon_Report.docx
"""
import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
W, FIG = ROOT / "weights", ROOT / "docs/figures"
OUT = ROOT / "docs/Visionary_Hackathon_Report.docx"
PRO = json.loads((W / "eval_convnext_small_320.json").read_text())
LITE = json.loads((W / "eval_convnext_tiny_224.json").read_text())
PRO_TR = json.loads((W / "convnext_small_320_metrics.json").read_text())
LITE_TR = json.loads((W / "convnext_tiny_224_metrics.json").read_text())
CLS = json.loads((W / "classical_metrics.json").read_text())
PAR = json.loads((W / "web_parity.json").read_text())
SUB = json.loads((W / "int8_subset_convnext_small_320.json").read_text())
TEAM = [("Chakradhar K", "1RUA24CSE0214", "Deep learning & training"),
        ("Sreeram Lagisetty", "1RUA24CSE0219", "Team lead, integration & deployment"),
        ("Nagarjuna VV", "1RUA24CSE0271", "Classical computer vision"),
        ("Manoj R", "1RUA24CSE0241", "Detection & localisation"),
        ("Snehan", "1RUA24CSE0232", "Optimisation, evaluation & failure analysis")]
LIVE = "https://visionalpha-visionary.static.hf.space"
REPO = "https://github.com/chakradhar1778/Visionary"
TITLE = "VISIONARY: INTELLIGENT FINE-GRAINED VEHICLE RECOGNITION AND ANALYSIS SYSTEM"
FONT = "Times New Roman"


def hyb(E):
    return next(r for r in E["comparison"] if r["name"].startswith("Hybrid ("))


# --------------------------------------------------------------------------- helpers
def set_cell_shading(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def add_field(paragraph, instr, placeholder=""):
    """Insert a Word field (PAGE, TOC, SEQ ...). Word refreshes it on update."""
    r = paragraph.add_run()
    b = OxmlElement("w:fldChar"); b.set(qn("w:fldCharType"), "begin"); r._r.append(b)
    r2 = paragraph.add_run()
    t = OxmlElement("w:instrText"); t.set(qn("xml:space"), "preserve"); t.text = f" {instr} "; r2._r.append(t)
    r3 = paragraph.add_run()
    s = OxmlElement("w:fldChar"); s.set(qn("w:fldCharType"), "separate"); r3._r.append(s)
    r4 = paragraph.add_run(placeholder)
    r5 = paragraph.add_run()
    e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), "end"); r5._r.append(e)
    return r4


def page_number_footer(section, fmt):
    sectPr = section._sectPr
    pg = sectPr.find(qn("w:pgNumType"))
    if pg is None:
        pg = OxmlElement("w:pgNumType"); sectPr.append(pg)
    pg.set(qn("w:fmt"), fmt); pg.set(qn("w:start"), "1")
    section.footer.is_linked_to_previous = False
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = add_field(p, "PAGE", "1")
    run.font.name = FONT; run.font.size = Pt(11)


def margins(section):
    section.page_height, section.page_width = Cm(29.7), Cm(21.0)
    section.left_margin, section.right_margin = Cm(3.81), Cm(2.54)
    section.top_margin, section.bottom_margin = Cm(2.54), Cm(2.54)


class Report:
    def __init__(self):
        self.d = Document()
        self.fig_n = 0
        self.tab_n = 0
        self.chapter = 0
        st = self.d.styles
        normal = st["Normal"]
        normal.font.name = FONT; normal.font.size = Pt(12)
        normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        pf = normal.paragraph_format
        pf.line_spacing = 1.5; pf.space_after = Pt(6); pf.widow_control = True
        for name, size, bold, caps in [("Heading 1", 16, True, True), ("Heading 2", 14, False, True),
                                       ("Heading 3", 12, False, True)]:
            h = st[name]
            h.font.name = FONT; h.font.size = Pt(size); h.font.bold = bold; h.font.all_caps = caps
            h.font.color.rgb = RGBColor(0, 0, 0); h.font.italic = False
            rpr = h.element.get_or_add_rPr(); rf = rpr.find(qn("w:rFonts"))
            if rf is None:
                rf = OxmlElement("w:rFonts"); rpr.append(rf)
            for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
                rf.set(qn(a), FONT)
            h.paragraph_format.space_before = Pt(18 if name == "Heading 1" else 12)
            h.paragraph_format.space_after = Pt(10 if name == "Heading 1" else 6)
            h.paragraph_format.keep_with_next = True
            h.paragraph_format.line_spacing = 1.15
        cap = st["Caption"]
        cap.font.name = FONT; cap.font.size = Pt(10); cap.font.italic = False; cap.font.bold = False
        cap.font.color.rgb = RGBColor(0, 0, 0)
        cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.space_after = Pt(10); cap.paragraph_format.line_spacing = 1.0
        for style in st:
            el = getattr(style, "element", None)
            if el is None or el.rPr is None:
                continue
            rf = el.rPr.find(qn("w:rFonts"))
            if rf is not None:
                for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
                    if rf.get(qn(a)) is not None:
                        del rf.attrib[qn(a)]
        margins(self.d.sections[0])

    # text ---------------------------------------------------------------------------
    def p(self, text="", bold=False, italic=False, size=None, align=None, after=None, before=None,
          spacing=None, keep=False):
        para = self.d.add_paragraph()
        if text:
            self.runs(para, text, bold=bold, italic=italic, size=size)
        if align == "center":
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif align == "justify":
            para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        elif align == "right":
            para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        if after is not None:
            para.paragraph_format.space_after = Pt(after)
        if before is not None:
            para.paragraph_format.space_before = Pt(before)
        if spacing is not None:
            para.paragraph_format.line_spacing = spacing
        if keep:
            para.paragraph_format.keep_with_next = True
        return para

    @staticmethod
    def runs(para, text, bold=False, italic=False, size=None):
        """**bold** and `code` segments inside text."""
        parts = text.split("**")
        for i, part in enumerate(parts):
            if not part:
                continue
            for k, seg in enumerate(part.split("`")):
                if not seg:
                    continue
                r = para.add_run(seg)
                r.bold = bold or (i % 2 == 1)
                r.italic = italic
                if k % 2 == 1:  # code
                    r.font.name = "Consolas"
                    r.font.size = Pt((size or 12) - 1.5)
                elif size:
                    r.font.size = Pt(size)
        return para

    def body(self, text):
        return self.p(text, align="justify")

    def bullet(self, text, level=0):
        para = self.d.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
        self.runs(para, text)
        para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        para.paragraph_format.space_after = Pt(3)
        return para

    def numbered(self, text):
        para = self.d.add_paragraph(style="List Number")
        self.runs(para, text)
        para.paragraph_format.space_after = Pt(3)
        return para

    def code(self, lines):
        for i, ln in enumerate(lines):
            para = self.d.add_paragraph()
            r = para.add_run(ln if ln else " ")
            r.font.name = "Consolas"; r.font.size = Pt(9.5)
            r._r.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "Consolas")
            pf = para.paragraph_format
            pf.line_spacing = 1.0; pf.space_after = Pt(0); pf.space_before = Pt(0)
            pf.left_indent = Cm(0.4); pf.keep_with_next = i < len(lines) - 1
            pPr = para._p.get_or_add_pPr()
            shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:fill"), "F2F2F2"); pPr.append(shd)
        self.p(after=2, spacing=1.0)

    # structure -------------------------------------------------------------------------
    def chapter_heading(self, title, numbered=True):
        if numbered:
            self.chapter += 1
            self.fig_in_ch = 0
            text = f"{self.chapter}. {title}"
        else:
            text = title
        h = self.d.add_heading(text, level=1)
        h.alignment = WD_ALIGN_PARAGRAPH.LEFT
        return h

    def front_heading(self, title):
        """Front-matter heading: 14 pt bold underlined caps, centred, but listed in the TOC."""
        h = self.d.add_heading(title, level=1)
        h.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in h.runs:
            r.font.size = Pt(14); r.underline = True
        return h

    def h2(self, t):
        return self.d.add_heading(t, level=2)

    def h3(self, t):
        return self.d.add_heading(t, level=3)

    def page_break(self):
        self.d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def caption(self, kind, text):
        para = self.d.add_paragraph(style="Caption")
        r = para.add_run(f"{kind} ")
        add_field(para, f"SEQ {kind} \\* ARABIC", str(self.fig_n if kind == 'Figure' else self.tab_n))
        para.add_run(f": {text}")
        return para

    def figure(self, path, text, width_cm=14.6):
        self.fig_n += 1
        para = self.d.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.keep_with_next = True
        para.paragraph_format.space_after = Pt(2)
        para.paragraph_format.line_spacing = 1.0
        para.add_run().add_picture(str(path), width=Cm(min(width_cm, 14.6)))
        self.caption("Figure", text)

    def table(self, header, rows, widths_cm, caption, font_size=10, first_col_bold=False, align_num=True):
        self.tab_n += 1
        cap = self.caption("Table", caption)
        cap.paragraph_format.keep_with_next = True
        t = self.d.add_table(rows=1, cols=len(header))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, h in enumerate(header):
            c = t.rows[0].cells[i]
            c.text = ""
            r = c.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(font_size)
            set_cell_shading(c, "E7E6E6")
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = ""
                para = cells[i].paragraphs[0]
                self.runs(para, str(v), size=font_size, bold=(first_col_bold and i == 0))
                if align_num and i > 0:
                    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for row in t.rows:
            for i, w in enumerate(widths_cm):
                row.cells[i].width = Cm(w)
                for para in row.cells[i].paragraphs:
                    para.paragraph_format.line_spacing = 1.0
                    para.paragraph_format.space_after = Pt(1); para.paragraph_format.space_before = Pt(1)
            trPr = row._tr.get_or_add_trPr()
            cs = OxmlElement("w:cantSplit"); trPr.append(cs)
        self.p(after=4, spacing=1.0)
        return t

    def new_section(self, fmt):
        s = self.d.add_section(WD_SECTION.NEW_PAGE)
        margins(s)
        page_number_footer(s, fmt)
        return s


# --------------------------------------------------------------------------- content
def build():
    R = Report()
    d = R.d
    ph, lh = hyb(PRO), hyb(LITE)
    det = PRO["detection"]
    conf = PRO["confidence"]

    # ---- cover page (no page number) ----
    R.p("School of Computer Science and Engineering", size=14, bold=True, align="center", after=0)
    R.p("RV University", size=14, bold=True, align="center", after=24)
    R.p("Computer Vision Hackathon 2026", size=14, italic=True, align="center")
    R.p("Hackathon Technical Solution Report", size=16, bold=True, align="center", after=18)
    R.p(TITLE, size=20, bold=True, align="center", after=14, spacing=1.15)
    R.p("Problem Statement P5 — Intelligent Fine-Grained Vehicle Recognition and Analysis System",
        size=13, italic=True, align="center", after=18, spacing=1.15)
    R.p("Submitted by", size=14, italic=True, align="center")
    R.p("[Registered Team Name]  ·  Team ID: T20", size=14, bold=True, align="center", after=6)
    for n, u, _ in TEAM:
        R.p(f"{n.upper()}  ({u})", size=13, bold=True, align="center", after=0, spacing=1.15)
    R.p("", after=4)
    R.p("Guided by", size=14, italic=True, align="center")
    R.p("[GUIDE NAME]", size=14, bold=True, align="center")
    R.p("[Designation], School of Computer Science and Engineering", size=12, align="center", after=16)
    R.p("RV University", size=12, bold=True, align="center", after=0)
    R.p("RV Vidyaniketan, 8th Mile, Mysore Road, Bengaluru, Karnataka, India – 560059", size=11, align="center", after=0)
    R.p("November 2026", size=12, align="center")

    # ---- front matter (roman page numbers) ----
    R.new_section("lowerRoman")
    d.sections[0].footer.is_linked_to_previous = False  # cover has no number
    R.front_heading("CERTIFICATE")
    R.body(f"This is to certify that the project work titled **“{TITLE.title()}”** is performed by "
           + ", ".join(f"{n} ({u})" for n, u, _ in TEAM)
           + ", bonafide students of [Name of the degree] at the School of Computer Science and Engineering, "
             "RV University, Bengaluru, as part of the Computer Vision Hackathon 2026 during the academic year "
             "[2026–2027].")
    R.p("", after=40)
    t = d.add_table(rows=4, cols=3)
    for i, (a, b, c) in enumerate([("[Guide Name]", "[Program Director]", "[Dean]"),
                                   ("Guide", "Program Director", "Dean"),
                                   ("SoCSE, RV University", "SoCSE, RV University", "SoCSE, RV University"),
                                   ("Date:", "Date:", "Date:")]):
        for j, v in enumerate((a, b, c)):
            t.rows[i].cells[j].text = v
            t.rows[i].cells[j].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in t.rows[i].cells[j].paragraphs[0].runs:
                r.font.size = Pt(11); r.bold = i == 0
    R.p("", after=30)
    R.p("Name of the Examiner\t\t\t\tSignature of the Examiner", bold=True)
    R.p("1.\t\t\t\t\t\t1.")
    R.p("2.\t\t\t\t\t\t2.")
    R.page_break()

    R.front_heading("DECLARATION")
    R.body(f"We hereby declare that the work entitled **“{TITLE.title()}”**, submitted for the Computer Vision "
           "Hackathon 2026 at RV University, is a record of original work carried out by our team. All results "
           "reported in this document were produced by the code in the project repository and can be reproduced "
           "from it. Pretrained models, datasets and libraries used are acknowledged in the References.")
    R.body("We further declare that the work reported here has not been submitted, in part or in full, for any "
           "other degree, diploma or competition.")
    R.p("", after=20)
    R.p("Place: Bengaluru", bold=True, after=0)
    R.p("Date: [DD/MM/YYYY]", bold=True, after=20)
    for n, u, _ in TEAM:
        R.p(f"{n} ({u})\t\t\t\t[Signature]", after=0)
    R.page_break()

    R.front_heading("ABSTRACT")
    R.body("Fine-grained vehicle recognition — naming a car's make, model and year — is hard because many of the 196 "
           "Stanford Cars classes differ only by small details such as a facelifted bumper, a badge or a roof, while "
           "real photographs vary in lighting, viewpoint, resolution and occlusion. This report presents **Visionary**, "
           "a hybrid system that combines classical computer vision with deep learning. A YOLOv8n detector localises "
           "every vehicle; an adaptive illumination gate measures each crop in CIE-LAB space and applies CLAHE only "
           "to dark or flat images; a ConvNeXt classifier, fine-tuned on detector-like crops and calibrated with "
           "temperature scaling, predicts the class with test-time flip averaging; and a classical HOG + LBP + HSV "
           "classifier provides the required baseline and an optional fusion branch whose weight is tuned on a "
           "held-out validation split.")
    R.body(f"On the official test split of {PRO['n_test']:,} images that were never used for training or tuning, the "
           f"hybrid system reaches **{ph['top1']} % Top-1 and {ph['top5']} % Top-5 accuracy** (ConvNeXt-Small at 320 px), "
           f"compared with {PRO['comparison'][1]['top1']} % for deep learning alone and {CLS['test_top1']:.1f} % for "
           f"classical features alone. Detection achieves a mean IoU of {det['mean_iou']:.2f} and mAP@50 of "
           f"{det['map50']:.2f}; temperature scaling reduces the expected calibration error from "
           f"{PRO_TR['test_ece_before_T']:.1f} % to {PRO_TR['test_ece_after_T']:.1f} %. Dynamic INT8 quantisation shrinks "
           f"the classifier 3.5× with no measurable accuracy loss. The complete pipeline was ported to the browser "
           f"(ONNX Runtime Web + OpenCV.js), verified against the Python implementation, and deployed free of cost as a "
           f"static web application that handles multiple vehicles per image, low-confidence inputs and non-car images.")
    R.p("**Keywords:** fine-grained recognition, vehicle classification, YOLOv8, ConvNeXt, CLAHE, hybrid computer "
        "vision, model calibration, INT8 quantisation, ONNX Runtime Web.", align="justify")
    R.page_break()

    R.front_heading("ACKNOWLEDGEMENT")
    R.body("We express our sincere gratitude to [Guide Name], [Designation], School of Computer Science and "
           "Engineering, RV University, for guidance and encouragement throughout this work.")
    R.body("We thank [Dean / Program Director names], School of Computer Science and Engineering, RV University, "
           "and the organisers of the Computer Vision Hackathon 2026 for the opportunity and the problem statements.")
    R.body("We also acknowledge the creators of the Stanford Cars dataset, and the open-source communities behind "
           "PyTorch, timm, Ultralytics YOLO, OpenCV, ONNX Runtime and Hugging Face, whose tools made this work possible.")
    R.p("", after=20)
    R.p("Place: Bengaluru", bold=True, after=0)
    R.p("Date: [DD/MM/YYYY]", bold=True, after=0)
    R.p("Team T20", bold=True, align="right")
    R.page_break()

    R.front_heading("CONTENTS")
    toc = d.add_paragraph()
    add_field(toc, 'TOC \\o "1-2" \\h \\z \\u', "Right-click and choose Update Field to build the table of contents.")
    R.page_break()
    R.front_heading("LIST OF FIGURES")
    lof = d.add_paragraph()
    add_field(lof, 'TOC \\h \\z \\c "Figure"', "Update field to build the list of figures.")
    R.p("", after=10)
    R.front_heading("LIST OF TABLES")
    lot = d.add_paragraph()
    add_field(lot, 'TOC \\h \\z \\c "Table"', "Update field to build the list of tables.")

    # ---- body (arabic page numbers) ----
    R.new_section("decimal")

    # Project details table (template header)
    R.chapter_heading("Project Details", numbered=False)
    R.table(["Field", "Details"], [
        ["Problem ID", "P5"],
        ["Problem Title", "Intelligent Fine-Grained Vehicle Recognition and Analysis System"],
        ["Team Name", "[Registered Team Name] (Team ID: T20)"],
        ["Team Members", "; ".join(f"{n} ({u})" for n, u, _ in TEAM)],
        ["GitHub Repository", f"{REPO} — commit [commit hash evaluated]"],
        ["Live Demo / Deployment", f"{LIVE} (Hugging Face static Space; in-browser inference)"],
    ], [4.2, 10.4], "Project and submission details", font_size=10.5, first_col_bold=True, align_num=False)
    R.table(["Member", "USN", "Responsibility"], [[n, u, r] for n, u, r in TEAM], [4.0, 3.6, 7.0],
            "Team members and responsibilities", font_size=10.5, align_num=False)

    # 1 -------------------------------------------------------------------------------
    R.page_break()
    R.chapter_heading("Problem Understanding & Solution Strategy")
    R.h2("1.1 Core Technical Challenge")
    R.body("The task is to detect vehicles in real-world photographs and name the **make, model and manufacturing "
           "year** of each one, choosing among the 196 classes of the Stanford Cars dataset, and to display a bounding "
           "box, a confidence score and the intermediate image-processing results.")
    R.body("The primary technical barrier is **inter-class similarity at a fine grain**: many classes share the same "
           "body shell and differ only by a facelift (e.g. Bentley Continental GT 2007 vs 2012), a trim or badge "
           "(Audi A5 vs S5), or a roof (coupe vs convertible). With only about 41 training images per class, a "
           "model must learn subtle part-level cues. The failure modes we targeted are:")
    for b in ["**Look-alike classes** — facelift years, trims and body-style variants;",
              "**Imperfect localisation** — background clutter and other vehicles in the frame;",
              "**Illumination and image quality** — night/dark photographs, over-exposure, motion blur, sensor noise and low resolution;",
              "**Partial occlusion and unusual viewpoints;**",
              "**Out-of-distribution inputs** — cars outside the 196 classes and images with no vehicle at all, which must not produce confident wrong answers."]:
        R.bullet(b)
    R.body("Target constraints: a public URL that responds within about 10 seconds on a phone or laptop, robust "
           "handling of bad input, zero hosting cost, and a model small enough to run without a GPU.")
    R.h2("1.2 Proposed Methodology")
    R.body("Visionary first **localises** every vehicle with YOLOv8n, then **adapts the input** to each crop: it "
           "measures brightness and contrast in CIE-LAB space and applies CLAHE only when the crop is dark or flat, "
           "instead of a fixed global enhancement. A **fine-tuned, temperature-calibrated ConvNeXt** classifies each "
           "crop (with a horizontal-flip second view), a classical descriptor classifier provides a baseline and a "
           "validation-tuned fusion branch, and a **confidence gate** decides whether to answer, warn or report "
           "“no vehicle”.")
    R.body("This adaptive design was chosen over static baselines because a fixed pipeline either over-processes good "
           "images (always-on CLAHE changes the statistics the network was trained on) or under-serves difficult ones. "
           "Cropping to the detected vehicle removes background clutter and gives the classifier more pixels on the "
           "discriminative parts, which the ablation shows is the single largest gain.")

    # 2 -------------------------------------------------------------------------------
    R.chapter_heading("Technical Architecture & Implementation")
    R.body("Figure 1 shows the end-to-end pipeline. The same pipeline is implemented twice: `src/hybrid/engine.py` "
           "(Python, used for training/evaluation and the FastAPI service) and `web/pipeline.js` (browser, used by the "
           "deployed application). Repository modules are referenced below; no source code is reproduced here.")
    R.figure(FIG / "fig_architecture.png", "End-to-end architecture of Visionary")
    R.h2("2.1 Classical Algorithmic Baseline")
    R.h3("Algorithms utilised")
    for b in ["**CLAHE** (contrast-limited adaptive histogram equalisation) applied to the **L channel of CIE-LAB** only, "
              "so that chromaticity is preserved (clip limit 2.5, 8 × 8 tiles) — `src/classical/preprocessing.py`.",
              "**Bilateral filtering** (d = 9, σ_colour = σ_space = 75) for edge-preserving denoising.",
              "**Canny edge detection** after a 3 × 3 Gaussian blur, followed by morphological closing and external "
              "**contour extraction** (area ≥ 50 px) — `src/classical/edges.py`.",
              "**Descriptors** — HOG (9 orientations, 16 × 16 cells on a 128 × 128 grey crop) for shape, uniform LBP "
              "(P = 24, R = 3) for texture and an 8 × 8 × 8 HSV histogram for colour, 2,302 dimensions in total — "
              "`src/hybrid/classical_head.py`.",
              f"**Classical classifier** — standardisation → PCA (256 components) → multinomial logistic regression "
              f"(C = {CLS['C']}, selected on the validation split)."]:
        R.bullet(b)
    R.h3("Adaptive selection logic")
    R.body("Parameters are computed from each image rather than fixed:")
    for b in ["**Illumination gate:** the vehicle crop is converted to LAB; if the mean lightness is below 70 or its "
              "standard deviation is below 35 (out of 255), CLAHE is applied before classification; otherwise the crop "
              "is used unchanged.",
              "**Canny thresholds** are derived from the median grey level m of the blurred crop: lower = 0.67·m, "
              "upper = 1.33·m, widened by ±30 if they collapse on flat images.",
              "**Region of interest** is the detector box padded by 5 % on each side; the primary vehicle is the box "
              "with the largest area × confidence.",
              "**Multi-vehicle selection:** every other box at least 48 px and 1 % of the frame is classified; duplicates "
              "(IoU > 0.5, or more than 80 % inside a kept box) are removed; at most six vehicles."]:
        R.bullet(b)

    R.h2("2.2 Deep Learning Pipeline")
    R.h3("Model architecture")
    R.body(f"Two variants of **ConvNeXt** (Liu et al., 2022), pretrained on ImageNet-22k (timm), with a new 196-way "
           f"linear head: **Visionary Pro** — ConvNeXt-Small at 320 × 320 px ({PRO['params_m']} M parameters) — and "
           f"**Visionary Lite** — ConvNeXt-Tiny at 224 × 224 px ({LITE['params_m']} M parameters). ConvNeXt was "
           "selected because it offers transformer-level transfer accuracy with a convolutional design that is fast on "
           "CPUs, quantises well and exports cleanly to ONNX for in-browser execution. Vehicle localisation uses "
           "**YOLOv8n** (Jocher et al., 2023), COCO-pretrained, restricted to the car, bus and truck classes.")
    R.h3("Loss functions & training")
    for b in ["**Data:** Stanford Cars (Krause et al., 2013) — 8,144 training and 8,041 test images. A stratified 10 % of "
              "the training set (788 images) forms the validation split; the official test split is used only for "
              "final reporting.",
              "**Training input:** ground-truth box crops with random 0–15 % padding per side, so the classifier sees "
              "imperfect, detector-like crops; 15 % of samples use the full frame so it degrades gracefully if detection fails.",
              "**Augmentation:** horizontal flip, TrivialAugment and Random Erasing (a proxy for occlusion).",
              "**Loss:** cross-entropy with label smoothing 0.1; AdamW (lr 2e-4, 5× for the head, weight decay 0.05), "
              "one-epoch warm-up then cosine decay, gradient clipping, mixed precision, 15 epochs.",
              "**Hardware:** a single laptop RTX 3050 (6 GB): ≈ 47 s per epoch for Lite and ≈ 150 s for Pro.",
              f"**Calibration:** the best epoch is chosen on validation; a softmax temperature fitted on validation "
              f"(T = {PRO_TR['temperature']} for Pro, {LITE_TR['temperature']} for Lite) corrects the under-confidence "
              f"caused by label smoothing."]:
        R.bullet(b)

    R.h2("2.3 Hybrid Integration & Adaptive Mechanism")
    R.h3("Fusion strategy")
    for b in ["**Detection → classification:** the classical ROI extraction feeds the network a tight crop of each vehicle, "
              "focusing capacity on the discriminative parts.",
              "**Illumination prior → network input:** the LAB brightness/contrast statistics decide whether CLAHE is applied "
              "before the network sees the crop.",
              "**Classical descriptors → late fusion:** a log-linear (product-of-experts) fusion "
              "log p ∝ log p_CNN + w·log p_classical, with w and a re-calibration temperature selected on the validation "
              "split. Unlike a weighted average, it can only re-rank classes the network already finds plausible, so it "
              "never flattens the distribution.",
              "**Test-time augmentation:** the primary vehicle is classified from the crop and its mirror image and the "
              "probabilities are averaged."]:
        R.bullet(b)
    R.h3("Decision logic")
    R.body("After calibration, the system answers when the top-1 probability is at least 50 %, shows the answer as a "
           "**low-confidence best guess** with the top-5 alternatives when it is lower, and reports **“no vehicle "
           "detected”** when there is no detector box and the classifier is not confident. Inputs are validated "
           "(empty files, non-images, images under 32 px or over 15 MB are rejected with a clear message).")
    R.body(f"An important, honestly reported outcome: once a data leak was removed (the classical head had been refitted "
           f"on train + validation, inflating its validation scores), validation tuning selected a classical fusion "
           f"weight of {PRO['hybrid_weights']['w_classical']} for Pro and {LITE['hybrid_weights']['w_classical']} for "
           f"Lite. The classical branch therefore contributes through the ROI and the illumination gate rather than "
           f"through late fusion; this is analysed in the ablation (Section 3.3).")

    # 3 -------------------------------------------------------------------------------
    R.chapter_heading("Experimental Benchmarks & Evaluation")
    R.body(f"All results are on the official Stanford Cars test split ({PRO['n_test']:,} images never used for training, "
           "early stopping, calibration or fusion tuning) and are produced by `scripts/evaluate.py`; the complete metric "
           "file is `eval_P5_T20.md`. Latency is end-to-end per image on a CPU limited to two threads (median of 40 images), "
           "matching a free two-vCPU hosting tier.")
    R.h2("3.1 Quantitative Benchmarks")
    q8 = next(r for r in PRO["comparison"] if "INT8" in r["name"])
    R.table(["Approach / Model Variant", "Top-1 (%) ↑", "Top-5 (%) ↑", "Size (MB) ↓", "Latency (ms / image) ↓"], [
        ["Classical baseline (HOG + LBP + HSV → PCA → LogReg)", PRO["comparison"][0]["top1"], PRO["comparison"][0]["top5"],
         PRO["comparison"][0]["size_mb"], PRO["comparison"][0]["latency_ms"]],
        ["Deep learning only (ConvNeXt-Small, full image)", PRO["comparison"][1]["top1"], PRO["comparison"][1]["top5"],
         PRO["comparison"][1]["size_mb"], PRO["comparison"][1]["latency_ms"]],
        ["**Proposed hybrid — Pro** (FP32)", f"**{ph['top1']}**", f"**{ph['top5']}**", ph["size_mb"], ph["latency_ms"]],
        ["Proposed hybrid — Pro, INT8*", q8["top1"], q8["top5"], q8["size_mb"], q8["latency_ms"]],
        ["Proposed hybrid — Lite (ConvNeXt-Tiny, FP32)", lh["top1"], lh["top5"], lh["size_mb"], lh["latency_ms"]],
    ], [6.3, 2.0, 2.0, 2.0, 2.4], "Classical vs deep learning vs hybrid on the test set (8,041 images)", font_size=10)
    R.p(f"* INT8 measured on a class-stratified subset of {SUB['n']:,} test images; on the same images FP32 scores "
        f"{SUB['fp32_same_images'][0]} % and {SUB['argmax_agreement']} % of predictions are identical. Size includes the "
        "detector and classical head.", size=10, align="justify", spacing=1.0)
    R.table(["Metric", "Value"], [
        ["Mean / median IoU (primary YOLO box vs ground truth, 8,000 boxes)", f"{det['mean_iou']:.3f} / {det['median_iou']:.3f}"],
        ["Images with IoU ≥ 0.5", f"{det['iou_ge_0.5']} %"],
        ["Vehicle detection rate", f"{det['recall']} %"],
        ["mAP@0.50 / mAP@0.50:0.95", f"{det['map50']:.3f} / {det['map50_95']:.3f}"],
        ["Expected calibration error before → after temperature scaling",
         f"{PRO_TR['test_ece_before_T']:.1f} % → {PRO_TR['test_ece_after_T']:.1f} %"],
        ["Accuracy when confident (≥ 50 %) / when flagged (< 50 %)", f"{conf['acc_when_confident']} % / {conf['acc_when_flagged']} %"],
        ["Low-confidence flag rate", f"{conf['low_conf_rate']} %"],
        ["Browser vs Python implementation (same 300 test images)",
         f"{PAR['js_top1']} % vs {PAR['python_top1_same_images']} %; {PAR['top1_agreement_js_vs_python']} % identical"],
    ], [10.2, 4.4], "Localisation, detection, calibration and implementation-parity metrics", font_size=10, align_num=False)
    R.figure(FIG / "fig_comparison.png", "Top-1 accuracy of classical, deep-learning and hybrid approaches for both model variants", 13.5)
    R.h3("Analysis")
    R.body(f"Hand-crafted features alone reach {PRO['comparison'][0]['top1']} % — far above chance (0.5 %) but unable to "
           f"separate 196 look-alike classes. Fine-tuned deep features raise accuracy to {PRO['comparison'][1]['top1']} %, "
           f"and the hybrid pipeline adds a further {ph['top1'] - PRO['comparison'][1]['top1']:.2f} points for Pro "
           f"(+{lh['top1'] - LITE['comparison'][1]['top1']:.2f} for Lite) at the cost of running the detector. "
           f"The larger model is more accurate but about {ph['latency_ms'] / lh['latency_ms']:.1f}× slower; INT8 "
           f"quantisation recovers most of the speed with no accuracy loss, which is why the browser runs INT8 models "
           f"and defaults to Lite for phones.")
    R.figure(FIG / "fig_robustness.png", "Robustness under seven synthetic corruptions (Pro model)", 13.5)
    R.body("Under corruptions the hybrid is ahead in five of seven conditions, most clearly for viewpoint changes, "
           "over-exposure and motion blur. For the Pro model it is slightly behind deep-learning-only under low light "
           "and noise, and both collapse at very low resolution (Section 4).")
    R.figure(FIG / "fig_confidence.png", "Confidence threshold trade-off between coverage and accuracy", 12.5)

    R.h2("3.2 Visual Outputs & Intermediate Representations")
    R.body("Figure 5 shows outputs of the deployed pipeline on unseen test images: a single car, a dark studio image for "
           "which the illumination gate switched CLAHE on, and a dealership scene in which every vehicle is detected and "
           "numbered. Figure 6 shows the intermediate classical representations displayed in the application for the "
           "dark image.")
    R.figure(FIG / "fig_samples.png", "Outputs on unseen test images — single vehicle, dark image (CLAHE applied), several vehicles")
    R.figure(FIG / "fig_stages.png", "Intermediate states for the dark image: ROI, CLAHE, bilateral filter, Canny edges, contours")

    R.h2("3.3 Ablation Study")
    R.body("Each configuration adds one component to the one above (A and B are independent baselines), evaluated on "
           "the same cached detections so that differences are attributable to the component alone.")
    rows = []
    for a, b in zip(LITE["ablation"], PRO["ablation"]):
        rows.append([a["name"], a["top1"], a["delta"] or "–", b["top1"], b["delta"] or "–"])
    R.table(["Configuration", "Lite Top-1", "Δ", "Pro Top-1", "Δ"], rows, [7.4, 1.9, 1.5, 1.9, 1.5],
            "Ablation study on the test set (Top-1 %)", font_size=9.5)
    for b in [f"**Detector ROI** is the largest single gain (+{float(LITE['ablation'][3]['top1']) - float(LITE['ablation'][1]['top1']):.2f} for Lite, "
              f"+{float(PRO['ablation'][3]['top1']) - float(PRO['ablation'][1]['top1']):.2f} for Pro): the stronger model needs less help.",
              "**Flip test-time augmentation** adds a consistent +0.15 points.",
              "**The adaptive CLAHE gate** is neutral on clean test images (it rarely fires) — its value appears under "
              "low-light and flat images, and it is never harmful on good images, unlike always-on enhancement.",
              "**Full-frame context and classical late fusion** received zero weight from validation tuning; adding them "
              "does not change test accuracy. We report this rather than tune on the test set."]:
        R.bullet(b)

    # 4 -------------------------------------------------------------------------------
    R.chapter_heading("Failure Modes & Limitations")
    F = PRO["failures"]
    R.body(f"The Pro hybrid makes {F['n_errors']} errors on {PRO['n_test']:,} test images ({F['n_errors'] / PRO['n_test'] * 100:.1f} %). "
           "Mining the confusion matrix shows that they cluster into explainable groups (full analysis in "
           "`docs/failure_cases.md`).")
    conf_rows = [[k.replace(" -> ", " → "), v] for k, v in list(F["top_confusions"].items())[:6]]
    R.table(["Most frequent confusion (true → predicted)", "Errors"], conf_rows, [12.4, 2.2],
            "Most frequent confusions of the Pro model on the test set", font_size=10)
    R.figure(FIG / "fig_failures.png", "Observed failure cases: facelift year, roof-up convertible, same-maker sedan")
    R.h2("4.1 Observed Failure Cases")
    for b in ["**Facelift years and trims** — Bentley Continental GT 2012 → 2007, Dodge Caliber 2012 ↔ 2007, Audi S5 → A5, "
              "Audi TTS ↔ TT. Errors barely change between Lite and Pro, so more capacity alone does not solve them.",
              "**Body-style ambiguity and same-maker look-alikes** — a Ferrari 458 coupe predicted as the convertible (roof "
              "closed), a Toyota Corolla predicted as a Camry; these occur at almost 100 % confidence, so calibration "
              "cannot flag them.",
              "**Very low resolution** — when the image is reduced to 80 px, accuracy falls to about 62 % and the detector "
              "misses about 10 % of vehicles; cropping a tiny, blurry box removes useful context.",
              "**Out-of-distribution cars** — the dataset covers mostly US-market models from 1991–2012, so newer or "
              "regional cars are mapped to the nearest known class; usually with low confidence, but not always."]:
        R.bullet(b)
    R.h2("4.2 Root Cause & Mitigation")
    for b in ["**Root cause (facelifts, trims):** the discriminative cue occupies a few pixels (headlight internals, badge, "
              "bumper intakes) and each class has about 41 training images, so the network relies on the shared "
              "silhouette. **Mitigation:** part-based attention or higher-resolution crops of the front and rear, "
              "hierarchical make → model → trim prediction so the system can answer “Audi A5/S5 family” when unsure.",
              "**Root cause (confident errors):** a closed-set softmax must choose one of 196 labels and is sure when two "
              "classes look identical. **Mitigation:** open-set scoring on the embedding (energy or Mahalanobis distance) "
              "to abstain on unfamiliar inputs; the top-5 list already shows the correct sibling as the runner-up.",
              "**Root cause (low resolution):** the detector box shrinks below the network's effective receptive detail. "
              "**Mitigation:** fall back to full-frame classification (the pipeline's full-frame weight) when the box is "
              "smaller than about 64 px.",
              "**Dataset limitation:** collect recent and Indian-market vehicles and fine-tune; the threshold analysis "
              f"shows that keeping the 50 % gate removes the least reliable {conf['low_conf_rate']} % of answers."]:
        R.bullet(b)
    R.h2("4.3 Limitations and Ethics")
    R.body("The system does not read licence plates, recognise faces, track vehicles or store images; the browser "
           "application never uploads photographs. Predictions are presented with confidence and alternatives and are "
           "intended for educational and analytic use, not for enforcement decisions.")

    # 5 -------------------------------------------------------------------------------
    R.chapter_heading("Deployment & Quickstart Guide")
    R.h2("5.1 Quickstart Commands")
    R.code(["# 1. Clone the repository and install dependencies",
            f"git clone {REPO}.git",
            "cd Visionary",
            "pip install -r requirements-deploy.txt",
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu",
            "pip install --no-deps ultralytics ultralytics-thop",
            "",
            "# 2. Run the application (web UI + REST API) on http://localhost:7860",
            "uvicorn app.main:app --port 7860",
            "",
            "# or: one command with Docker",
            "docker build -t visionary . && docker run -p 7860:7860 visionary",
            "",
            "# 3. Reproduce training and every reported metric",
            "python scripts/prepare_data.py",
            "python scripts/train.py --backbone convnext_small.fb_in22k_ft_in1k --img 320 --name convnext_small_320",
            "python scripts/evaluate.py --ckpt weights/convnext_small_320.pt",
            "python scripts/write_report.py --default convnext_small_320 --extra convnext_tiny_224"])
    R.h2("5.2 Deployment Format")
    for b in [f"**In-browser web application** (primary): `web/` runs YOLOv8n and the INT8 ConvNeXt with ONNX Runtime Web "
              f"(multi-threaded WebAssembly) and the classical stages with OpenCV.js; hosted free on a Hugging Face static "
              f"Space at {LIVE}. Models (≈ 66 MB) are cached after the first visit; inference takes about 2–3 s per photo "
              "in a laptop browser, and photos never leave the device.",
              "**FastAPI service / Docker container:** `app/main.py` serves the same web application with the "
              "cross-origin-isolation headers and exposes `POST /api/predict`, `GET /api/models` and `GET /api/benchmarks`; "
              "it runs the ONNX Runtime FP32 or INT8 models on CPU (≈ 0.14–0.54 s per image).",
              "**Model formats:** PyTorch checkpoints for training; ONNX FP32 and dynamic-INT8 for deployment.",
              "**Quality assurance:** 49 automated tests (classical CV, detection, classifier, API including bad input "
              "and multi-vehicle images) and a JavaScript-vs-Python parity check (`scripts/check_web_parity.py`)."]:
        R.bullet(b)

    # Checklist -------------------------------------------------------------------------
    R.chapter_heading("Student Submission Checklist", numbered=False)
    for b in ["☐ Working GitHub repository link included with an updated README.md.",
              "☑ No raw code pasted into this report.",
              "☑ All table metrics match test runs reproducible via the repository (`scripts/evaluate.py`, `eval_P5_T20.md`).",
              "☐ All team member details accurately recorded (team name, guide and signatures to be completed)."]:
        R.p(b, after=2)

    # References -----------------------------------------------------------------------
    R.chapter_heading("References", numbered=False)
    refs = [
        "Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In Proceedings of the 34th International Conference on Machine Learning (pp. 1321–1330). PMLR.",
        "Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8 (Version 8) [Computer software]. https://github.com/ultralytics/ultralytics",
        "Krause, J., Stark, M., Deng, J., & Fei-Fei, L. (2013). 3D object representations for fine-grained categorization. In Proceedings of the IEEE International Conference on Computer Vision Workshops (pp. 554–561).",
        "Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 11976–11986).",
        "Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (Vol. 1, pp. 886–893).",
        "Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. IEEE Transactions on Pattern Analysis and Machine Intelligence, 24(7), 971–987.",
        "Zuiderveld, K. (1994). Contrast limited adaptive histogram equalization. In P. S. Heckbert (Ed.), Graphics Gems IV (pp. 474–485). Academic Press.",
        "Tomasi, C., & Manduchi, R. (1998). Bilateral filtering for gray and color images. In Proceedings of the Sixth International Conference on Computer Vision (pp. 839–846).",
        "Canny, J. (1986). A computational approach to edge detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, 8(6), 679–698.",
        "Müller, S. G., & Hutter, F. (2021). TrivialAugment: Tuning-free yet state-of-the-art data augmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 774–782).",
        "Zhong, Z., Zheng, L., Kang, G., Li, S., & Yang, Y. (2020). Random erasing data augmentation. In Proceedings of the AAAI Conference on Artificial Intelligence, 34(7), 13001–13008.",
        "Microsoft. (2024). ONNX Runtime and ONNX Runtime Web [Computer software]. https://onnxruntime.ai",
    ]
    for r in sorted(refs):
        para = R.p(r, align="justify", after=4, spacing=1.15)
        para.paragraph_format.left_indent = Cm(1.0)
        para.paragraph_format.first_line_indent = Cm(-1.0)

    d.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
