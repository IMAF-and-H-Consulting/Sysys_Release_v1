"""Citation guide and per-tab help dialogs for the Sysys GUI."""
from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.theme import COLORS

CITATION_GUIDE_TEXT = """SystS Citation Guide

Use this guide when reporting a systematic review or meta-analysis prepared with
SystS. Always cite the primary studies included in your review; the items below
cover the reporting guideline and the built-in synthesis methods.

Suggested methods text

Screening, extraction, and evidence synthesis were organized in SystS
(Hypothesis Consensus Analyzer; version 0.1.0). Meta-analytic effect sizes were
pooled using a random-effects model with the DerSimonian-Laird estimator.
Publication-bias diagnostics included funnel plots, Egger's regression test,
Duval and Tweedie's trim-and-fill method, and Rosenthal's fail-safe N when
available. Reporting was structured with PRISMA 2020 guidance.

Recommended references

1. Page MJ, McKenzie JE, Bossuyt PM, Boutron I, Hoffmann TC, Mulrow CD, et al.
   The PRISMA 2020 statement: an updated guideline for reporting systematic
   reviews. BMJ. 2021;372:n71. doi:10.1136/bmj.n71

2. DerSimonian R, Laird N. Meta-analysis in clinical trials. Controlled Clinical
   Trials. 1986;7(3):177-188. doi:10.1016/0197-2456(86)90046-2

3. Egger M, Davey Smith G, Schneider M, Minder C. Bias in meta-analysis detected
   by a simple, graphical test. BMJ. 1997;315(7109):629-634.
   doi:10.1136/bmj.315.7109.629

4. Duval S, Tweedie R. Trim and fill: a simple funnel-plot-based method of
   testing and adjusting for publication bias in meta-analysis. Biometrics.
   2000;56(2):455-463. doi:10.1111/j.0006-341X.2000.00455.x

5. Rosenthal R. The file drawer problem and tolerance for null results.
   Psychological Bulletin. 1979;86(3):638-641. doi:10.1037/0033-2909.86.3.638

6. SystS: Hypothesis Consensus Analyzer. Version 0.1.0. Computer software.
   Use your project's repository URL, release, or commit hash if available.

BibTeX

@article{page_prisma_2021,
  title = {The {PRISMA} 2020 statement: an updated guideline for reporting systematic reviews},
  author = {Page, Matthew J. and McKenzie, Joanne E. and Bossuyt, Patrick M. and Boutron, Isabelle and Hoffmann, Tammy C. and Mulrow, Cynthia D. and Shamseer, Larissa and Tetzlaff, Jennifer M. and Akl, Elie A. and Brennan, Sue E. and Chou, Roger and Glanville, Julie and Grimshaw, Jeremy M. and Hrobjartsson, Asbjorn and Lalu, Manoj M. and Li, Tianjing and Loder, Elizabeth W. and Mayo-Wilson, Evan and McDonald, Steve and McGuinness, Luke A. and Stewart, Lesley A. and Thomas, James and Tricco, Andrea C. and Welch, Vivian A. and Whiting, Penny and Moher, David},
  journal = {BMJ},
  volume = {372},
  pages = {n71},
  year = {2021},
  doi = {10.1136/bmj.n71}
}

@article{dersimonian_laird_1986,
  title = {Meta-analysis in clinical trials},
  author = {DerSimonian, Rebecca and Laird, Nan},
  journal = {Controlled Clinical Trials},
  volume = {7},
  number = {3},
  pages = {177--188},
  year = {1986},
  doi = {10.1016/0197-2456(86)90046-2}
}

@article{egger_bias_1997,
  title = {Bias in meta-analysis detected by a simple, graphical test},
  author = {Egger, Matthias and Davey Smith, George and Schneider, Martin and Minder, Christoph},
  journal = {BMJ},
  volume = {315},
  number = {7109},
  pages = {629--634},
  year = {1997},
  doi = {10.1136/bmj.315.7109.629}
}

@article{duval_tweedie_2000,
  title = {Trim and fill: a simple funnel-plot-based method of testing and adjusting for publication bias in meta-analysis},
  author = {Duval, Sue and Tweedie, Richard},
  journal = {Biometrics},
  volume = {56},
  number = {2},
  pages = {455--463},
  year = {2000},
  doi = {10.1111/j.0006-341X.2000.00455.x}
}

@article{rosenthal_file_drawer_1979,
  title = {The file drawer problem and tolerance for null results},
  author = {Rosenthal, Robert},
  journal = {Psychological Bulletin},
  volume = {86},
  number = {3},
  pages = {638--641},
  year = {1979},
  doi = {10.1037/0033-2909.86.3.638}
}

@software{systs_hypothesis_consensus_analyzer,
  title = {{SystS}: Hypothesis Consensus Analyzer},
  version = {0.1.0},
  note = {Computer software for systematic evidence synthesis. Replace this note with the repository URL, release, or commit hash used in the analysis.}
}
"""


class CitationGuideDialog(QDialog):
    """Copyable citation guidance for reporting SystS analyses."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Citation Guide")
        self.setMinimumSize(640, 560)
        self.resize(720, 640)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        intro = QLabel(
            "Use these references for SystS-assisted systematic reviews and "
            "meta-analyses. Include only the methods you actually report."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        self.citation_text = QTextEdit()
        self.citation_text.setReadOnly(True)
        self.citation_text.setAcceptRichText(False)
        self.citation_text.setPlainText(CITATION_GUIDE_TEXT)
        text_font = QFont("Menlo", 10)
        self.citation_text.setFont(text_font)
        self.citation_text.setAccessibleName("SystS citation guide text")
        outer.addWidget(self.citation_text, 1)

        bottom = QHBoxLayout()
        copy_btn = QPushButton("Copy Citation Text")
        copy_btn.setAccessibleName("Copy SystS citation guide text")
        copy_btn.clicked.connect(self._copy_citation_text)
        bottom.addWidget(copy_btn)
        bottom.addStretch()

        close_btn = QPushButton("Close")
        close_btn.setMinimumWidth(80)
        close_btn.clicked.connect(self.accept)
        bottom.addWidget(close_btn)
        outer.addLayout(bottom)

    def _copy_citation_text(self):
        QApplication.clipboard().setText(CITATION_GUIDE_TEXT)
        QMessageBox.information(
            self,
            "Citation Guide",
            "Citation guidance copied to the clipboard.",
        )


# ---------------------------------------------------------------------------
# HelpDialog — per-tab help with methodology context
# ---------------------------------------------------------------------------

class HelpDialog(QDialog):
    """Modal help dialog for a specific pipeline tab."""

    def __init__(self, tab_name: str, parent=None):
        super().__init__(parent)
        from gui.help_content import HELP_CONTENT

        self._tab_name = tab_name
        content = HELP_CONTENT.get(tab_name, {})

        self.setWindowTitle(f"Help \u2014 {content.get('title', tab_name)}")
        self.setMinimumSize(500, 550)
        self.resize(500, 550)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(6)

        # Scrollable content area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setSpacing(10)

        # Title
        icon = content.get("icon", "")
        title = content.get("title", tab_name)
        title_label = QLabel(f"{icon}  {title}")
        title_font = QFont("Tahoma", 12)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        # Separator
        sep = QWidget()
        sep.setFixedHeight(2)
        sep.setStyleSheet(
            "border-top: 1px solid #808080; border-bottom: 1px solid #ffffff;"
        )
        layout.addWidget(sep)

        # What This Stage Does
        layout.addWidget(self._section_header("What This Stage Does"))
        what_label = QLabel(content.get("what", ""))
        what_label.setWordWrap(True)
        layout.addWidget(what_label)

        # Why This Matters
        layout.addWidget(self._section_header("Why This Matters"))
        why_label = QLabel(content.get("why", ""))
        why_label.setWordWrap(True)
        layout.addWidget(why_label)

        # Understanding the Outputs
        layout.addWidget(self._section_header("Understanding the Outputs"))
        out_label = QLabel(content.get("outputs", ""))
        out_label.setWordWrap(True)
        layout.addWidget(out_label)

        # Further Reading
        reading = content.get("reading", [])
        if reading:
            layout.addWidget(self._section_header("Further Reading"))
            for ref in reading:
                link_html = (
                    f'<a href="{ref["url"]}" '
                    f'style="color: #0000FF; text-decoration: underline;">'
                    f'{ref["text"]}</a>'
                )
                link_label = QLabel(link_html)
                link_label.setOpenExternalLinks(True)
                link_label.setWordWrap(True)
                layout.addWidget(link_label)

        layout.addStretch()
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        # Bottom row: close button. (Help is only ever opened on request, so
        # there is no "don't show automatically" option; 18-L10.)
        bottom = QHBoxLayout()
        bottom.addStretch()

        close_btn = QPushButton("Close")
        close_btn.setMinimumWidth(80)
        close_btn.clicked.connect(self._on_close)
        bottom.addWidget(close_btn)
        outer.addLayout(bottom)

    @staticmethod
    def _section_header(text: str) -> QLabel:
        label = QLabel(text)
        font = QFont("Tahoma", 9)
        font.setBold(True)
        label.setFont(font)
        label.setStyleSheet(f"color: {COLORS['text']}; margin-top: 4px;")
        return label

    def _on_close(self):
        self.accept()
