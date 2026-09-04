import os
import re
import html
import json

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)
from reportlab.lib.units import mm

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether
)


# ============================================================
# 1. FONT SETUP
# ============================================================

def register_fonts():
    """
    Register a Unicode-compatible font.

    This is useful because lesson content may contain:
    CO₂
    m²
    °C
    ×
    →
    Δ
    etc.
    """

    possible_regular_fonts = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]

    possible_bold_fonts = [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    ]


    regular_font = None
    bold_font = None


    for path in possible_regular_fonts:

        if os.path.exists(path):

            regular_font = path

            break


    for path in possible_bold_fonts:

        if os.path.exists(path):

            bold_font = path

            break


    if regular_font and bold_font:

        pdfmetrics.registerFont(
            TTFont(
                "LessonRegular",
                regular_font
            )
        )

        pdfmetrics.registerFont(
            TTFont(
                "LessonBold",
                bold_font
            )
        )

        return (
            "LessonRegular",
            "LessonBold"
        )


    return (
        "Helvetica",
        "Helvetica-Bold"
    )


REGULAR_FONT, BOLD_FONT = register_fonts()


# ============================================================
# 2. PDF STYLES
# ============================================================

def create_styles():

    styles = getSampleStyleSheet()


    title_style = ParagraphStyle(

        name="LessonTitle",

        parent=styles["Title"],

        fontName=BOLD_FONT,

        fontSize=24,

        leading=29,

        alignment=TA_CENTER,

        spaceAfter=8,

        textColor=colors.HexColor(
            "#17365D"
        )
    )


    subtitle_style = ParagraphStyle(

        name="LessonSubtitle",

        parent=styles["Heading2"],

        fontName=BOLD_FONT,

        fontSize=15,

        leading=19,

        alignment=TA_CENTER,

        spaceAfter=16,

        textColor=colors.HexColor(
            "#365F91"
        )
    )


    section_style = ParagraphStyle(

        name="SectionHeading",

        parent=styles["Heading2"],

        fontName=BOLD_FONT,

        fontSize=15,

        leading=19,

        spaceBefore=10,

        spaceAfter=8,

        textColor=colors.HexColor(
            "#365F91"
        )
    )


    subsection_style = ParagraphStyle(

        name="SubsectionHeading",

        parent=styles["Heading3"],

        fontName=BOLD_FONT,

        fontSize=11.5,

        leading=15,

        spaceBefore=6,

        spaceAfter=4,

        textColor=colors.HexColor(
            "#244062"
        )
    )


    body_style = ParagraphStyle(

        name="BodyTextCustom",

        parent=styles["BodyText"],

        fontName=REGULAR_FONT,

        fontSize=9.5,

        leading=13,

        spaceAfter=5
    )


    bullet_style = ParagraphStyle(

        name="BulletCustom",

        parent=body_style,

        leftIndent=12,

        firstLineIndent=-7,

        bulletIndent=4,

        spaceAfter=4
    )


    small_style = ParagraphStyle(

        name="SmallText",

        parent=body_style,

        fontSize=8.5,

        leading=11
    )


    answer_style = ParagraphStyle(

        name="AnswerText",

        parent=body_style,

        leftIndent=8,

        rightIndent=8,

        borderPadding=6,

        backColor=colors.HexColor(
            "#EAF2F8"
        ),

        textColor=colors.HexColor(
            "#17365D"
        )
    )


    return {
        "title": title_style,
        "subtitle": subtitle_style,
        "section": section_style,
        "subsection": subsection_style,
        "body": body_style,
        "bullet": bullet_style,
        "small": small_style,
        "answer": answer_style
    }


STYLES = create_styles()


# ============================================================
# 3. TEXT HELPERS
# ============================================================


def convert_unicode_super_subscripts(text):
    """
    Convert Unicode superscript/subscript characters into
    ReportLab <super> and <sub> tags.

    This prevents missing-glyph squares such as:
        kg□¹

    and lets ReportLab render:
        kg^-1
    with -1 properly superscripted.
    """

    superscript_map = {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "⁺": "+",
        "⁻": "-",
        "⁼": "=",
        "⁽": "(",
        "⁾": ")"
    }

    subscript_map = {
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
        "₊": "+",
        "₋": "-",
        "₌": "=",
        "₍": "(",
        "₎": ")"
    }

    result = []
    i = 0

    while i < len(text):

        character = text[i]

        if character in superscript_map:

            converted = []

            while (
                i < len(text)
                and text[i] in superscript_map
            ):
                converted.append(
                    superscript_map[text[i]]
                )
                i += 1

            result.append(
                "<super>"
                + "".join(converted)
                + "</super>"
            )
            continue

        if character in subscript_map:

            converted = []

            while (
                i < len(text)
                and text[i] in subscript_map
            ):
                converted.append(
                    subscript_map[text[i]]
                )
                i += 1

            result.append(
                "<sub>"
                + "".join(converted)
                + "</sub>"
            )
            continue

        result.append(character)
        i += 1

    return "".join(result)


def safe_text(value):
    """
    Prepare AI-generated text for use inside a ReportLab
    Paragraph.

    Handles:

    - Markdown bold:
        **Kinetic Energy**

    - Markdown italics:
        *mass*

    - Common LaTeX wrappers:
        \\( ... \\)
        \\[ ... \\]

    - Common fractions:
        \\frac{1}{2}

    - Subscripts:
        E_k
        E_{k}

    - Superscripts:
        m^2
        s^{-1}

    - Common mathematical commands:
        \\Delta
        \\times
        \\cdot

    - Problematic Unicode hyphens / spaces
    """

    if value is None:

        return ""


    text = str(
        value
    )

    # Convert Unicode superscripts/subscripts before
    # HTML escaping so ReportLab can render them.
    text = convert_unicode_super_subscripts(
        text
    )


    # ========================================================
    # 1. NORMALISE PROBLEMATIC UNICODE
    # ========================================================

    replacements = {

        "\u2010": "-",     # hyphen
        "\u2011": "-",     # non-breaking hyphen
        "\u2012": "-",     # figure dash
        "\u2013": "-",     # en dash
        "\u2014": "-",     # em dash

        "\u2212": "-",     # mathematical minus

        "\u00a0": " ",     # non-breaking space

        "\u200b": "",      # zero-width space
        "\ufeff": "",      # BOM / zero-width no-break space
    }


    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )


    # ========================================================
    # 2. REMOVE LATEX DISPLAY WRAPPERS
    # ========================================================

    text = text.replace(
        "\\(",
        ""
    )

    text = text.replace(
        "\\)",
        ""
    )

    text = text.replace(
        "\\[",
        ""
    )

    text = text.replace(
        "\\]",
        ""
    )


    # ========================================================
    # 3. CONVERT COMMON LATEX FRACTIONS
    #
    # \frac{1}{2}
    # becomes:
    # 1/2
    # ========================================================

    text = re.sub(
        r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
        r"\1/\2",
        text
    )


    # A second pass helps with another simple nested case.

    text = re.sub(
        r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
        r"\1/\2",
        text
    )


    # ========================================================
    # 4. COMMON LATEX COMMANDS
    # ========================================================

    text = text.replace(
        "\\Delta",
        "Δ"
    )

    text = text.replace(
        "\\delta",
        "δ"
    )

    text = text.replace(
        "\\times",
        "×"
    )

    text = text.replace(
        "\\cdot",
        "×"
    )

    text = text.replace(
        "\\approx",
        "≈"
    )

    text = text.replace(
        "\\propto",
        "∝"
    )

    text = text.replace(
        "\\leq",
        "≤"
    )

    text = text.replace(
        "\\geq",
        "≥"
    )

    text = text.replace(
        "\\neq",
        "≠"
    )


    # Remove common LaTeX spacing commands.

    text = text.replace(
        "\\,",
        " "
    )

    text = text.replace(
        "\\;",
        " "
    )

    text = text.replace(
        "\\:",
        " "
    )

    text = text.replace(
        "\\!",
        ""
    )


    # ========================================================
    # 5. SIMPLE LATEX TEXT COMMANDS
    #
    # \text{hello} -> hello
    # ========================================================

    text = re.sub(
        r"\\text\s*\{([^{}]+)\}",
        r"\1",
        text
    )


    text = re.sub(
        r"\\mathrm\s*\{([^{}]+)\}",
        r"\1",
        text
    )


    # ========================================================
    # 6. CONVERT SUBSCRIPTS
    #
    # E_k       -> E<sub>k</sub>
    # E_{elastic} -> E<sub>elastic</sub>
    # ========================================================

    text = re.sub(
        r"_\{([^{}]+)\}",
        r"<sub>\1</sub>",
        text
    )


    text = re.sub(
        r"_([A-Za-z0-9])",
        r"<sub>\1</sub>",
        text
    )


    # ========================================================
    # 7. CONVERT SUPERSCRIPTS
    #
    # m^2       -> m<super>2</super>
    # s^{-1}    -> s<super>-1</super>
    # ========================================================

    text = re.sub(
        r"\^\{([^{}]+)\}",
        r"<super>\1</super>",
        text
    )


    text = re.sub(
        r"\^([A-Za-z0-9+\-]+)",
        r"<super>\1</super>",
        text
    )


    # ========================================================
    # 8. PROTECT REPORTLAB TAGS BEFORE HTML ESCAPING
    # ========================================================

    protected_tags = {

        "<sub>":
            "__SUB_OPEN__",

        "</sub>":
            "__SUB_CLOSE__",

        "<super>":
            "__SUPER_OPEN__",

        "</super>":
            "__SUPER_CLOSE__",
    }


    for tag, placeholder in protected_tags.items():

        text = text.replace(
            tag,
            placeholder
        )


    # ========================================================
    # 9. ESCAPE USER / AI TEXT
    #
    # Prevent random < > & characters from breaking
    # ReportLab's Paragraph parser.
    # ========================================================

    text = html.escape(
        text
    )


    # ========================================================
    # 10. RESTORE SAFE REPORTLAB TAGS
    # ========================================================

    for tag, placeholder in protected_tags.items():

        text = text.replace(
            placeholder,
            tag
        )


    # ========================================================
    # 11. CONVERT MARKDOWN BOLD
    #
    # **Kinetic Energy**
    # ->
    # <b>Kinetic Energy</b>
    # ========================================================

    text = re.sub(
        r"\*\*(.+?)\*\*",
        r"<b>\1</b>",
        text
    )


    # ========================================================
    # 12. CONVERT SIMPLE MARKDOWN ITALICS
    #
    # *mass*
    # ->
    # <i>mass</i>
    # ========================================================

    text = re.sub(
        r"(?<!\*)\*([^*\n]+?)\*(?!\*)",
        r"<i>\1</i>",
        text
    )


    # ========================================================
    # 13. CLEAN STRAY LATEX BACKSLASHES
    #
    # Only remove backslashes immediately before characters
    # that are commonly left over after simple LaTeX.
    # ========================================================

    text = re.sub(
        r"\\(?=[A-Za-z])",
        "",
        text
    )


    return text


# ============================================================
# BULLET HELPER
#
# already_formatted=True is important when we manually insert
# ReportLab formatting such as <b>LO1</b>.
# ============================================================

def bullet(
    text,
    already_formatted=False
):

    if already_formatted:

        final_text = text

    else:

        final_text = safe_text(
            text
        )


    return Paragraph(
        "• " + final_text,
        STYLES["bullet"]
    )


def section_heading(
    number,
    title
):

    return Paragraph(
        f"{number}. {safe_text(title)}",
        STYLES["section"]
    )


# ============================================================
# 4. HEADER / FOOTER
# ============================================================

def add_page_number(
    canvas,
    document
):

    canvas.saveState()


    width, height = A4


    canvas.setFont(
        REGULAR_FONT,
        8
    )


    canvas.setFillColor(
        colors.HexColor(
            "#666666"
        )
    )


    canvas.drawString(
        18 * mm,
        10 * mm,
        "AI-Assisted Lesson Planner"
    )


    page_number = canvas.getPageNumber()


    canvas.drawRightString(
        width - 18 * mm,
        10 * mm,
        f"Page {page_number}"
    )


    canvas.restoreState()


# ============================================================
# 5. METADATA
# ============================================================

def add_metadata(
    story,
    metadata
):

    topic = metadata.get(
        "topic",
        "Lesson"
    )

    level = metadata.get(
        "level",
        ""
    )

    lesson_length = metadata.get(
        "lessonLengthMinutes",
        ""
    )

    scope_note = metadata.get(
        "scopeNote",
        ""
    )


    data = [

        [
            Paragraph(
                "<b>Topic</b>",
                STYLES["body"]
            ),

            Paragraph(
                safe_text(
                    topic
                ),
                STYLES["body"]
            )
        ],

        [
            Paragraph(
                "<b>Level</b>",
                STYLES["body"]
            ),

            Paragraph(
                safe_text(
                    level
                ),
                STYLES["body"]
            )
        ],

        [
            Paragraph(
                "<b>Lesson Length</b>",
                STYLES["body"]
            ),

            Paragraph(
                (
                    f"{safe_text(lesson_length)} "
                    f"minutes"
                ),
                STYLES["body"]
            )
        ]
    ]


    table = Table(
        data,
        colWidths=[
            45 * mm,
            120 * mm
        ]
    )


    table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor(
                    "#E7E6E6"
                )
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),

            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor(
                    "#C9C9C9"
                )
            ),

            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.25,
                colors.HexColor(
                    "#DDDDDD"
                )
            ),

            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                7
            ),

            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                7
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                6
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                6
            )
        ])
    )


    story.append(
        table
    )


    if scope_note:

        story.append(
            Spacer(
                1,
                7
            )
        )


        scope_table = Table(
            [[
                Paragraph(
                    (
                        "<b>Scope Note</b><br/>"
                        + safe_text(
                            scope_note
                        )
                    ),
                    STYLES["body"]
                )
            ]],
            colWidths=[
                165 * mm
            ]
        )


        scope_table.setStyle(
            TableStyle([

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor(
                        "#F2F2F2"
                    )
                ),

                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor(
                        "#D0D0D0"
                    )
                ),

                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                ),

                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                )
            ])
        )


        story.append(
            scope_table
        )


# ============================================================
# 6. LEARNING OBJECTIVES
# ============================================================

def add_learning_objectives(
    story,
    objectives
):

    for objective in objectives:

        objective_id = safe_text(
            objective.get(
                "id",
                ""
            )
        )


        objective_text = safe_text(
            objective.get(
                "objective",
                ""
            )
        )


        # IMPORTANT:
        #
        # We intentionally create the <b> tag AFTER the
        # underlying values have passed through safe_text().
        #
        # The bullet helper is then told not to escape the
        # formatting again.

        text = (
            f"<b>{objective_id}</b>: "
            f"{objective_text}"
        )


        story.append(
            bullet(
                text,
                already_formatted=True
            )
        )


# ============================================================
# 7. PREREQUISITE KNOWLEDGE
# ============================================================

def add_prerequisites(
    story,
    prerequisites
):

    for item in prerequisites:

        story.append(
            bullet(
                item
            )
        )


# ============================================================
# 8. KEY TERMS TABLE
# ============================================================

def add_key_terms(
    story,
    key_terms
):

    rows = [

        [
            Paragraph(
                "<b>Term</b>",
                STYLES["body"]
            ),

            Paragraph(
                "<b>Definition</b>",
                STYLES["body"]
            )
        ]
    ]


    for item in key_terms:

        rows.append([

            Paragraph(
                safe_text(
                    item.get(
                        "term",
                        ""
                    )
                ),
                STYLES["body"]
            ),

            Paragraph(
                safe_text(
                    item.get(
                        "definition",
                        ""
                    )
                ),
                STYLES["body"]
            )
        ])


    table = Table(
        rows,

        colWidths=[
            45 * mm,
            120 * mm
        ],

        repeatRows=1
    )


    table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor(
                    "#D9EAD3"
                )
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.HexColor(
                    "#777777"
                )
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),

            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                5
            ),

            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                5
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                4
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                4
            )
        ])
    )


    story.append(
        table
    )


# ============================================================
# 9. KEY POINTS
# ============================================================

def add_key_points(
    story,
    key_points
):

    for item in key_points:

        if isinstance(
            item,
            dict
        ):

            text = item.get(
                "point",
                ""
            )

        else:

            text = item


        story.append(
            bullet(
                text
            )
        )


# ============================================================
# 10. REAL-WORLD APPLICATIONS
# ============================================================

def add_real_world_applications(
    story,
    applications,
    teacher_version
):

    if not applications:

        story.append(
            Paragraph(
                (
                    "No specific real-world applications "
                    "were generated."
                ),
                STYLES["body"]
            )
        )

        return


    for application in applications:

        block = []


        title = application.get(
            "title",
            "Application"
        )


        block.append(
            Paragraph(
                safe_text(
                    title
                ),
                STYLES["subsection"]
            )
        )


        explanation = application.get(
            "explanation",
            ""
        )


        if explanation:

            block.append(
                Paragraph(
                    safe_text(
                        explanation
                    ),
                    STYLES["body"]
                )
            )


        question = application.get(
            "question",
            ""
        )


        if question:

            block.append(
                Paragraph(
                    (
                        "<b>Think about:</b> "
                        + safe_text(
                            question
                        )
                    ),
                    STYLES["body"]
                )
            )


        if teacher_version:

            answer = application.get(
                "expectedAnswer",
                ""
            )


            if answer:

                block.append(
                    Paragraph(
                        (
                            "<b>Expected answer:</b> "
                            + safe_text(
                                answer
                            )
                        ),
                        STYLES["answer"]
                    )
                )


        estimated = application.get(
            "estimatedMinutes"
        )


        if estimated:

            block.append(
                Paragraph(
                    (
                        "<i>Suggested time: "
                        f"{safe_text(estimated)} "
                        "minutes</i>"
                    ),
                    STYLES["small"]
                )
            )


        story.append(
            KeepTogether(
                block
            )
        )


        story.append(
            Spacer(
                1,
                6
            )
        )


# ============================================================
# 11. TEACHER SCHEDULE
# ============================================================

def add_teacher_schedule(
    story,
    schedule
):

    if not schedule:

        story.append(
            Paragraph(
                "No teacher schedule was generated.",
                STYLES["body"]
            )
        )

        return


    rows = [

        [
            Paragraph(
                "<b>Time</b>",
                STYLES["small"]
            ),

            Paragraph(
                "<b>Stage</b>",
                STYLES["small"]
            ),

            Paragraph(
                "<b>Teacher Direction</b>",
                STYLES["small"]
            ),

            Paragraph(
                "<b>Student Action</b>",
                STYLES["small"]
            )
        ]
    ]


    for stage in schedule:

        start = stage.get(
            "startMinute",
            ""
        )

        end = stage.get(
            "endMinute",
            ""
        )


        rows.append([

            Paragraph(
                f"{start}-{end} min",
                STYLES["small"]
            ),

            Paragraph(
                safe_text(
                    stage.get(
                        "stage",
                        ""
                    )
                ),
                STYLES["small"]
            ),

            Paragraph(
                safe_text(
                    stage.get(
                        "teacherDirection",
                        ""
                    )
                ),
                STYLES["small"]
            ),

            Paragraph(
                safe_text(
                    stage.get(
                        "studentAction",
                        ""
                    )
                ),
                STYLES["small"]
            )
        ])


    table = Table(

        rows,

        colWidths=[
            22 * mm,
            33 * mm,
            62 * mm,
            48 * mm
        ],

        repeatRows=1
    )


    table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor(
                    "#D9EAF7"
                )
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.35,
                colors.HexColor(
                    "#777777"
                )
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),

            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                4
            ),

            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                4
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                4
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                4
            )
        ])
    )


    story.append(
        table
    )


# ============================================================
# 12. CORE NOTES
# ============================================================

def add_core_notes(
    story,
    core_notes
):

    if not core_notes:

        story.append(
            Paragraph(
                "No core notes were generated.",
                STYLES["body"]
            )
        )

        return


    for note in core_notes:

        heading = note.get(
            "heading",
            ""
        )

        content = note.get(
            "content",
            ""
        )


        story.append(
            Paragraph(
                safe_text(
                    heading
                ),
                STYLES["subsection"]
            )
        )


        # ====================================================
        # PROCESS CONTENT
        #
        # This now converts:
        #
        # **bold**
        # *italics*
        # E_k
        # m^2
        # \frac{1}{2}
        #
        # before displaying it.
        # ====================================================

        content = safe_text(
            content
        )


        # Preserve AI-created line breaks.

        content = content.replace(
            "\n",
            "<br/>"
        )


        story.append(
            Paragraph(
                content,
                STYLES["body"]
            )
        )


        story.append(
            Spacer(
                1,
                5
            )
        )


# ============================================================
# 13. ACTIVITIES
# ============================================================

def add_activities(
    story,
    activities,
    activity_scheduled=None,
    selected_activity=None,
    teacher_version=False
):

    if not activities:

        story.append(
            Paragraph(
                "No activities were generated.",
                STYLES["body"]
            )
        )

        return


    if teacher_version:

        if activity_scheduled:

            story.append(
                Paragraph(
                    (
                        "<b>Scheduled activity:</b> "
                        + safe_text(
                            selected_activity.get(
                                "title",
                                ""
                            )
                            if selected_activity
                            else ""
                        )
                    ),
                    STYLES["body"]
                )
            )

        else:

            story.append(
                Paragraph(
                    (
                        "<b>Suggested Activities "
                        "If Time Available</b>"
                    ),
                    STYLES["body"]
                )
            )


            story.append(
                Spacer(
                    1,
                    4
                )
            )


    for activity in activities:

        block = []


        title = activity.get(
            "title",
            ""
        )

        minutes = activity.get(
            "estimatedMinutes",
            ""
        )


        block.append(
            Paragraph(
                (
                    f"{safe_text(title)} "
                    f"({safe_text(minutes)} min)"
                ),
                STYLES["subsection"]
            )
        )


        description = activity.get(
            "description",
            ""
        )


        if description:

            block.append(
                Paragraph(
                    safe_text(
                        description
                    ),
                    STYLES["body"]
                )
            )


        story.append(
            KeepTogether(
                block
            )
        )


        story.append(
            Spacer(
                1,
                5
            )
        )


# ============================================================
# 14. QUESTIONS
# ============================================================

def add_questions(
    story,
    questions,
    show_answers
):

    if not questions:

        story.append(
            Paragraph(
                "No questions were generated.",
                STYLES["body"]
            )
        )

        return


    for question in questions:

        question_id = question.get(
            "id",
            ""
        )

        question_text = question.get(
            "question",
            ""
        )

        block = []


        # We add <b> ourselves here,
        # but question_id and question_text are individually
        # passed through safe_text first.

        question_id_safe = safe_text(
            question_id
        )

        question_text_safe = safe_text(
            question_text
        )


        block.append(
            Paragraph(
                (
                    f"<b>{question_id_safe}.</b> "
                    f"{question_text_safe}"
                ),
                STYLES["body"]
            )
        )


        if show_answers:

            expected_answer = question.get(
                "expectedAnswer",
                ""
            )


            if expected_answer:

                block.append(
                    Paragraph(
                        (
                            "<b>Answer:</b> "
                            + safe_text(
                                expected_answer
                            )
                        ),
                        STYLES["answer"]
                    )
                )


        story.append(
            KeepTogether(
                block
            )
        )


        story.append(
            Spacer(
                1,
                8
            )
        )


# ============================================================
# 15. CREATE TEACHER PDF
# ============================================================

def create_teacher_pdf(
    lesson_data,
    output_path
):

    metadata = lesson_data.get(
        "lessonMetadata",
        {}
    )

    shared = lesson_data.get(
        "shared",
        {}
    )

    teacher_only = lesson_data.get(
        "teacherOnly",
        {}
    )


    topic = metadata.get(
        "topic",
        "Lesson"
    )


    document = SimpleDocTemplate(

        output_path,

        pagesize=A4,

        rightMargin=18 * mm,

        leftMargin=18 * mm,

        topMargin=18 * mm,

        bottomMargin=18 * mm,

        title=(
            f"{topic} - Teacher Lesson Guide"
        ),

        author=(
            "AI-Assisted Lesson Planner"
        )
    )


    story = []


    # ========================================================
    # TITLE
    # ========================================================

    story.append(
        Paragraph(
            safe_text(
                topic.upper()
            ),
            STYLES["title"]
        )
    )


    story.append(
        Paragraph(
            "Teacher Lesson Guide",
            STYLES["subtitle"]
        )
    )


    # ========================================================
    # 1. METADATA
    # ========================================================

    story.append(
        section_heading(
            1,
            "Metadata"
        )
    )


    add_metadata(
        story,
        metadata
    )


    # ========================================================
    # 2. LEARNING OBJECTIVES
    # ========================================================

    story.append(
        section_heading(
            2,
            "Learning Objectives"
        )
    )


    add_learning_objectives(
        story,
        shared.get(
            "learningObjectives",
            []
        )
    )


    # ========================================================
    # 3. PREREQUISITE KNOWLEDGE
    # ========================================================

    story.append(
        section_heading(
            3,
            "Prerequisite Knowledge"
        )
    )


    add_prerequisites(
        story,
        shared.get(
            "prerequisiteKnowledge",
            []
        )
    )


    # ========================================================
    # 4. KEY TERMS
    # ========================================================

    story.append(
        section_heading(
            4,
            "Key Terms"
        )
    )


    add_key_terms(
        story,
        shared.get(
            "keyTerms",
            []
        )
    )


    # ========================================================
    # 5. KEY POINTS
    # ========================================================

    story.append(
        section_heading(
            5,
            "Key Points"
        )
    )


    add_key_points(
        story,
        shared.get(
            "keyPoints",
            []
        )
    )


    # ========================================================
    # 6. REAL-WORLD APPLICATIONS
    # ========================================================

    story.append(
        section_heading(
            6,
            "Real-World Applications"
        )
    )


    add_real_world_applications(
        story,
        shared.get(
            "realWorldApplications",
            []
        ),
        teacher_version=True
    )


    # ========================================================
    # 7. TEACHER SCHEDULE
    # ========================================================

    story.append(
        section_heading(
            7,
            "Teacher Schedule"
        )
    )


    add_teacher_schedule(
        story,
        teacher_only.get(
            "lessonSchedule",
            []
        )
    )


    # ========================================================
    # 8. ACTIVITIES
    # ========================================================

    story.append(
        section_heading(
            8,
            "Activities"
        )
    )


    activities = shared.get(
        "activities",
        []
    )


    add_activities(

        story,

        activities,

        activity_scheduled=
            teacher_only.get(
                "activityScheduled",
                False
            ),

        selected_activity=
            teacher_only.get(
                "selectedActivity"
            ),

        teacher_version=True
    )


    # ========================================================
    # 9. QUESTIONS + ANSWERS
    # ========================================================

    story.append(
        section_heading(
            9,
            "Questions and Answers"
        )
    )


    add_questions(
        story,
        shared.get(
            "questions",
            []
        ),
        show_answers=True
    )


    document.build(

        story,

        onFirstPage=
            add_page_number,

        onLaterPages=
            add_page_number
    )


    print(
        f"Teacher PDF created: "
        f"{output_path}"
    )


# ============================================================
# 16. CREATE STUDENT PDF
# ============================================================

def create_student_pdf(
    lesson_data,
    output_path
):

    metadata = lesson_data.get(
        "lessonMetadata",
        {}
    )

    shared = lesson_data.get(
        "shared",
        {}
    )

    student_only = lesson_data.get(
        "studentOnly",
        {}
    )


    topic = metadata.get(
        "topic",
        "Lesson"
    )


    document = SimpleDocTemplate(

        output_path,

        pagesize=A4,

        rightMargin=18 * mm,

        leftMargin=18 * mm,

        topMargin=18 * mm,

        bottomMargin=18 * mm,

        title=(
            f"{topic} - Student Learning Pack"
        ),

        author=(
            "AI-Assisted Lesson Planner"
        )
    )


    story = []


    # ========================================================
    # TITLE
    # ========================================================

    story.append(
        Paragraph(
            safe_text(
                topic.upper()
            ),
            STYLES["title"]
        )
    )


    story.append(
        Paragraph(
            "Student Learning Pack",
            STYLES["subtitle"]
        )
    )


    # ========================================================
    # 1. METADATA
    # ========================================================

    story.append(
        section_heading(
            1,
            "Metadata"
        )
    )


    add_metadata(
        story,
        metadata
    )


    # ========================================================
    # 2. LEARNING OBJECTIVES
    # ========================================================

    story.append(
        section_heading(
            2,
            "Learning Objectives"
        )
    )


    add_learning_objectives(
        story,
        shared.get(
            "learningObjectives",
            []
        )
    )


    # ========================================================
    # 3. PREREQUISITE KNOWLEDGE
    # ========================================================

    story.append(
        section_heading(
            3,
            "Prerequisite Knowledge"
        )
    )


    add_prerequisites(
        story,
        shared.get(
            "prerequisiteKnowledge",
            []
        )
    )


    # ========================================================
    # 4. KEY TERMS
    # ========================================================

    story.append(
        section_heading(
            4,
            "Key Terms"
        )
    )


    add_key_terms(
        story,
        shared.get(
            "keyTerms",
            []
        )
    )


    # ========================================================
    # 5. KEY POINTS
    # ========================================================

    story.append(
        section_heading(
            5,
            "Key Points"
        )
    )


    add_key_points(
        story,
        shared.get(
            "keyPoints",
            []
        )
    )


    # ========================================================
    # 6. REAL-WORLD APPLICATIONS
    # ========================================================

    story.append(
        section_heading(
            6,
            "Real-World Applications"
        )
    )


    add_real_world_applications(
        story,
        shared.get(
            "realWorldApplications",
            []
        ),
        teacher_version=False
    )


    # ========================================================
    # 7. CORE NOTES
    # ========================================================

    story.append(
        section_heading(
            7,
            "Core Notes"
        )
    )


    add_core_notes(
        story,
        student_only.get(
            "coreNotes",
            []
        )
    )


    # ========================================================
    # 8. ACTIVITIES
    # ========================================================

    story.append(
        section_heading(
            8,
            "Activities"
        )
    )


    add_activities(
        story,
        shared.get(
            "activities",
            []
        ),
        teacher_version=False
    )


    # ========================================================
    # 9. QUESTIONS
    # NO ANSWERS
    # ========================================================

    story.append(
        section_heading(
            9,
            "Questions"
        )
    )


    add_questions(
        story,
        shared.get(
            "questions",
            []
        ),
        show_answers=False
    )


    document.build(

        story,

        onFirstPage=
            add_page_number,

        onLaterPages=
            add_page_number
    )


    print(
        f"Student PDF created: "
        f"{output_path}"
    )


# ============================================================
# 17. CREATE BOTH PDFs FROM PYTHON DICTIONARY
# ============================================================

def create_lesson_pdfs(
    lesson_data,
    output_directory="generated_lesson_pdfs"
):

    os.makedirs(
        output_directory,
        exist_ok=True
    )


    metadata = lesson_data.get(
        "lessonMetadata",
        {}
    )


    topic = metadata.get(
        "topic",
        "Lesson"
    )


    # ========================================================
    # SAFE WINDOWS FILENAME
    # ========================================================

    safe_topic = "".join(

        character

        for character in topic

        if character.isalnum()

        or character in (
            " ",
            "-",
            "_"
        )

    ).strip()


    if not safe_topic:

        safe_topic = "Lesson"


    teacher_path = os.path.join(
        output_directory,
        f"{safe_topic}_Teacher_Lesson_Guide.pdf"
    )


    student_path = os.path.join(
        output_directory,
        f"{safe_topic}_Student_Learning_Pack.pdf"
    )


    create_teacher_pdf(
        lesson_data,
        teacher_path
    )


    create_student_pdf(
        lesson_data,
        student_path
    )


    return {

        "teacherPdf":
            teacher_path,

        "studentPdf":
            student_path
    }


# ============================================================
# 18. LOAD JSON FILE AND CREATE PDFs
# ============================================================

def create_lesson_pdfs_from_json(
    json_path,
    output_directory="generated_lesson_pdfs"
):

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as file:

        lesson_data = json.load(
            file
        )


    return create_lesson_pdfs(
        lesson_data,
        output_directory
    )


# ============================================================
# 19. TEST / STANDALONE USE
# ============================================================

if __name__ == "__main__":

    JSON_PATH = (
        "generated_lesson_plan.json"
    )


    generated_files = (
        create_lesson_pdfs_from_json(
            JSON_PATH
        )
    )


    print(
        "\n========================================"
    )

    print(
        "LESSON PDFs GENERATED"
    )

    print(
        "========================================"
    )


    print(
        f"Teacher: "
        f"{generated_files['teacherPdf']}"
    )


    print(
        f"Student: "
        f"{generated_files['studentPdf']}"
    )
