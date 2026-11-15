import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def create_callout(doc, text, bold_prefix="", border_color="1E40AF", bg_color="F0F4F8"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, bg_color)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    
    # Left border only
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    
    left = OxmlElement('w:left')
    left.set(qn('w:val'), 'single')
    left.set(qn('w:sz'), '24')
    left.set(qn('w:space'), '0')
    left.set(qn('w:color'), border_color)
    tcBorders.append(left)
    
    for side in ['top', 'bottom', 'right']:
        node = OxmlElement(f'w:{side}')
        node.set(qn('w:val'), 'none')
        tcBorders.append(node)
    tcPr.append(tcBorders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_b = p.add_run(bold_prefix + " ")
        r_b.bold = True
        r_b.font.color.rgb = RGBColor(15, 23, 42)
        r_b.font.size = Pt(10)
    r_t = p.add_run(text)
    r_t.font.size = Pt(10)
    r_t.font.color.rgb = RGBColor(30, 41, 59)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

def style_table(tbl, col_widths, headers, data, header_bg="1E3A8A"):
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    # Header row
    hdr_cells = tbl.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        set_cell_background(hdr_cells[i], header_bg)
        set_cell_margins(hdr_cells[i], top=120, bottom=120, left=150, right=150)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)
            run.font.size = Pt(9.5)
            run.font.name = "Arial"
            
    # Data rows
    for r_idx, row_data in enumerate(data):
        row_cells = tbl.add_row().cells
        bg = "FFFFFF" if r_idx % 2 == 0 else "F8FAFC"
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = str(val)
            set_cell_background(row_cells[c_idx], bg)
            set_cell_margins(row_cells[c_idx], top=100, bottom=100, left=150, right=150)
            p = row_cells[c_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 or c_idx == 1 else WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.font.size = Pt(9)
                run.font.name = "Arial"
                run.font.color.rgb = RGBColor(30, 41, 59)
                
    # Set widths
    for row in tbl.rows:
        for idx, width in enumerate(col_widths):
            row.cells[idx].width = width

def build_document():
    doc = Document()
    
    # Page setup - Standard 1 inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    # Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Arial'
    normal_style.font.size = Pt(10.5)
    normal_style.font.color.rgb = RGBColor(30, 41, 59)
    normal_style.paragraph_format.line_spacing = 1.2
    normal_style.paragraph_format.space_after = Pt(6)

    # -------------------------------------------------------------------------
    # DOCUMENT HEADER / TITLE BLOCK
    # -------------------------------------------------------------------------
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(4)
    run_t = p_title.add_run("ACADEMIC PROJECT REPORT & VIVA DEFENSE GUIDE")
    run_t.bold = True
    run_t.font.size = Pt(18)
    run_t.font.color.rgb = RGBColor(30, 58, 138) # Deep Oxford Blue

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(2)
    run_s = p_sub.add_run("Smart Library: Book Popularity Classification Using CART Decision Tree")
    run_s.bold = True
    run_s.font.size = Pt(13)
    run_s.font.color.rgb = RGBColor(71, 85, 105)

    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_meta.paragraph_format.space_after = Pt(18)
    run_m = p_meta.add_run("Course: B.Tech Computer Science & Engineering | Subject: Machine Learning (Unit 5)\nFocus Topic: Decision Trees (CART Classification Algorithm) | Student: Nikitha Deepak")
    run_m.font.size = Pt(9.5)
    run_m.font.italic = True
    run_m.font.color.rgb = RGBColor(100, 116, 139)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # -------------------------------------------------------------------------
    # 1. EXECUTIVE SUMMARY & PROBLEM STATEMENT
    # -------------------------------------------------------------------------
    h1 = doc.add_heading("1. Executive Summary & Problem Formulation", level=1)
    h1.runs[0].font.color.rgb = RGBColor(30, 58, 138)
    h1.paragraph_format.space_before = Pt(12)
    h1.paragraph_format.space_after = Pt(6)

    doc.add_paragraph(
        "Academic and public libraries face an ongoing resource allocation challenge: selecting which new book titles to acquire, "
        "predicting expected circulation frequency, and optimizing shelf placement. This project develops an intelligent "
        "machine learning classification system titled 'Smart Library: Book Popularity Classification Using CART Decision Tree'. "
        "The system classifies books into three discrete operational tiers: Popular, Moderately Popular, and Less Popular."
    )
    
    doc.add_paragraph(
        "A critical academic requirement of this project is strict adherence to Unit 5 of the B.Tech Machine Learning curriculum: "
        "Tree-based Learning and the CART (Classification and Regression Trees) algorithm. The project specifically avoids hybrid "
        "ensembles or black-box algorithms, providing complete white-box interpretability that can be clearly defended during viva voce."
    )

    create_callout(
        doc,
        "A major failure mode in naive machine learning projects is Target Leakage—using variables derived from the target (e.g., circulation counts, "
        "average rating scores, or review counts) as input features. When a new book arrives at a library, circulation statistics do not exist yet! "
        "In this project, rating counts are strictly isolated to define the historical ground-truth target label. The feature matrix contains exclusively "
        "intrinsic bibliographic metadata available at the exact moment of book procurement.",
        bold_prefix="Core Academic Integrity Principle (Zero Target Leakage):"
    )

    # -------------------------------------------------------------------------
    # 2. DATASET ARCHITECTURE & EMPIRICAL INSPECTION
    # -------------------------------------------------------------------------
    h2 = doc.add_heading("2. Dataset Architecture & Empirical Inspection", level=1)
    h2.runs[0].font.color.rgb = RGBColor(30, 58, 138)
    h2.paragraph_format.space_before = Pt(14)
    h2.paragraph_format.space_after = Pt(6)

    doc.add_paragraph(
        "The project is implemented using the renowned Book-Crossing benchmark dataset, collected by Cai-Nicolas Ziegler (University of Freiburg). "
        "The raw dataset consists of three relational CSV files structured as follows:"
    )

    tbl_ds = doc.add_table(rows=1, cols=4)
    style_table(
        tbl_ds,
        [Inches(1.5), Inches(1.2), Inches(1.8), Inches(2.0)],
        ["CSV File Name", "Record Count", "Attribute Columns", "Domain Significance"],
        [
            ["books.csv", "271,360", "ISBN, Book-Title, Book-Author, Year, Publisher, URLs", "Catalog bibliographic records"],
            ["users.csv", "278,858", "User-ID, Location, Age", "Patron demographic profiles"],
            ["ratings.csv", "1,149,780", "User-ID, ISBN, Book-Rating (0–10)", "Circulation and reader interactions"]
        ]
    )
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    doc.add_heading("Data Cleaning & Anomaly Resolution:", level=2).runs[0].font.color.rgb = RGBColor(51, 65, 85)
    doc.add_paragraph(
        "1. Delimiter & Encoding: All three files use semicolon delimiters (';') and require Latin-1 ('latin-1') character encoding. "
        "Standard UTF-8 parsing fails due to Western European author names and accented characters.\n"
        "2. Malformed Line Handling: A small number of records (e.g., historical records for DK Publishing and Gallimard) contained unescaped "
        "quotations that shifted columns. These were handled safely using pandas error handling ('on_bad_lines=\"skip\"') without data corruption.\n"
        "3. Year of Publication Imputation: 4,636 records possessed invalid publication years (values equal to 0 or greater than 2006). "
        "These were cleanly imputed using the catalog median publication year (1996).\n"
        "4. ISBN Deduplication: ISBN keys were uppercase-normalized and stripped of whitespace, yielding 271,045 unique catalog titles."
    )

    doc.add_heading("In-Depth Analysis of Zero Ratings (Crucial Viva Question):", level=2).runs[0].font.color.rgb = RGBColor(51, 65, 85)
    doc.add_paragraph(
        "A superficial inspection of ratings.csv shows that 716,109 ratings out of 1,149,780 (62.28%) are recorded as '0'. "
        "It is fundamentally incorrect to interpret a rating of 0 as a negative zero-star review. In the Book-Crossing benchmark:"
    )
    doc.add_paragraph(
        "• Rating 0 = Implicit Interaction: The user borrowed, shelved, or read the book but did not enter an explicit 1–10 star rating.\n"
        "• Ratings 1 to 10 = Explicit Reviews: Numeric evaluations expressing subjective reader appreciation.\n"
        "• Academic Decision: Because borrowing or shelving a book represents active reader engagement and library circulation demand, "
        "zero ratings are intentionally retained when calculating the total interaction volume per book."
    )

    # -------------------------------------------------------------------------
    # 3. TARGET VARIABLE FORMULATION
    # -------------------------------------------------------------------------
    h3 = doc.add_heading("3. Target Variable Formulation: 'Popularity'", level=1)
    h3.runs[0].font.color.rgb = RGBColor(30, 58, 138)
    h3.paragraph_format.space_before = Pt(14)
    h3.paragraph_format.space_after = Pt(6)

    doc.add_paragraph(
        "To formulate a clinically sound 3-class target variable, the total interaction count per unique book was computed across all 1.15 million transactions. "
        "The empirical interaction distribution exhibits a pronounced power-law (long-tail) distribution where the 50th percentile is 1 interaction, the 75th percentile is 3, "
        "and the 90th percentile is 7 interactions. Thresholds were established based on these distribution percentiles:"
    )

    tbl_target = doc.add_table(rows=1, cols=5)
    style_table(
        tbl_target,
        [Inches(1.5), Inches(1.3), Inches(1.2), Inches(1.1), Inches(1.4)],
        ["Popularity Tier", "Interaction Range", "Book Count", "Class Share", "Library Policy"],
        [
            ["Less Popular", "0 to 1 interaction", "146,571", "54.08%", "Procure 1 copy / digital"],
            ["Moderately Popular", "2 to 5 interactions", "91,717", "33.84%", "Procure 2–3 copies"],
            ["Popular", ">= 6 interactions", "32,754", "12.08%", "Procure 4–6 copies + front shelf"]
        ]
    )
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # -------------------------------------------------------------------------
    # 4. CART DECISION TREE MATHEMATICS (UNIT 5 THEORY)
    # -------------------------------------------------------------------------
    h4 = doc.add_heading("4. Machine Learning Algorithm: CART Decision Tree", level=1)
    h4.runs[0].font.color.rgb = RGBColor(30, 58, 138)
    h4.paragraph_format.space_before = Pt(14)
    h4.paragraph_format.space_after = Pt(6)

    doc.add_paragraph(
        "The Classification and Regression Trees (CART) algorithm, introduced by Leo Breiman et al. (1984), builds non-parametric binary decision trees. "
        "Unlike ID3 or C4.5 which create multi-way branches using Information Gain or Gain Ratio (Entropy), CART strictly produces binary trees "
        "by recursively partitioning feature space using the Gini Impurity index."
    )

    doc.add_heading("Mathematical Formulation: Gini Impurity Index", level=2).runs[0].font.color.rgb = RGBColor(51, 65, 85)
    doc.add_paragraph(
        "For a node D containing samples across k classes, the Gini Impurity is defined as:\n\n"
        "       Gini(D) = 1 - SUM [ (p_i)^2 ]  for i = 1 to k\n\n"
        "where p_i represents the relative proportion of class i instances in node D. When a node is completely pure (all samples belong to one class), "
        "Gini(D) = 0. In a balanced 3-class problem, maximum impurity reaches 1 - 3*(1/3)^2 = 0.667."
    )

    doc.add_paragraph(
        "When evaluating a potential binary split on feature A at threshold t dividing D into subsets D1 and D2, CART computes the weighted impurity:\n\n"
        "       Gini_split(D, A) = (|D1| / |D|) * Gini(D1) + (|D2| / |D|) * Gini(D2)\n\n"
        "The optimal split chosen at each node maximizes the impurity reduction: Delta_Gini(A) = Gini(D) - Gini_split(D, A)."
    )

    doc.add_heading("Hyperparameter Configuration & Regularization:", level=2).runs[0].font.color.rgb = RGBColor(51, 65, 85)
    doc.add_paragraph(
        "• criterion='gini': Enforces standard CART optimization.\n"
        "• max_depth=5: Pre-pruning mechanism. Restricting tree depth prevents overfitting, memorization of noise, and exponential leaf growth.\n"
        "• min_samples_leaf=50: Mandates that every decision leaf must contain at least 50 historical training books, ensuring robust statistical support.\n"
        "• random_state=42: Guarantees exact experimental reproducibility."
    )

    # -------------------------------------------------------------------------
    # 5. FEATURE ENGINEERING (NO TARGET LEAKAGE)
    # -------------------------------------------------------------------------
    h5 = doc.add_heading("5. Feature Engineering (Bibliographic Metadata Only)", level=1)
    h5.runs[0].font.color.rgb = RGBColor(30, 58, 138)
    h5.paragraph_format.space_before = Pt(14)
    h5.paragraph_format.space_after = Pt(6)

    doc.add_paragraph(
        "To classify books without target leakage, seven non-leaking features were engineered directly from bibliographic metadata:"
    )

    tbl_feat = doc.add_table(rows=1, cols=4)
    style_table(
        tbl_feat,
        [Inches(1.8), Inches(1.0), Inches(1.3), Inches(2.4)],
        ["Feature Name", "Data Type", "Gini Importance", "Analytical Rationale"],
        [
            ["Author_Book_Count", "Integer", "58.38%", "Catalog prolificacy. Prolific authors have loyal built-in followings."],
            ["Publisher_Book_Count", "Integer", "30.29%", "Publisher scale. Major conglomerates have national distribution."],
            ["Book_Age", "Integer", "7.14%", "Years elapsed since 2004. Classic books accumulate steady circulations."],
            ["Title_Length", "Integer", "2.57%", "Character length. Concise commercial fiction vs long academic titles."],
            ["Year-Of-Publication", "Integer", "1.62%", "Historical release era signal."],
            ["Title_Word_Count", "Integer", "0.00%", "Subsumed by Title_Length in greedy tree splitting."],
            ["Has_Subtitle", "Binary", "0.00%", "Binary indicator (1 if ':' or '-' present, else 0)."]
        ]
    )
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # -------------------------------------------------------------------------
    # 6. MODEL EVALUATION & EXPERIMENTAL RESULTS
    # -------------------------------------------------------------------------
    h6 = doc.add_heading("6. Model Evaluation & Experimental Results", level=1)
    h6.runs[0].font.color.rgb = RGBColor(30, 58, 138)
    h6.paragraph_format.space_before = Pt(14)
    h6.paragraph_format.space_after = Pt(6)

    doc.add_paragraph(
        "The model was evaluated using an 80-20 Stratified Train-Test Split (216,833 training samples and 54,209 unseen testing samples). "
        "Stratification ensures that the 54:34:12 class distribution is strictly preserved across both partitions."
    )

    tbl_perf = doc.add_table(rows=1, cols=5)
    style_table(
        tbl_perf,
        [Inches(1.8), Inches(1.2), Inches(1.2), Inches(1.2), Inches(1.1)],
        ["Popularity Class", "Precision", "Recall", "F1-Score", "Test Support"],
        [
            ["Less Popular", "0.6170 (61.7%)", "0.8459 (84.6%)", "0.7136", "29,314"],
            ["Moderately Popular", "0.4236 (42.4%)", "0.2797 (28.0%)", "0.3369", "18,344"],
            ["Popular", "0.4390 (43.9%)", "0.1281 (12.8%)", "0.1983", "6,551"],
            ["Overall Accuracy", "56.75%", "Macro F1", "0.4163", "54,209"]
        ]
    )
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    doc.add_heading("Confusion Matrix on 54,209 Unseen Testing Books:", level=2).runs[0].font.color.rgb = RGBColor(51, 65, 85)
    doc.add_paragraph(
        "• Actual Less Popular (29,314): 24,796 correctly predicted | 4,108 predicted Moderate | 410 predicted Popular\n"
        "• Actual Moderately Popular (18,344): 12,551 predicted Less | 5,131 correctly predicted | 662 predicted Popular\n"
        "• Actual Popular (6,551): 2,838 predicted Less | 2,874 predicted Moderate | 839 correctly predicted"
    )

    create_callout(
        doc,
        "Why is overall accuracy 56.75% rather than 95%? In a 3-class problem, random guessing yields 33.3%, and the majority class baseline is 54.0%. "
        "Predicting book popularity strictly from basic bibliographic metadata (without full manuscript text, reader sentiment reviews, or marketing ad spend) "
        "is an inherently noisy task. A model reporting 95% on this dataset almost certainly commits target leakage (such as leaking rating counts into the input). "
        "Achieving 56.75% without any data leakage is an authentic, defensible academic result.",
        bold_prefix="Honest Academic Assessment:"
    )

    # -------------------------------------------------------------------------
    # 7. COMPREHENSIVE VIVA VOCE DEFENSE & PROFESSOR Q&A
    # -------------------------------------------------------------------------
    doc.add_page_break()
    h7 = doc.add_heading("7. Comprehensive Viva Defense & Professor Q&A", level=1)
    h7.runs[0].font.color.rgb = RGBColor(30, 58, 138)
    h7.paragraph_format.space_before = Pt(14)
    h7.paragraph_format.space_after = Pt(6)

    doc.add_paragraph(
        "This section compiles 18 anticipated technical and conceptual questions organized across four critical domains. "
        "Study these detailed answers thoroughly before your lab examination or viva defense."
    )

    viva_questions = [
        # Domain 1: CART Theory
        ("Q1: What machine learning algorithm did you implement, and which unit does it belong to?",
         "Answer: I implemented the CART (Classification and Regression Trees) Decision Tree Classifier via scikit-learn's DecisionTreeClassifier. "
         "It belongs directly to Unit 5: Tree-based Learning. CART constructs binary decision trees by recursively partitioning the feature space "
         "using the Gini Impurity criterion."),

        ("Q2: How does CART differ from ID3 and C4.5 decision tree algorithms?",
         "Answer: ID3 and C4.5 use Information Gain and Gain Ratio based on Shannon Entropy (logarithmic), and ID3 creates multi-way splits for categorical variables. "
         "In contrast, CART strictly creates binary splits at every internal node and optimizes the Gini Impurity index, which is computationally faster "
         "because it avoids logarithmic evaluations."),

        ("Q3: What is the mathematical definition of Gini Impurity? What do minimum and maximum values mean?",
         "Answer: Gini Impurity is defined as Gini(D) = 1 - sum(p_i^2) for i=1 to k, where p_i is the proportion of class i samples in node D. "
         "A minimum value of 0.0 indicates a perfectly pure node (all instances belong to a single class). For a balanced 3-class problem, "
         "maximum impurity is 1 - 3*(1/3)^2 = 0.667, indicating maximum uncertainty."),

        ("Q4: Why did you set max_depth=5 and min_samples_leaf=50?",
         "Answer: Without constraints, a decision tree will grow until every leaf is pure (Gini = 0), memorizing outliers and noise (severe overfitting). "
         "Setting max_depth=5 acts as pre-pruning regularization, restricting the maximum decision path to 5 rules. Setting min_samples_leaf=50 guarantees "
         "that every classification rule is supported by at least 50 historical book samples, ensuring generalization."),

        # Domain 2: Target Leakage & Features
        ("Q5: What is Target Leakage, and how did you rigorously prevent it in this project?",
         "Answer: Target Leakage occurs when features used during training contain information about the target variable that would not be available "
         "in a live deployment scenario. In our project, book popularity is defined by the number of ratings received. If we had included rating counts, "
         "average rating score, or circulation logs as input features, that would be direct target leakage. We prevented this by using rating counts "
         "STRICTLY to establish historical ground-truth labels. The input feature matrix contains only bibliographic metadata (Author catalog count, "
         "Publisher count, Book age, Title length) available at the exact moment of book procurement."),

        ("Q6: How does the system predict popularity for an author or publisher never seen before in the training catalog?",
         "Answer: For novel authors or independent publishers not present in the catalog dictionary, the system defaults their frequency count to 1 "
         "(Author_Book_Count = 1, Publisher_Book_Count = 1). The decision tree then evaluates the remaining features such as Book_Age and Title_Length, "
         "typically classifying novel titles into the 'Less Popular' or 'Moderately Popular' tier until historical circulation data accumulates."),

        ("Q7: Which feature had the highest Gini importance, and what is the real-world intuition behind it?",
         "Answer: Author_Book_Count accounted for 58.38% of total Gini importance, followed by Publisher_Book_Count at 30.29%. "
         "The real-world intuition is that readership demand in libraries is predominantly author-driven (prolific authors like Stephen King or J.K. Rowling "
         "have large, established reader followings) and publisher-driven (major publishing houses have extensive marketing and distribution channels)."),

        # Domain 3: Dataset & Preprocessing
        ("Q8: Why are 62.28% of the ratings in ratings.csv equal to 0? How did you handle them?",
         "Answer: In the Book-Crossing benchmark dataset, rating 0 does NOT indicate a negative zero-star review. The dataset documentation explains "
         "that 0 denotes an IMPLICIT interaction (the user read, borrowed, or shelved the book without leaving an explicit numerical rating between 1 and 10). "
         "Because borrowing or shelving represents active reader interest and places demand on library inventory, zero ratings were retained when computing "
         "overall circulation popularity."),

        ("Q9: What data cleaning steps were necessary before model training?",
         "Answer: 1) Semicolon delimiter parsing with Latin-1 encoding to handle Western European author names; 2) Safely skipping corrupted lines where "
         "quotation marks shifted columns into the year field; 3) Imputing invalid publication years (< 1800 or > 2006) with the catalog median (1996); "
         "4) Deduplicating ISBN keys and filling null author/publisher values with 'Unknown'."),

        ("Q10: Why did you not include user demographics like Age and Location in the book popularity classifier?",
         "Answer: 1) Missingness: User Age was missing in 39.7% of patron records; 2) Deployment feasibility: When evaluating an unacquired book, "
         "no patrons have borrowed it yet, so patron demographic aggregations do not exist. Forcing user features would cause severe cold-start failure."),

        # Domain 4: Evaluation & Viva Defense
        ("Q11: Why did you use a Stratified Train-Test Split instead of standard random splitting?",
         "Answer: Because the target classes are imbalanced (54.08% Less Popular, 33.84% Moderately Popular, 12.08% Popular). "
         "A standard random split risks under-sampling the minority 'Popular' class in either the training or testing partition. "
         "Stratification enforces the exact 54:34:12 proportion in both the 216,833 training set and the 54,209 testing set."),

        ("Q12: Why is the model accuracy 56.75%? Can this be defended to the evaluation panel?",
         "Answer: Absolutely. In a 3-class classification problem, random guessing achieves 33.3%, and the majority-class baseline is 54.0%. "
         "Achieving 56.75% across 54,209 unseen test books using STRICTLY bibliographic metadata without target leakage is a genuine, reproducible result. "
         "Predicting whether a book will gain massive readership without reading the book's text or knowing marketing advertising budgets is an inherently "
         "challenging problem. Projects reporting 90%+ on this dataset almost always suffer from target leakage."),

        ("Q13: What does the Confusion Matrix tell us about the model's strengths and weaknesses?",
         "Answer: The model demonstrates exceptional recall on 'Less Popular' books (84.6%), accurately filtering out low-demand titles. "
         "Its primary challenge is distinguishing between 'Moderately Popular' and 'Popular' books, where boundary overlap occurs because both "
         "tiers often share similar major publishers. This is an honest reflection of metadata-only classification."),

        ("Q14: What is the difference between Macro F1-Score and Weighted F1-Score?",
         "Answer: Macro F1-Score (41.63%) calculates the unweighted mean of F1-scores across all three classes, treating minority 'Popular' books "
         "with equal importance as majority 'Less Popular' books. Weighted F1-Score (52.38%) weights each class's F1-score by its support in the test set, "
         "reflecting real-world class distribution performance."),

        ("Q15: If the professor asks you to improve this model in future work, what would you suggest?",
         "Answer: 1) Incorporating NLP feature extraction (TF-IDF or BERT embeddings) on book titles and subject synopsis keywords; "
         "2) Including author historical award or genre metadata; 3) Testing cost-sensitive learning or class weighting to boost recall on the minority 'Popular' class."),

        ("Q16: How does the Streamlit web application integrate with the trained model?",
         "Answer: The training pipeline serializes the trained CART model, feature column specifications, and frequency dictionaries into a joblib bundle "
         "('decision_tree_model.pkl'). The Streamlit application loads this bundle via st.cache_resource, derives the exact 7 non-leaking features from user input "
         "in real time, and executes dt_model.predict() and predict_proba() to output classification badges and librarian acquisition plans."),

        ("Q17: What operational decisions does this system provide to a library collection manager?",
         "Answer: It provides automated procurement action plans: 1) For 'Popular' titles: procure 4–6 physical copies, secure digital licenses, "
         "and assign front-of-library featured display; 2) For 'Moderately Popular' titles: procure 2–3 copies for standard subject stack shelves; "
         "3) For 'Less Popular' titles: procure a single reference copy or rely on inter-library on-demand lending, preserving acquisition budget."),

        ("Q18: What is the time complexity of training and predicting with a CART Decision Tree?",
         "Answer: For a training dataset with n samples and m features, CART training complexity is O(m * n * log(n) * d), where d is tree depth. "
         "Inference / prediction complexity is O(d), which is O(5) in our model. This guarantees instantaneous real-time prediction in under 1 millisecond per book.")
    ]

    for q, ans in viva_questions:
        p_q = doc.add_paragraph()
        p_q.paragraph_format.space_before = Pt(8)
        p_q.paragraph_format.space_after = Pt(2)
        r_q = p_q.add_run(q)
        r_q.bold = True
        r_q.font.color.rgb = RGBColor(30, 58, 138)
        r_q.font.size = Pt(10.5)

        p_a = doc.add_paragraph()
        p_a.paragraph_format.space_before = Pt(0)
        p_a.paragraph_format.space_after = Pt(6)
        r_a = p_a.add_run(ans)
        r_a.font.size = Pt(10)
        r_a.font.color.rgb = RGBColor(51, 65, 85)

    doc.add_paragraph().paragraph_format.space_after = Pt(10)
    p_end = doc.add_paragraph()
    p_end.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_end = p_end.add_run("--- End of Academic Report & Viva Defense Document ---")
    r_end.font.italic = True
    r_end.font.size = Pt(9.5)
    r_end.font.color.rgb = RGBColor(148, 163, 184)

    # Output path
    out_path = r"C:\Users\NIKITHA\.gemini\antigravity\scratch\smart_library\Smart_Library_ML_Model_and_Viva_Guide.docx"
    doc.save(out_path)
    print(f"Document successfully saved to: {out_path}")

if __name__ == "__main__":
    build_document()
