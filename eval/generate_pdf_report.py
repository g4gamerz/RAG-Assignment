"""
Generate PDF report from RAGAS evaluation results.
"""
import json
from datetime import datetime
from pathlib import Path

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
except ImportError:
    print("Error: reportlab not installed. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

def load_evaluation_results(file_path="eval/evaluation_results.json"):
    """Load evaluation results from JSON."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_pdf_report(results, output_file="eval/RAGAS_Evaluation_Report.pdf"):
    """Generate PDF report from evaluation results."""

    doc = SimpleDocTemplate(
        output_file,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18,
    )

    elements = []

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f2937'),
        spaceAfter=30,
        alignment=TA_CENTER
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2563eb'),
        spaceAfter=12,
        spaceBefore=12
    )

    normal_style = styles['Normal']

    title = Paragraph("RAG System Evaluation Report", title_style)
    elements.append(title)

    timestamp = datetime.fromisoformat(results['timestamp']).strftime('%B %d, %Y at %I:%M %p')
    date_text = Paragraph(f"<i>Generated on {timestamp}</i>", normal_style)
    elements.append(date_text)
    elements.append(Spacer(1, 0.3*inch))

    metrics = results['metrics']

    elements.append(Paragraph("Overall Performance", heading_style))

    performance_data = [
        ['Metric', 'Value'],
        ['Total Questions', str(metrics['total_questions'])],
        ['Successful Answers', str(metrics['successful_answers'])],
        ['Failed Answers', str(metrics['failed_answers'])],
        ['Success Rate', f"{metrics['success_rate']:.1%}"],
    ]

    performance_table = Table(performance_data, colWidths=[3*inch, 2*inch])
    performance_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563eb')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f3f4f6')]),
    ]))

    elements.append(performance_table)
    elements.append(Spacer(1, 0.3*inch))

    elements.append(Paragraph("Quality Metrics", heading_style))

    quality_data = [
        ['Metric', 'Value'],
        ['Average Confidence Score', f"{metrics['avg_confidence_score']:.3f} ({metrics['avg_confidence_score']:.1%})"],
        ['Average Retrieval Score', f"{metrics['avg_retrieval_score']:.3f}"],
    ]

    quality_table = Table(quality_data, colWidths=[3*inch, 2*inch])
    quality_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563eb')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f3f4f6')]),
    ]))

    elements.append(quality_table)
    elements.append(Spacer(1, 0.5*inch))

    elements.append(PageBreak())
    elements.append(Paragraph("Detailed Question Results", heading_style))
    elements.append(Spacer(1, 0.2*inch))

    detailed_data = [['#', 'Question', 'Confidence', 'Retrieval Score']]

    for i, result in enumerate(results['results'], 1):
        question_short = result['question'][:50] + '...' if len(result['question']) > 50 else result['question']
        detailed_data.append([
            str(i),
            question_short,
            f"{result['confidence_score']:.3f}",
            f"{result['avg_retrieval_score']:.3f}"
        ])

    detailed_table = Table(detailed_data, colWidths=[0.4*inch, 3.5*inch, 1.2*inch, 1.2*inch])
    detailed_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563eb')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f3f4f6')]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    elements.append(detailed_table)
    elements.append(Spacer(1, 0.3*inch))

    elements.append(PageBreak())
    elements.append(Paragraph("Sample Question-Answer Pairs", heading_style))
    elements.append(Spacer(1, 0.2*inch))

    top_results = sorted(results['results'], key=lambda x: x['confidence_score'], reverse=True)[:3]

    for i, result in enumerate(top_results, 1):
        q_text = f"<b>Q{i}: {result['question']}</b>"
        elements.append(Paragraph(q_text, normal_style))
        elements.append(Spacer(1, 0.1*inch))

        a_text = f"<i>Answer:</i> {result['answer'][:300]}{'...' if len(result['answer']) > 300 else ''}"
        elements.append(Paragraph(a_text, normal_style))

        conf_text = f"<i>Confidence: {result['confidence_score']:.3f} | Retrieved Docs: {result['retrieved_docs_count']} | Citations: {result['citations_count']}</i>"
        elements.append(Paragraph(conf_text, normal_style))
        elements.append(Spacer(1, 0.2*inch))

    elements.append(Spacer(1, 0.5*inch))
    footer_text = "<i>Report generated by RAG System Evaluation Tool</i>"
    elements.append(Paragraph(footer_text, normal_style))

    doc.build(elements)
    print(f"\n[OK] PDF report generated: {output_file}")

def main():
    """Main function."""
    print("Generating PDF Report from RAGAS Evaluation Results...")
    print("="*70)

    print("\n1. Loading evaluation results...")
    try:
        results = load_evaluation_results()
        print("   [OK] Results loaded")
    except Exception as e:
        print(f"   [X] Error loading results: {e}")
        return

    print("\n2. Generating PDF report...")
    try:
        create_pdf_report(results)
        print("   [OK] PDF generated successfully")
    except Exception as e:
        print(f"   [X] Error generating PDF: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "="*70)
    print("[OK] Report generation complete!")

if __name__ == "__main__":
    main()
