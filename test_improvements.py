"""
Test script for improved PDF processing with formatting-based header detection
"""

from pathlib import Path
from PDF_Processor import PDFProcessor

def test_all_pdfs():
    """Test the processor on all sample PDFs"""

    # Initialize processor with debug mode
    processor = PDFProcessor(
        max_chunk_size=2000,
        chunk_overlap=200,
        debug=True
    )

    # Get PDF directory
    script_dir = Path(__file__).parent
    pdf_dir = script_dir / "pdfs"

    if not pdf_dir.exists():
        print(f"ERROR: PDF directory not found: {pdf_dir}")
        return

    # Find all PDFs
    pdf_files = list(pdf_dir.glob("*.pdf"))

    print("\n" + "="*80)
    print(f"TESTING IMPROVED PDF PROCESSOR")
    print(f"Found {len(pdf_files)} PDF files")
    print("="*80)

    results = []

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n\n{'#'*80}")
        print(f"TEST {i}/{len(pdf_files)}: {pdf_path.name}")
        print(f"{'#'*80}\n")

        # Process with formatting extraction
        result = processor.process_single_pdf(
            str(pdf_path),
            extraction_method='formatting',
            document_title=pdf_path.stem
        )

        results.append(result)

        # Print summary
        print(f"\n{'='*80}")
        print(f"RESULTS FOR: {pdf_path.name}")
        print(f"{'='*80}")
        print(f"Success: {result.success}")
        print(f"Extraction method: {result.metadata.get('extraction_method')}")
        print(f"Pages: {result.metadata.get('page_count')}")
        print(f"Extracted text length: {len(result.extracted_text)} chars")
        print(f"Cleaned text length: {len(result.cleaned_text)} chars")
        print(f"Chunks created: {len(result.chunks)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Warnings: {len(result.warnings)}")

        if result.errors:
            print(f"\nErrors:")
            for error in result.errors:
                print(f"  - {error}")

        if result.warnings:
            print(f"\nWarnings:")
            for warning in result.warnings:
                print(f"  - {warning}")

        # Show first 3 chunks
        if result.chunks:
            print(f"\nFirst 3 chunks:")
            for j, chunk in enumerate(result.chunks[:3], 1):
                print(f"\n  Chunk {j}:")
                print(f"    Size: {chunk['chunk_size']} chars")
                print(f"    Metadata: {chunk['metadata']}")
                print(f"    Text preview: {chunk['text'][:150]}...")

    # Overall summary
    print(f"\n\n{'='*80}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*80}")
    successful = sum(1 for r in results if r.success)
    total_chunks = sum(len(r.chunks) for r in results)

    print(f"Total PDFs processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    print(f"Total chunks created: {total_chunks}")

    if successful > 0:
        avg_chunks = total_chunks / successful
        print(f"Average chunks per document: {avg_chunks:.1f}")

        all_chunk_sizes = [chunk['chunk_size'] for r in results for chunk in r.chunks]
        if all_chunk_sizes:
            print(f"\nChunk size statistics:")
            print(f"  Min: {min(all_chunk_sizes)} chars")
            print(f"  Max: {max(all_chunk_sizes)} chars")
            print(f"  Average: {sum(all_chunk_sizes)/len(all_chunk_sizes):.0f} chars")

    print("="*80)


if __name__ == "__main__":
    test_all_pdfs()
