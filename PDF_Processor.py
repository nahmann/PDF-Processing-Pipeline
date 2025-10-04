"""
PDF Text Extraction and Chunking Pipeline for RAG Systems

This module provides a complete pipeline for extracting text from PDFs,
cleaning and standardizing the text (especially for legal documents),
and chunking it for use in RAG systems.

Supports both AWS Textract and PyMuPDF for extraction.
"""

import os
import re
import boto3
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from botocore.exceptions import ClientError, NoCredentialsError
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.schema import Document


@dataclass
class ProcessingResult:
    """Container for processing results with metadata"""
    success: bool
    document_path: str
    document_name: str
    extracted_text: str = ""
    cleaned_text: str = ""
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, error: str):
        """Add an error message"""
        self.errors.append(error)
        self.success = False
    
    def add_warning(self, warning: str):
        """Add a warning message"""
        self.warnings.append(warning)


class PDFProcessor:
    """
    Complete pipeline for PDF text extraction, cleaning, and chunking.
    Supports both AWS Textract and PyMuPDF extraction methods.
    """
    
    def __init__(self, 
                 aws_region: str = 'us-east-1',
                 max_chunk_size: int = 2000,
                 chunk_overlap: int = 200,
                 debug: bool = False):
        """
        Initialize the PDF processor.
        
        Args:
            aws_region: AWS region for Textract service
            max_chunk_size: Maximum characters per chunk for LangChain
            chunk_overlap: Overlap between chunks
            debug: Enable debug logging
        """
        self.aws_region = aws_region
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.debug = debug
        self._textract_client = None
    
    # ============================================================================
    # EXTRACTION METHODS
    # ============================================================================
    
    def extract_with_textract(self, pdf_path: str) -> ProcessingResult:
        """
        Extract text from PDF using AWS Textract (synchronous processing).
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ProcessingResult with extracted text and metadata
        """
        result = ProcessingResult(
            success=False,
            document_path=pdf_path,
            document_name=Path(pdf_path).name
        )
        
        if self.debug:
            print(f"[DEBUG] Textract extraction for: {pdf_path}")
        
        try:
            # Initialize Textract client if needed
            if self._textract_client is None:
                self._textract_client = self._initialize_textract_client()
            
            # Check file exists and size
            pdf_file = Path(pdf_path)
            if not pdf_file.exists():
                result.add_error(f"PDF file not found: {pdf_path}")
                return result
            
            file_size = pdf_file.stat().st_size
            file_size_mb = round(file_size / (1024 * 1024), 2)
            result.metadata['file_size_mb'] = file_size_mb
            
            if self.debug:
                print(f"[DEBUG] File size: {file_size_mb} MB")
            
            # Check file size limit (10MB for synchronous processing)
            if file_size > 10 * 1024 * 1024:
                result.add_error(f"File too large for synchronous Textract: {file_size_mb} MB (max 10MB)")
                result.add_warning("Consider using PyMuPDF for larger files")
                return result
            
            # Read the PDF file
            with open(pdf_path, 'rb') as document:
                document_bytes = document.read()
            
            if self.debug:
                print("[DEBUG] Calling Textract API...")
            
            # Call Textract
            response = self._textract_client.detect_document_text(
                Document={'Bytes': document_bytes}
            )
            
            if self.debug:
                print(f"[DEBUG] Received {len(response.get('Blocks', []))} blocks")
            
            # Extract text from blocks
            extracted_lines = []
            confidences = []
            page_count = 0
            
            for block in response.get('Blocks', []):
                if block['BlockType'] == 'PAGE':
                    page_count += 1
                    extracted_lines.append(f"\n--- Page {page_count} ---\n")
                elif block['BlockType'] == 'LINE':
                    text = block.get('Text', '')
                    confidence = block.get('Confidence', 0)
                    extracted_lines.append(text)
                    confidences.append(confidence)
            
            result.extracted_text = '\n'.join(extracted_lines)
            result.metadata['page_count'] = page_count
            result.metadata['average_confidence'] = sum(confidences) / len(confidences) if confidences else 0
            result.metadata['extraction_method'] = 'textract'
            result.success = True
            
            if self.debug:
                print(f"[DEBUG] Extraction successful: {len(result.extracted_text)} characters")
                print(f"[DEBUG] Pages: {page_count}, Avg confidence: {result.metadata['average_confidence']:.1f}%")
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_msg = f"AWS Textract error ({error_code}): {e.response['Error']['Message']}"
            result.add_error(error_msg)
            if self.debug:
                print(f"[DEBUG] {error_msg}")
        
        except Exception as e:
            result.add_error(f"Error processing PDF with Textract: {str(e)}")
            if self.debug:
                print(f"[DEBUG] Exception: {str(e)}")
        
        return result
    
    def extract_with_pymupdf(self, pdf_path: str, use_layout: bool = False) -> ProcessingResult:
        """
        Extract text from PDF using PyMuPDF.
        
        Args:
            pdf_path: Path to the PDF file
            use_layout: If True, use block-based extraction for better paragraph detection
            
        Returns:
            ProcessingResult with extracted text and metadata
        """
        result = ProcessingResult(
            success=False,
            document_path=pdf_path,
            document_name=Path(pdf_path).name
        )
        
        if self.debug:
            print(f"[DEBUG] PyMuPDF extraction for: {pdf_path} (layout mode: {use_layout})")
        
        try:
            # Check file exists
            pdf_file = Path(pdf_path)
            if not pdf_file.exists():
                result.add_error(f"PDF file not found: {pdf_path}")
                return result
            
            file_size = pdf_file.stat().st_size
            file_size_mb = round(file_size / (1024 * 1024), 2)
            result.metadata['file_size_mb'] = file_size_mb
            
            # Open the PDF document
            doc = fitz.open(pdf_path)
            text_parts = []
            page_count = doc.page_count
            
            if self.debug:
                print(f"[DEBUG] PDF opened. Pages: {page_count}")
            
            for page_num in range(page_count):
                page = doc[page_num]
                text_parts.append(f"\n--- Page {page_num + 1} ---\n")
                
                if use_layout:
                    # Block-based extraction for better paragraph preservation
                    blocks = page.get_text("blocks")
                    
                    for block in blocks:
                        if block[6] == 0:  # Text block (not image)
                            block_text = block[4]
                            # Clean up block text
                            lines = [line.strip() for line in block_text.split('\n') if line.strip()]
                            cleaned_block = ' '.join(lines)
                            if cleaned_block:
                                text_parts.append(cleaned_block + "\n\n")
                else:
                    # Simple text extraction
                    page_text = page.get_text()
                    text_parts.append(page_text)
                    text_parts.append("\n")
            
            # Store page count before closing
            result.metadata['page_count'] = page_count
            
            # Close document
            doc.close()
            
            # Now build the final text after document is closed
            result.extracted_text = ''.join(text_parts)
            result.metadata['extraction_method'] = 'pymupdf_layout' if use_layout else 'pymupdf_simple'
            result.success = True
            
            if self.debug:
                print(f"[DEBUG] Extraction successful: {len(result.extracted_text)} characters")
        
        except Exception as e:
            result.add_error(f"Error processing PDF with PyMuPDF: {str(e)}")
            if self.debug:
                print(f"[DEBUG] Exception: {str(e)}")
        
        return result
    
    # ============================================================================
    # TEXT CLEANING METHODS
    # ============================================================================
    
    def clean_text(self, text: str, validate: bool = True) -> Tuple[str, List[str]]:
        """
        Clean and renumber legal document text.
        
        Args:
            text: Raw extracted text
            validate: Whether to validate the cleaned output
            
        Returns:
            Tuple of (cleaned_text, warnings)
        """
        if self.debug:
            print("[DEBUG] Starting text cleaning pipeline")
        
        warnings = []
        
        # Step 1: Remove page markers
        text = self._remove_page_markers(text)
        
        # Step 2: Normalize whitespace
        text = self._normalize_whitespace(text)
        
        # Step 3: Parse document structure
        sections = self._parse_sections(text)
        
        if self.debug:
            print(f"[DEBUG] Parsed {len(sections)} main sections")
        
        # Step 4: Renumber sections
        cleaned_text = self._renumber_sections(sections)
        
        # Step 5: Validate if requested
        if validate:
            validation_warnings = self._validate_cleaned_text(cleaned_text, sections)
            warnings.extend(validation_warnings)
        
        if self.debug:
            print(f"[DEBUG] Cleaning complete. Warnings: {len(warnings)}")
        
        return cleaned_text, warnings
    
    def _remove_page_markers(self, text: str) -> str:
        """Remove page break markers like '--- Page N ---'"""
        if self.debug:
            print("[DEBUG] Removing page markers")
        return re.sub(r'-+\s*Page\s+\d+\s*-+\s*\n', '', text, flags=re.IGNORECASE)
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize excessive whitespace while preserving structure"""
        if self.debug:
            print("[DEBUG] Normalizing whitespace")
        
        # Collapse multiple spaces to single space
        text = re.sub(r' +', ' ', text)
        # Limit consecutive newlines to 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Strip whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        return '\n'.join(lines)
    
    def _parse_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse document into hierarchical sections using simplified logic.
        
        Simplified heuristic:
        - A numbered line is a main section if the next numbered line starts with '1.'
        - Otherwise it's a subsection
        """
        if self.debug:
            print("[DEBUG] Parsing document structure")
        
        lines = text.split('\n')
        sections = []
        current_section = None
        current_subsection = None
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
            
            # Check if line starts with number and period
            numbered_match = re.match(r'^(\d+)\.(\d+)?\s+(.+)$', line)
            
            if numbered_match:
                main_num = numbered_match.group(1)
                sub_num = numbered_match.group(2)
                content = numbered_match.group(3)
                
                if sub_num is None:
                    # This is a main section (e.g., "1. Title")
                    # Look ahead to confirm
                    next_numbered_idx = self._find_next_numbered_line(lines, i + 1)
                    
                    is_main_section = False
                    if next_numbered_idx is None:
                        # Last numbered item, treat as main section
                        is_main_section = True
                    else:
                        next_line = lines[next_numbered_idx].strip()
                        next_match = re.match(r'^(\d+)\.(\d+)?\s+', next_line)
                        if next_match:
                            next_main = next_match.group(1)
                            next_sub = next_match.group(2)
                            # If next line is X.1 or 1.something, current is main section
                            if next_sub == '1' or next_main == '1':
                                is_main_section = True
                    
                    if is_main_section:
                        # Save previous section
                        if current_section:
                            sections.append(current_section)
                        
                        current_section = {
                            'type': 'main_section',
                            'number': main_num,
                            'title': content,
                            'subsections': [],
                            'content': []
                        }
                        current_subsection = None
                        
                        if self.debug:
                            print(f"[DEBUG] Main section: {main_num}. {content[:50]}")
                    else:
                        # This is actually a subsection that wasn't formatted as X.Y
                        if current_section:
                            current_section['content'].append(line)
                else:
                    # This is a subsection (e.g., "1.1. Title")
                    if current_section:
                        current_subsection = {
                            'type': 'subsection',
                            'number': f"{main_num}.{sub_num}",
                            'content': [content],
                            'bullets': []
                        }
                        current_section['subsections'].append(current_subsection)
                        
                        if self.debug:
                            print(f"[DEBUG]   Subsection: {main_num}.{sub_num} {content[:50]}")
            
            elif line.startswith('-'):
                # Bullet point
                bullet_text = line[1:].strip()
                if bullet_text:
                    if current_subsection:
                        current_subsection['bullets'].append(bullet_text)
                    elif current_section:
                        current_section['content'].append(line)
            
            else:
                # Regular content line
                if current_subsection:
                    current_subsection['content'].append(line)
                elif current_section:
                    current_section['content'].append(line)
            
            i += 1
        
        # Add last section
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _find_next_numbered_line(self, lines: List[str], start_idx: int) -> Optional[int]:
        """Find the index of the next line that starts with a number"""
        for i in range(start_idx, len(lines)):
            line = lines[i].strip()
            if re.match(r'^\d+\.', line):
                return i
        return None
    
    def _renumber_sections(self, sections: List[Dict[str, Any]]) -> str:
        """Renumber sections consistently and format output"""
        if self.debug:
            print("[DEBUG] Renumbering sections")
        
        output_lines = []
        
        for main_idx, section in enumerate(sections, 1):
            # Main section header
            output_lines.append(f"{main_idx}. {section['title']}")
            output_lines.append("")
            
            # Section-level content
            for content in section['content']:
                if content.strip():
                    output_lines.append(content)
            
            if section['content']:
                output_lines.append("")
            
            # Subsections
            for sub_idx, subsection in enumerate(section['subsections'], 1):
                subsection_number = f"{main_idx}.{sub_idx}"
                
                # Subsection header and content
                if subsection['content']:
                    output_lines.append(f"{subsection_number}. {subsection['content'][0]}")
                    
                    # Additional content lines
                    for content_line in subsection['content'][1:]:
                        if content_line.strip():
                            output_lines.append(content_line)
                
                # Bullets
                if subsection['bullets']:
                    for bullet in subsection['bullets']:
                        output_lines.append(f"  - {bullet}")
                
                output_lines.append("")
            
            output_lines.append("")
        
        return '\n'.join(output_lines).strip()
    
    def _validate_cleaned_text(self, cleaned_text: str, sections: List[Dict[str, Any]]) -> List[str]:
        """Validate the cleaned text and return warnings"""
        warnings = []
        
        # Check if we have sections
        if not sections:
            warnings.append("No sections were detected in the document")
            return warnings
        
        # Check for very short sections
        for section in sections:
            if not section['subsections'] and not section['content']:
                warnings.append(f"Section '{section['title']}' appears empty")
        
        # Check section numbering continuity
        expected_num = 1
        for section in sections:
            actual_num = int(section['number'])
            if actual_num != expected_num:
                warnings.append(f"Section numbering gap: expected {expected_num}, found {actual_num}")
            expected_num = actual_num + 1
        
        if self.debug and warnings:
            print(f"[DEBUG] Validation found {len(warnings)} warnings:")
            for warning in warnings:
                print(f"[DEBUG]   - {warning}")
        
        return warnings
    
    # ============================================================================
    # CHUNKING METHODS
    # ============================================================================
    
    def chunk_document(self, cleaned_text: str, document_title: str = None) -> List[Dict[str, Any]]:
        """
        Chunk cleaned text using LangChain with section-aware splitting.
        
        Args:
            cleaned_text: Text cleaned with clean_text()
            document_title: Optional document title for metadata
            
        Returns:
            List of chunks with metadata
        """
        if self.debug:
            print("[DEBUG] Starting document chunking")
        
        # Convert to markdown headers
        markdown_text = self._convert_to_markdown(cleaned_text)
        
        # Define header hierarchy
        headers_to_split_on = [
            ("##", "section"),
            ("###", "subsection"),
            ("####", "subsubsection")
        ]
        
        # Create markdown splitter
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        
        # Split by headers
        header_chunks = markdown_splitter.split_text(markdown_text)
        
        if self.debug:
            print(f"[DEBUG] Created {len(header_chunks)} header-based chunks")
        
        # Process each chunk
        final_chunks = []
        
        for chunk in header_chunks:
            chunk_text = chunk.page_content
            chunk_metadata = chunk.metadata.copy()
            
            if document_title:
                chunk_metadata["document_title"] = document_title
            
            # Split oversized chunks
            if len(chunk_text) > self.max_chunk_size:
                sub_chunks = self._split_large_chunk(chunk_text, chunk_metadata)
                final_chunks.extend(sub_chunks)
            else:
                chunk_metadata["is_split_chunk"] = False
                final_chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata,
                    "chunk_size": len(chunk_text)
                })
        
        # Add hierarchical context
        final_chunks = self._add_section_hierarchy(final_chunks)
        
        if self.debug:
            print(f"[DEBUG] Final chunk count: {len(final_chunks)}")
            avg_size = sum(c['chunk_size'] for c in final_chunks) / len(final_chunks)
            print(f"[DEBUG] Average chunk size: {avg_size:.0f} characters")
        
        return final_chunks
    
    def _convert_to_markdown(self, text: str) -> str:
        """Convert numbered sections to markdown headers"""
        # Main sections (1. Title) to ## headers
        text = re.sub(r'^(\d+)\.\s+([A-Z][^\n]+)', r'## \1. \2', text, flags=re.MULTILINE)
        
        # Subsections (1.1 Title) to ### headers  
        text = re.sub(r'^(\d+\.\d+)\.?\s+([A-Z][^\n]+)', r'### \1 \2', text, flags=re.MULTILINE)
        
        # Sub-subsections (1.1.1 Title) to #### headers
        text = re.sub(r'^(\d+\.\d+\.\d+)\.?\s+([A-Z][^\n]+)', r'#### \1 \2', text, flags=re.MULTILINE)
        
        return text
    
    def _split_large_chunk(self, text: str, base_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split a chunk that exceeds max_chunk_size"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        sub_texts = splitter.split_text(text)
        sub_chunks = []
        
        for i, sub_text in enumerate(sub_texts):
            metadata = base_metadata.copy()
            metadata["chunk_part"] = f"{i+1}/{len(sub_texts)}"
            metadata["is_split_chunk"] = True
            
            sub_chunks.append({
                "text": sub_text,
                "metadata": metadata,
                "chunk_size": len(sub_text)
            })
        
        return sub_chunks
    
    def _add_section_hierarchy(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add hierarchical context to chunks"""
        for chunk in chunks:
            metadata = chunk["metadata"]
            context_parts = []
            
            if "section" in metadata:
                context_parts.append(f"Section: {metadata['section']}")
            if "subsection" in metadata:
                context_parts.append(f"Subsection: {metadata['subsection']}")
            if "subsubsection" in metadata:
                context_parts.append(f"Sub-subsection: {metadata['subsubsection']}")
            
            if context_parts:
                chunk["metadata"]["section_hierarchy"] = " > ".join(context_parts)
        
        return chunks
    
    def create_langchain_documents(self, chunks: List[Dict[str, Any]]) -> List[Document]:
        """
        Convert chunks to LangChain Document objects for vector store ingestion.
        
        Args:
            chunks: List of chunks from chunk_document()
            
        Returns:
            List of LangChain Document objects
        """
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk["text"],
                metadata=chunk["metadata"]
            )
            documents.append(doc)
        
        if self.debug:
            print(f"[DEBUG] Created {len(documents)} LangChain documents")
        
        return documents
    
    # ============================================================================
    # PIPELINE METHODS
    # ============================================================================
    
    def process_single_pdf(self, 
                          pdf_path: str,
                          extraction_method: str = 'pymupdf',
                          use_layout: bool = False,
                          document_title: Optional[str] = None) -> ProcessingResult:
        """
        Complete pipeline for a single PDF.
        
        Args:
            pdf_path: Path to PDF file
            extraction_method: 'pymupdf' or 'textract'
            use_layout: Use layout-aware extraction (PyMuPDF only)
            document_title: Optional title for metadata
            
        Returns:
            ProcessingResult with complete processing information
        """
        if self.debug:
            print(f"\n{'='*80}")
            print(f"PROCESSING: {Path(pdf_path).name}")
            print(f"{'='*80}")
        
        # Extract text
        if extraction_method.lower() == 'textract':
            result = self.extract_with_textract(pdf_path)
        else:
            result = self.extract_with_pymupdf(pdf_path, use_layout)
        
        if not result.success:
            return result
        
        # Clean text
        cleaned_text, warnings = self.clean_text(result.extracted_text)
        result.cleaned_text = cleaned_text
        result.warnings.extend(warnings)
        
        # Chunk text
        try:
            title = document_title or Path(pdf_path).stem
            chunks = self.chunk_document(cleaned_text, title)
            result.chunks = chunks
            result.metadata['chunk_count'] = len(chunks)
        except Exception as e:
            result.add_error(f"Chunking failed: {str(e)}")
        
        if self.debug:
            print(f"\n[DEBUG] Processing complete for {result.document_name}")
            print(f"[DEBUG] Success: {result.success}")
            print(f"[DEBUG] Errors: {len(result.errors)}")
            print(f"[DEBUG] Warnings: {len(result.warnings)}")
            print(f"[DEBUG] Chunks: {len(result.chunks)}")
        
        return result
    
    def process_directory(self,
                         directory_path: str,
                         extraction_method: str = 'pymupdf',
                         use_layout: bool = False,
                         recursive: bool = True) -> List[ProcessingResult]:
        """
        Process all PDFs in a directory.
        
        Args:
            directory_path: Path to directory containing PDFs
            extraction_method: 'pymupdf' or 'textract'
            use_layout: Use layout-aware extraction (PyMuPDF only)
            recursive: Search subdirectories
            
        Returns:
            List of ProcessingResult objects
        """
        if self.debug:
            print(f"\n{'='*80}")
            print(f"PROCESSING DIRECTORY: {directory_path}")
            print(f"{'='*80}")
        
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find all PDFs
        if recursive:
            pdf_files = list(directory.rglob("*.pdf"))
        else:
            pdf_files = list(directory.glob("*.pdf"))
        
        print(f"Found {len(pdf_files)} PDF files")
        
        results = []
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"\nProcessing {i}/{len(pdf_files)}: {pdf_path.name}")
            
            result = self.process_single_pdf(
                str(pdf_path),
                extraction_method,
                use_layout,
                pdf_path.stem
            )
            results.append(result)
            
            if result.success:
                print(f"  ✓ Success - {len(result.chunks)} chunks created")
            else:
                print(f"  ✗ Failed - {len(result.errors)} errors")
        
        # Summary
        successful = sum(1 for r in results if r.success)
        print(f"\n{'='*80}")
        print(f"DIRECTORY PROCESSING COMPLETE")
        print(f"Total files: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")
        print(f"{'='*80}")
        
        return results
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def _initialize_textract_client(self) -> boto3.client:
        """Initialize AWS Textract client"""
        if self.debug:
            print(f"[DEBUG] Initializing Textract client (region: {self.aws_region})")
        
        try:
            client = boto3.client('textract', region_name=self.aws_region)
            if self.debug:
                print("[DEBUG] Textract client initialized successfully")
            return client
        except NoCredentialsError:
            raise Exception("AWS credentials not found. Please configure AWS credentials.")
        except Exception as e:
            raise Exception(f"Failed to initialize Textract client: {str(e)}")


# ============================================================================
# MAIN / TEST FUNCTIONS
# ============================================================================

def main():
    """Test the PDF processor with sample files"""
    
    # Initialize processor
    processor = PDFProcessor(
        max_chunk_size=2000,
        chunk_overlap=200,
        debug=True
    )
    
    print("\n" + "="*80)
    print("PDF PROCESSING PIPELINE TEST")
    print("="*80)
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()
    print(f"\nScript directory: {script_dir}")
    print(f"Current working directory: {Path.cwd()}")
    
    # Test 1: Process single PDF
    print("\n\nTEST 1: Processing single PDF with PyMuPDF")
    print("-"*80)
    
    single_pdf_path = script_dir / "pdfs" / "engineering-department-budget-allocation.pdf"
    print(f"Looking for PDF at: {single_pdf_path}")
    
    if single_pdf_path.exists():
        result = processor.process_single_pdf(
            str(single_pdf_path),
            extraction_method='pymupdf',
            use_layout=False
        )
        
        print("\nSingle PDF Results:")
        print(f"  Success: {result.success}")
        print(f"  Extracted text length: {len(result.extracted_text)}")
        print(f"  Cleaned text length: {len(result.cleaned_text)}")
        print(f"  Chunks created: {len(result.chunks)}")
        print(f"  Errors: {len(result.errors)}")
        print(f"  Warnings: {len(result.warnings)}")
        
        if result.errors:
            print("\n  Errors:")
            for error in result.errors:
                print(f"    - {error}")
        
        if result.warnings:
            print("\n  Warnings:")
            for warning in result.warnings:
                print(f"    - {warning}")
        
        # Show first chunk as example
        if result.chunks:
            print("\n  First chunk preview:")
            chunk = result.chunks[0]
            print(f"    Size: {chunk['chunk_size']} chars")
            print(f"    Metadata: {chunk['metadata']}")
            print(f"    Text preview: {chunk['text'][:150]}...")
        
        # Test creating LangChain documents
        langchain_docs = processor.create_langchain_documents(result.chunks)
        print(f"\n  LangChain documents created: {len(langchain_docs)}")
    else:
        print(f"  Skipping - file not found: {single_pdf_path}")
    
    # Test 2: Process directory
    # print("\n\nTEST 2: Processing directory with PyMuPDF")
    # print("-"*80)
    
    # pdf_directory = script_dir / "pdfs"
    # print(f"Looking for PDFs in: {pdf_directory}")
    
    # if pdf_directory.exists():
    #     results = processor.process_directory(
    #         str(pdf_directory),
    #         extraction_method='pymupdf',
    #         use_layout=False,
    #         recursive=True
    #     )
        
    #     print("\nDirectory Processing Summary:")
    #     print(f"  Total PDFs processed: {len(results)}")
        
    #     successful_results = [r for r in results if r.success]
    #     print(f"  Successful: {len(successful_results)}")
        
    #     total_chunks = sum(len(r.chunks) for r in successful_results)
    #     print(f"  Total chunks created: {total_chunks}")
        
    #     if successful_results:
    #         avg_chunks = total_chunks / len(successful_results)
    #         print(f"  Average chunks per document: {avg_chunks:.1f}")
        
    #     # Show statistics
    #     all_chunks = [chunk for r in successful_results for chunk in r.chunks]
    #     if all_chunks:
    #         chunk_sizes = [c['chunk_size'] for c in all_chunks]
    #         print(f"\n  Chunk size statistics:")
    #         print(f"    Min: {min(chunk_sizes)} chars")
    #         print(f"    Max: {max(chunk_sizes)} chars")
    #         print(f"    Average: {sum(chunk_sizes)/len(chunk_sizes):.0f} chars")
    # else:
    #     print(f"  Skipping - directory not found: {pdf_directory}")
    
    # Test 3: Test with layout mode
    # print("\n\nTEST 3: Processing with layout-aware extraction")
    # print("-"*80)
    
    # if single_pdf_path.exists():
    #     result_layout = processor.process_single_pdf(
    #         str(single_pdf_path),
    #         extraction_method='pymupdf',
    #         use_layout=True
    #     )
        
    #     print("\nLayout Mode Results:")
    #     print(f"  Success: {result_layout.success}")
    #     print(f"  Chunks created: {len(result_layout.chunks)}")
        
    #     # Compare with non-layout mode
    #     if 'result' in locals() and result.success and result_layout.success:
    #         print(f"\n  Comparison (simple vs layout):")
    #         print(f"    Extracted text length: {len(result.extracted_text)} vs {len(result_layout.extracted_text)}")
    #         print(f"    Cleaned text length: {len(result.cleaned_text)} vs {len(result_layout.cleaned_text)}")
    #         print(f"    Chunk count: {len(result.chunks)} vs {len(result_layout.chunks)}")
    # else:
    #     print(f"  Skipping - file not found: {single_pdf_path}")
    
    # print("\n" + "="*80)
    # print("TESTING COMPLETE")
    # print("="*80)


if __name__ == "__main__":
    main()














    # Note: final paragraph assumption of main header didnt work properly. Decided that a subsection was a main section because of it I think