"""
PDF Text Extraction and Chunking Pipeline for RAG Systems

This module provides a complete pipeline for extracting text from PDFs,
cleaning and standardizing the text (especially for legal documents),
and chunking it for use in RAG systems.

Supports both AWS Textract and PyMuPDF for extraction.
"""

import os
import re
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.schema import Document

# Optional imports for AWS Textract
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    TEXTRACT_AVAILABLE = True
except ImportError:
    TEXTRACT_AVAILABLE = False
    ClientError = Exception
    NoCredentialsError = Exception


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

        if not TEXTRACT_AVAILABLE:
            result.add_error("AWS Textract not available - boto3 not installed")
            return result

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
        Extract text from PDF using PyMuPDF with formatting metadata.

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
            page_text_map = {}  # Map to track which text came from which page

            if self.debug:
                print(f"[DEBUG] PDF opened. Pages: {page_count}")

            for page_num in range(page_count):
                page = doc[page_num]
                page_start_pos = len(''.join(text_parts))
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

                # Track page boundaries
                page_end_pos = len(''.join(text_parts))
                page_text_map[page_num + 1] = (page_start_pos, page_end_pos)

            # Store page count before closing
            result.metadata['page_count'] = page_count
            result.metadata['page_text_map'] = page_text_map

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

    def extract_with_formatting(self, pdf_path: str) -> ProcessingResult:
        """
        Extract text from PDF with detailed formatting information for header detection.
        Uses font properties (size, weight, flags) to identify headers.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            ProcessingResult with extracted text and formatting metadata
        """
        result = ProcessingResult(
            success=False,
            document_path=pdf_path,
            document_name=Path(pdf_path).name
        )

        if self.debug:
            print(f"[DEBUG] Format-aware extraction for: {pdf_path}")

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
            page_count = doc.page_count

            if self.debug:
                print(f"[DEBUG] PDF opened. Pages: {page_count}")

            # Extract text with formatting details
            formatted_blocks = []

            for page_num in range(page_count):
                page = doc[page_num]

                # Get text as dictionary with detailed formatting
                text_dict = page.get_text("dict")

                # Calculate font size statistics for this page to find headers
                page_font_sizes = []
                for block in text_dict.get("blocks", []):
                    if block.get("type") == 0:
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                page_font_sizes.append(span.get("size", 0))

                # Determine normal body font size (most common)
                normal_font_size = max(set(page_font_sizes), key=page_font_sizes.count) if page_font_sizes else 11

                for block in text_dict.get("blocks", []):
                        if block.get("type") == 0:  # Text block
                            for line in block.get("lines", []):
                                line_text = ""
                                # Track font properties for this line
                                font_sizes = []
                                font_flags = []

                                for span in line.get("spans", []):
                                    text = span.get("text", "")
                                    line_text += text
                                    font_sizes.append(span.get("size", 0))
                                    font_flags.append(span.get("flags", 0))

                                line_text = line_text.strip()
                                if line_text:
                                    # Calculate line properties
                                    avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0
                                    # Check if bold (flag & 16) or if all uppercase
                                    is_bold = any(flag & 16 for flag in font_flags)
                                    is_all_caps = line_text.isupper() and len(line_text) > 3
                                    is_larger = avg_font_size > normal_font_size
                                    is_short = len(line_text) < 80  # Headers are typically short

                                    # Improved header detection:
                                    # 1. Bold + All Caps (like "CALL TO ORDER")
                                    # 2. Bold + Larger font (like "1. Introduction and Purpose")
                                    # 3. Bold + Short + Starts with number/letter (like "2. Employment Policies")
                                    is_likely_header = (
                                        (is_bold and is_all_caps) or
                                        (is_bold and is_larger) or
                                        (is_bold and is_short and re.match(r'^[\d\w]', line_text))
                                    )

                                    formatted_blocks.append({
                                        'text': line_text,
                                        'page': page_num + 1,
                                        'font_size': avg_font_size,
                                        'is_bold': is_bold,
                                        'is_all_caps': is_all_caps,
                                        'is_larger': is_larger,
                                        'is_likely_header': is_likely_header
                                    })

                                    if self.debug and is_likely_header:
                                        print(f"[DEBUG] Detected likely header on page {page_num + 1}: '{line_text[:60]}...'")

            doc.close()

            if self.debug:
                print(f"[DEBUG] Extracted {len(formatted_blocks)} raw blocks")

            # Reconstruct lines broken by PDF wrapping
            formatted_blocks = self._reconstruct_wrapped_lines(formatted_blocks)

            if self.debug:
                print(f"[DEBUG] After line reconstruction: {len(formatted_blocks)} blocks")

            # Store formatted blocks for later processing
            result.metadata['formatted_blocks'] = formatted_blocks
            result.metadata['page_count'] = page_count
            result.metadata['extraction_method'] = 'pymupdf_formatted'

            # Build plain text with header markers
            text_parts = []
            for block in formatted_blocks:
                if block['is_likely_header']:
                    text_parts.append(f"\n## {block['text']}\n")
                else:
                    text_parts.append(block['text'] + "\n")

            result.extracted_text = ''.join(text_parts)
            result.success = True

            if self.debug:
                print(f"[DEBUG] Format-aware extraction successful")
                print(f"[DEBUG] Total blocks: {len(formatted_blocks)}")
                header_count = sum(1 for b in formatted_blocks if b['is_likely_header'])
                print(f"[DEBUG] Detected headers: {header_count}")

        except Exception as e:
            result.add_error(f"Error processing PDF with formatting extraction: {str(e)}")
            if self.debug:
                print(f"[DEBUG] Exception: {str(e)}")

        return result

    def _reconstruct_wrapped_lines(self, formatted_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reconstruct lines that were broken by PDF line wrapping.
        Merges continuation lines to prevent false header detection.

        Args:
            formatted_blocks: List of text blocks with formatting metadata

        Returns:
            List of blocks with wrapped lines merged
        """
        if not formatted_blocks:
            return formatted_blocks

        if self.debug:
            print("[DEBUG] Reconstructing wrapped lines")

        reconstructed = []
        buffer = None

        for block in formatted_blocks:
            if buffer is None:
                buffer = block.copy()
                continue

            # Check if current block should be merged with buffer
            if self._should_merge_lines(buffer, block):
                # Merge the lines
                buffer['text'] += ' ' + block['text']

                if self.debug:
                    print(f"[DEBUG]   Merged: '{buffer['text'][:80]}...'")
            else:
                # Save buffer and start new one
                # Re-evaluate header status after reconstruction
                buffer = self._reevaluate_header_status(buffer)
                reconstructed.append(buffer)
                buffer = block.copy()

        # Don't forget the last buffer
        if buffer:
            buffer = self._reevaluate_header_status(buffer)
            reconstructed.append(buffer)

        if self.debug:
            merge_count = len(formatted_blocks) - len(reconstructed)
            print(f"[DEBUG] Merged {merge_count} wrapped lines")

        return reconstructed

    def _should_merge_lines(self, prev: Dict[str, Any], curr: Dict[str, Any]) -> bool:
        """
        Determine if current line should be merged with previous line.

        Merge if:
        - Same page
        - Same formatting (bold status, similar font size)
        - Previous doesn't end with sentence terminator
        - Current starts with continuation indicator
        """
        # Must be on same page
        if prev['page'] != curr['page']:
            return False

        # Must have same bold status (don't merge header with content)
        if prev['is_bold'] != curr['is_bold']:
            return False

        # Font size should be similar (within 1pt)
        if abs(prev['font_size'] - curr['font_size']) > 1.0:
            return False

        prev_text = prev['text'].strip()
        curr_text = curr['text'].strip()

        # Don't merge if previous line is very short (likely a header)
        if len(prev_text) < 15:
            return False

        # Check if previous line ends with sentence terminator
        sentence_terminators = ('.', ':', '!', '?', ';')
        if prev_text.endswith(sentence_terminators):
            return False

        # Check if current line starts with continuation markers
        continuation_starts = (
            curr_text[0].islower() if curr_text else False,
            curr_text.startswith('and '),
            curr_text.startswith('or '),
            curr_text.startswith('the '),
            curr_text.startswith('to '),
            curr_text.startswith('of '),
            curr_text.startswith('in '),
            curr_text.startswith('for '),
            curr_text.startswith('with ')
        )

        return any(continuation_starts)

    def _reevaluate_header_status(self, block: Dict[str, Any]) -> Dict[str, Any]:
        """
        Re-evaluate if a block is a header after line reconstruction.
        Uses multi-signal scoring for better accuracy.

        Scoring criteria:
        - MUST be bold OR all-caps (mandatory)
        - THEN need 2+ additional signals from:
          1. Larger font
          2. Reasonable length (15-80 chars)
          3. Ends with colon OR is short phrase
          4. Not a list item pattern
        """
        text = block['text'].strip()

        # MANDATORY: Must be bold OR all-caps
        is_bold = block.get('is_bold', False)
        is_all_caps = block.get('is_all_caps', False)

        if not (is_bold or is_all_caps):
            block['is_likely_header'] = False
            block['header_score'] = 0
            block['header_signals'] = []
            return block

        # Calculate additional signals
        score = 0
        reasons = []

        # Signal 1: Larger font
        if block.get('is_larger', False):
            score += 1
            reasons.append("larger")

        # Signal 2: Reasonable header length (15-80 chars)
        if 15 <= len(text) <= 80:
            score += 1
            reasons.append("good-length")

        # Signal 3: Ends with colon OR is short standalone phrase
        if text.endswith(':') or (len(text) < 40 and not ',' in text):
            score += 1
            reasons.append("header-pattern")

        # Signal 4: NOT a list item (doesn't match common list patterns)
        is_list_item = (
            text.count(',') >= 2 or  # Has multiple commas (like "Name, Title, Dept")
            re.match(r'^-\s+', text) or  # Starts with bullet dash
            re.match(r'^\d+\)\s+', text) or  # Starts with 1) 2) etc
            re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+,\s+', text)  # "First Last, Title" pattern
        )
        if not is_list_item:
            score += 1
            reasons.append("not-list")

        # Require 2+ additional signals beyond bold/caps
        is_header = score >= 2

        # Add the mandatory signal to reasons
        if is_bold:
            reasons.insert(0, "bold")
        if is_all_caps:
            reasons.insert(0, "all-caps")

        if self.debug and block.get('is_likely_header', False) != is_header:
            status = "NOW HEADER" if is_header else "NO LONGER HEADER"
            print(f"[DEBUG]   {status} (score {score}/4 + mandatory: {', '.join(reasons)}): '{text[:60]}'")

        block['is_likely_header'] = is_header
        block['header_score'] = score
        block['header_signals'] = reasons

        return block

    def _is_sentence_complete(self, text: str) -> bool:
        """
        Check if text appears to be a complete sentence or standalone phrase.

        Complete if:
        - Ends with sentence terminator (. : ! ? ;)
        - Is a short standalone phrase (< 40 chars and doesn't end with continuation word)
        """
        text = text.strip()

        # Ends with sentence terminator
        if text.endswith(('.', ':', '!', '?', ';')):
            return True

        # Short standalone phrase
        if len(text) < 40:
            continuation_endings = ('and', 'or', 'the', 'a', 'an', 'of', 'to', 'in', 'for', 'with', 'by')
            words = text.lower().split()
            if words and words[-1] not in continuation_endings:
                return True

        return False

    # ============================================================================
    # TEXT CLEANING METHODS
    # ============================================================================
    
    def clean_text(self, text: str, validate: bool = True, formatted_blocks: Optional[List[Dict]] = None) -> Tuple[str, List[str]]:
        """
        Clean and structure document text.

        Args:
            text: Raw extracted text
            validate: Whether to validate the cleaned output
            formatted_blocks: Optional formatting metadata from extract_with_formatting()

        Returns:
            Tuple of (cleaned_text, warnings)
        """
        if self.debug:
            print("[DEBUG] Starting text cleaning pipeline")

        warnings = []
        original_text = text

        # Step 1: Remove page markers
        text = self._remove_page_markers(text)

        # Step 2: Normalize whitespace
        text = self._normalize_whitespace(text)

        # Step 3: Parse document structure (try format-based first, then numbered)
        if formatted_blocks:
            if self.debug:
                print("[DEBUG] Using format-based header detection")
            sections = self._parse_formatted_sections(formatted_blocks)
        else:
            if self.debug:
                print("[DEBUG] Attempting numbered section detection")
            sections = self._parse_sections(text)

        if self.debug:
            print(f"[DEBUG] Parsed {len(sections)} main sections")

        # Step 4: Check if we found any sections
        if not sections or (len(sections) == 1 and not sections[0].get('subsections') and not sections[0].get('content')):
            warnings.append("No structured sections detected - using minimal cleaning")
            if self.debug:
                print("[DEBUG] WARNING: No structured sections found, using fallback")
            # Return the normalized text as-is for fallback chunking
            return text, warnings

        # Step 5: Format sections into clean text
        if formatted_blocks:
            cleaned_text = self._format_formatted_sections(sections)
        else:
            cleaned_text = self._renumber_sections(sections)

        # Step 6: Validate if requested
        if validate:
            validation_warnings = self._validate_cleaned_text(cleaned_text, original_text, sections)
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

    def _parse_formatted_sections(self, formatted_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parse document into sections based on formatting metadata (bold + caps).

        Args:
            formatted_blocks: List of text blocks with formatting info

        Returns:
            List of section dictionaries
        """
        if self.debug:
            print("[DEBUG] Parsing document using formatting metadata")

        sections = []
        current_section = None

        for block in formatted_blocks:
            text = block['text']

            if block['is_likely_header']:
                # This is a header - start new section
                if current_section:
                    sections.append(current_section)

                current_section = {
                    'type': 'section',
                    'title': text,
                    'content': [],
                    'page': block['page']
                }

                if self.debug:
                    print(f"[DEBUG] New section: '{text}'")

            else:
                # Regular content
                if current_section:
                    current_section['content'].append(text)
                else:
                    # Content before first header - create preamble section
                    if not sections or sections[0]['title'] != 'PREAMBLE':
                        current_section = {
                            'type': 'section',
                            'title': 'PREAMBLE',
                            'content': [text],
                            'page': block['page']
                        }

        # Add last section
        if current_section:
            sections.append(current_section)

        if self.debug:
            print(f"[DEBUG] Parsed {len(sections)} sections from formatting")

        return sections

    def _format_formatted_sections(self, sections: List[Dict[str, Any]]) -> str:
        """
        Format sections detected by formatting into clean text with markdown headers.

        Args:
            sections: List of section dictionaries from _parse_formatted_sections

        Returns:
            Formatted text string
        """
        if self.debug:
            print("[DEBUG] Formatting sections into clean text")

        output_lines = []

        for section in sections:
            # Section header as markdown
            output_lines.append(f"## {section['title']}")
            output_lines.append("")

            # Section content
            for content in section['content']:
                if content.strip():
                    output_lines.append(content)

            output_lines.append("")

        return '\n'.join(output_lines).strip()
    
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
    
    def _validate_cleaned_text(self, cleaned_text: str, original_text: str, sections: List[Dict[str, Any]]) -> List[str]:
        """Validate the cleaned text and return warnings with structural checks"""
        warnings = []

        # Check if we have sections
        if not sections:
            warnings.append("No sections were detected in the document")
            return warnings

        # Check for significant content loss
        orig_len = len(re.sub(r'\s', '', original_text))
        clean_len = len(re.sub(r'\s', '', cleaned_text))

        if orig_len > 0:
            loss_pct = (1 - clean_len / orig_len) * 100
            if loss_pct > 10:
                warnings.append(f"Significant content loss detected: {loss_pct:.1f}%")
            elif self.debug:
                print(f"[DEBUG] Content preservation: {100 - loss_pct:.1f}%")

        # Structural Validation: Check for suspiciously short sections
        section_lengths = []
        for section in sections:
            title = section.get('title', '')
            content = section.get('content', [])
            total_text = title + ' '.join(content) if isinstance(content, list) else title + content
            section_len = len(total_text)
            section_lengths.append(section_len)

            # Flag sections that are extremely short (< 20 chars = likely fragment)
            if section_len < 20:
                warnings.append(f"Suspiciously short section: '{title[:40]}' ({section_len} chars)")

        # Structural Validation: Check section size distribution
        if section_lengths:
            import statistics
            median_len = statistics.median(section_lengths)
            max_len = max(section_lengths)

            # Flag if largest section is 10x bigger than median (might have missed headers)
            if median_len > 0 and max_len > 10 * median_len:
                warnings.append(f"Uneven section sizes detected - largest is {max_len/median_len:.1f}x median (possible missed headers)")

        # Check for very empty sections
        for section in sections:
            section_has_content = (
                section.get('content') or
                section.get('subsections') or
                section.get('title')
            )
            if not section_has_content:
                warnings.append(f"Section appears empty: {section.get('title', 'Unknown')}")

        # Check section numbering continuity (only for numbered sections)
        if sections and 'number' in sections[0]:
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
        """
        Convert numbered sections to markdown headers.
        Fixed to only match actual section headers, not content lines.
        """
        lines = text.split('\n')
        output_lines = []

        for line in lines:
            # Check for sub-subsection (1.1.1)
            match = re.match(r'^(\d+\.\d+\.\d+)\.?\s+(.+)$', line)
            if match and self._is_likely_section_header(match.group(2)):
                output_lines.append(f'#### {match.group(1)} {match.group(2)}')
                continue

            # Check for subsection (1.1)
            match = re.match(r'^(\d+\.\d+)\.?\s+(.+)$', line)
            if match and self._is_likely_section_header(match.group(2)):
                output_lines.append(f'### {match.group(1)} {match.group(2)}')
                continue

            # Check for main section (1.)
            match = re.match(r'^(\d+)\.\s+(.+)$', line)
            if match and self._is_likely_section_header(match.group(2)):
                output_lines.append(f'## {match.group(1)}. {match.group(2)}')
                continue

            # Check if line already has markdown header (from formatting extraction)
            if line.startswith('##'):
                output_lines.append(line)
                continue

            # Regular line - keep as is
            output_lines.append(line)

        return '\n'.join(output_lines)

    def _is_likely_section_header(self, text: str) -> bool:
        """
        Heuristic to determine if text is likely a section header vs content.

        A line is likely a header if:
        - Starts with capital letter
        - Is relatively short (< 100 chars)
        - Doesn't end with continuation indicators
        - Ideally title-cased
        """
        text = text.strip()

        if not text:
            return False

        # Must start with capital
        if not text[0].isupper():
            return False

        # Too long to be a header
        if len(text) > 100:
            return False

        # Ends with continuation (likely incomplete sentence from content)
        if text.endswith((',', 'and', 'or', 'the', 'a', 'an', 'of', 'to', 'in')):
            return False

        # Contains common sentence continuations
        lowered = text.lower()
        if any(lowered.endswith(word) for word in ['applicable to', 'conditions', 'procedures', 'including']):
            return False

        return True
    
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
                          extraction_method: str = 'formatting',
                          use_layout: bool = False,
                          document_title: Optional[str] = None) -> ProcessingResult:
        """
        Complete pipeline for a single PDF.

        Args:
            pdf_path: Path to PDF file
            extraction_method: 'formatting', 'pymupdf', or 'textract'
            use_layout: Use layout-aware extraction (PyMuPDF only, ignored if formatting)
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
            formatted_blocks = None
        elif extraction_method.lower() == 'formatting':
            result = self.extract_with_formatting(pdf_path)
            formatted_blocks = result.metadata.get('formatted_blocks')
        else:
            result = self.extract_with_pymupdf(pdf_path, use_layout)
            formatted_blocks = None

        if not result.success:
            return result

        # Clean text
        cleaned_text, warnings = self.clean_text(result.extracted_text, formatted_blocks=formatted_blocks)
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
                         extraction_method: str = 'formatting',
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
    
    def _initialize_textract_client(self):
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

    @staticmethod
    def get_random_pdf(directory: str, n: int = 1, seed: Optional[int] = None) -> List[str]:
        """
        Get random PDF(s) from a directory tree.
        
        Args:
            directory: Root directory to search
            n: Number of random PDFs to return
            seed: Random seed for reproducibility
            
        Returns:
            List of paths to randomly selected PDFs
        """
        import random
        
        if seed is not None:
            random.seed(seed)
        
        # Find all PDFs recursively
        directory_path = Path(directory)
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        all_pdfs = list(directory_path.rglob("*.pdf"))
        
        if not all_pdfs:
            raise ValueError(f"No PDF files found in {directory}")
        
        # Return n random PDFs
        n = min(n, len(all_pdfs))
        selected = random.sample(all_pdfs, n)
        
        return [str(pdf) for pdf in selected]


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


def test_random_pdfs(directory: str, n: int = 3):
    """Test the processor on random PDFs from a directory"""
    
    print("\n" + "="*80)
    print(f"TESTING ON {n} RANDOM PDFs")
    print("="*80)
    
    # Get random PDFs
    try:
        random_pdfs = PDFProcessor.get_random_pdf(directory, n=n, seed=42)
        print(f"\nSelected {len(random_pdfs)} random PDFs:")
        for pdf in random_pdfs:
            print(f"  - {Path(pdf).name}")
    except Exception as e:
        print(f"Error selecting random PDFs: {e}")
        return
    
    # Initialize processor
    processor = PDFProcessor(
        max_chunk_size=2000,
        chunk_overlap=200,
        debug=True
    )
    
    # Process each PDF
    results = []
    for i, pdf_path in enumerate(random_pdfs, 1):
        print(f"\n{'='*80}")
        print(f"RANDOM TEST {i}/{len(random_pdfs)}: {Path(pdf_path).name}")
        print(f"{'='*80}")
        
        result = processor.process_single_pdf(
            pdf_path,
            extraction_method='pymupdf',
            use_layout=False
        )
        results.append(result)
        
        print(f"\nResults for {Path(pdf_path).name}:")
        print(f"  Success: {result.success}")
        print(f"  Pages: {result.metadata.get('page_count', 'N/A')}")
        print(f"  Chunks: {len(result.chunks)}")
        print(f"  Errors: {len(result.errors)}")
        print(f"  Warnings: {len(result.warnings)}")
        
        if result.errors:
            print("\n  Errors:")
            for error in result.errors:
                print(f"    - {error}")
        
        if result.warnings:
            print("\n  Warnings:")
            for warning in result.warnings[:5]:  # Show first 5
                print(f"    - {warning}")
        
        if result.chunks:
            print("\n  Sample chunk:")
            chunk = result.chunks[0]
            print(f"    Size: {chunk['chunk_size']} chars")
            if 'start_page' in chunk['metadata']:
                print(f"    Pages: {chunk['metadata']['start_page']}-{chunk['metadata']['end_page']}")
            print(f"    Metadata keys: {list(chunk['metadata'].keys())}")
            print(f"    Text preview: {chunk['text'][:100]}...")
    
    # Summary
    print("\n" + "="*80)
    print("RANDOM TESTING SUMMARY")
    print("="*80)
    successful = sum(1 for r in results if r.success)
    total_chunks = sum(len(r.chunks) for r in results)
    print(f"Total PDFs tested: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    print(f"Total chunks created: {total_chunks}")
    if successful > 0:
        print(f"Average chunks per document: {total_chunks / successful:.1f}")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    # Check if user wants random testing
    if len(sys.argv) > 1 and sys.argv[1] == "--random":
        script_dir = Path(__file__).parent.resolve()
        pdf_directory = script_dir / "pdfs"
        
        n_samples = 3
        if len(sys.argv) > 2:
            try:
                n_samples = int(sys.argv[2])
            except ValueError:
                print("Invalid number of samples, using default (3)")
        
        if not pdf_directory.exists():
            print(f"Error: PDF directory not found: {pdf_directory}")
            print("Please create a 'pdfs' folder in the same directory as this script")
            sys.exit(1)
        
        test_random_pdfs(str(pdf_directory), n=n_samples)
    else:
        # Run standard test suite
        main()