import os
import json
import glob
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, DirectoryLoader
from pathlib import Path
from tqdm import tqdm


def detect_category(file_path: str) -> str:
    """Detect category from file path"""
    path_lower = file_path.lower()
    if 'ssa' in path_lower or 'ssn' in path_lower or 'social_security' in path_lower:
        return 'ssa'
    elif 'uscis' in path_lower or 'immigration' in path_lower:
        return 'uscis'
    elif 'travel' in path_lower or 'state' in path_lower or 'passport' in path_lower:
        return 'travel_state'
    return 'general'


def parse_ssa_json(data: Dict, source: str) -> List[Document]:
    """Parse SSA JSON format (sections with heading, paragraphs, list_items)"""
    documents = []
    
    title = data.get('title', 'Untitled')
    url = data.get('url', source)
    
    for section in data.get('sections', []):
        heading = section.get('heading', '')
        paragraphs = section.get('paragraphs', [])
        list_items = section.get('list_items', [])
        link = section.get('link', '')
        
        content_parts = []
        if heading:
            content_parts.append(f"## {heading}")
        if paragraphs:
            content_parts.extend(paragraphs)
        if list_items:
            content_parts.append("\n".join(f"- {item}" for item in list_items))
        
        if content_parts:
            content = "\n\n".join(content_parts)
            metadata = {
                'source': source,
                'title': title,
                'url': url,
                'category': 'ssa',
                'section': heading,
                'link': link
            }
            documents.append(Document(page_content=content, metadata=metadata))
    
    for subpage in data.get('subpages', []):
        sub_title = subpage.get('title', title)
        sub_url = subpage.get('url', url)
        
        for section in subpage.get('sections', []):
            heading = section.get('heading', '')
            paragraphs = section.get('paragraphs', [])
            list_items = section.get('list_items', [])
            
            content_parts = []
            if heading:
                content_parts.append(f"## {heading}")
            if paragraphs:
                content_parts.extend(paragraphs)
            if list_items:
                content_parts.append("\n".join(f"- {item}" for item in list_items))
            
            if content_parts:
                content = "\n\n".join(content_parts)
                metadata = {
                    'source': source,
                    'title': sub_title,
                    'url': sub_url,
                    'category': 'ssa',
                    'section': heading,
                    'subpage': True
                }
                documents.append(Document(page_content=content, metadata=metadata))
    
    return documents


def parse_uscis_json(data: Dict, source: str) -> List[Document]:
    """Parse USCIS JSON format (faq_pages with questions)"""
    documents = []
    
    if 'faq_pages' in data:
        for page in data.get('faq_pages', []):
            title = page.get('title', 'USCIS FAQ')
            url = page.get('url', source)
            last_updated = page.get('last_updated', 'Unknown')
            
            for qa in page.get('questions', []):
                question = qa.get('question', '')
                answer = qa.get('answer', '')
                
                if question and answer:
                    content = f"Q: {question}\n\nA: {answer}"
                    metadata = {
                        'source': source,
                        'title': title,
                        'url': url,
                        'category': 'uscis',
                        'doc_type': 'faq',
                        'last_updated': last_updated,
                        'question': question
                    }
                    documents.append(Document(page_content=content, metadata=metadata))
    
    elif 'sections' in data or 'pages' in data:
        title = data.get('title', 'USCIS Document')
        
        for section in data.get('sections', []):
            heading = section.get('heading', section.get('title', ''))
            content = section.get('content', '')
            
            if content:
                full_content = f"## {heading}\n\n{content}" if heading else content
                metadata = {
                    'source': source,
                    'title': title,
                    'category': 'uscis',
                    'section': heading
                }
                documents.append(Document(page_content=full_content, metadata=metadata))
        
        for page in data.get('pages', []):
            page_title = page.get('title', title)
            content = page.get('content', '')
            
            if content:
                metadata = {
                    'source': source,
                    'title': page_title,
                    'category': 'uscis',
                    'url': page.get('url', '')
                }
                documents.append(Document(page_content=content, metadata=metadata))
    
    return documents


def parse_travel_state_json(data: Dict, source: str) -> List[Document]:
    """Parse Travel.State JSON format (pages array)"""
    documents = []
    
    for page in data.get('pages', []):
        doc_type = page.get('type', 'general')
        title = page.get('title', 'Travel Information')
        url = page.get('url', source)
        content = page.get('content', '')
        
        if content:
            metadata = {
                'source': source,
                'title': title,
                'url': url,
                'category': 'travel_state',
                'doc_type': doc_type,
                'scraped_at': page.get('scraped_at', '')
            }
            documents.append(Document(page_content=content, metadata=metadata))
    
    return documents


def load_json_data(base_path: str = "../data") -> List[Document]:
    """Load and parse all JSON files with automatic format detection"""
    json_path = os.path.join(base_path, "json_data")
    json_files = glob.glob(os.path.join(json_path, "**/*.json"), recursive=True)
    
    all_docs = []
    
    for file_path in tqdm(json_files, desc="Loading JSON files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            category = detect_category(file_path)
            
            if 'faq_pages' in data:
                docs = parse_uscis_json(data, file_path)
            elif 'sections' in data and any('heading' in s for s in data.get('sections', [])):
                docs = parse_ssa_json(data, file_path)
            elif 'pages' in data and isinstance(data['pages'], list):
                docs = parse_travel_state_json(data, file_path)
            else:
                content = json.dumps(data, indent=2)
                docs = [Document(
                    page_content=content,
                    metadata={
                        'source': file_path,
                        'category': category,
                        'format': 'generic_json'
                    }
                )]
            
            all_docs.extend(docs)
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    print(f"Loaded {len(all_docs)} documents from {len(json_files)} JSON files")
    return all_docs


def load_text_data(base_path: str = "../data") -> List[Document]:
    """Load text files"""
    text_path = os.path.join(base_path, "text_data")
    if not os.path.exists(text_path):
        return []
    
    text_files = glob.glob(os.path.join(text_path, "*.txt"))
    docs = []
    
    for file_path in text_files:
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            file_docs = loader.load()
            
            for doc in file_docs:
                doc.metadata['category'] = detect_category(file_path)
            
            docs.extend(file_docs)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    print(f"Loaded {len(docs)} text documents")
    return docs


def load_pdf_data(base_path: str = "../data") -> List[Document]:
    """Load PDF files"""
    pdf_path = os.path.join(base_path, "ssn_pdf")
    if not os.path.exists(pdf_path):
        return []
    
    try:
        loader = DirectoryLoader(
            pdf_path,
            glob="**/*.pdf",
            loader_cls=PyMuPDFLoader,
            show_progress=True,
        )
        pdf_docs = loader.load()
        
        for doc in pdf_docs:
            doc.metadata['category'] = detect_category(doc.metadata.get('source', ''))
        
        print(f"Loaded {len(pdf_docs)} PDF documents")
        return pdf_docs
    except Exception as e:
        print(f"Error loading PDFs: {e}")
        return []


def split_documents_by_category(
    documents: List[Document],
    default_chunk_size: int = 1000,  # OPTIMIZED: Increased from 800
    default_overlap: int = 200       # OPTIMIZED: Increased from 150
) -> List[Document]:
    """
    Split documents with category-aware chunking.
    
    OPTIMIZED CHUNKING STRATEGY:
    
    Why larger chunks?
    - More context per chunk = better semantic understanding
    - Reduces fragmentation of related information
    - Improves LLM comprehension with complete thoughts
    
    Why more overlap?
    - Ensures important information at chunk boundaries isn't lost
    - Improves retrieval recall for cross-boundary queries
    - Better handles questions spanning multiple sentences
    
    Category-specific tuning:
    - SSA: 1000/200 - Moderate chunks for benefits info (paragraphs + lists)
    - USCIS: 1200/250 - Larger chunks for complex immigration procedures
    - Travel/State: 800/150 - Smaller chunks for concise travel advisories
    - General: 1000/200 - Balanced default
    """
    
    category_settings = {
        'ssa': {
            'chunk_size': 1000,     # Was 800 - increased for better context
            'chunk_overlap': 200    # Was 150 - increased to preserve boundaries
        },
        'uscis': {
            'chunk_size': 1200,     # Was 900 - immigration docs are complex, need more context
            'chunk_overlap': 250    # Was 150 - high overlap for procedural continuity
        },
        'travel_state': {
            'chunk_size': 800,      # Was 700 - travel info is concise, keep smaller
            'chunk_overlap': 150    # Was 100 - moderate overlap
        },
        'general': {
            'chunk_size': default_chunk_size, 
            'chunk_overlap': default_overlap
        }
    }
    
    docs_by_category = {}
    for doc in documents:
        category = doc.metadata.get('category', 'general')
        if category not in docs_by_category:
            docs_by_category[category] = []
        docs_by_category[category].append(doc)
    
    all_chunks = []
    for category, docs in docs_by_category.items():
        settings = category_settings.get(category, category_settings['general'])
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings['chunk_size'],
            chunk_overlap=settings['chunk_overlap'],
            length_function=len,
            # Optimized separators - prioritize natural boundaries
            separators=[
                "\n\n\n",    # Multiple newlines (section breaks)
                "\n\n",      # Paragraph breaks
                "\n",        # Line breaks
                ". ",        # Sentences
                "! ",        # Exclamations
                "? ",        # Questions
                "; ",        # Semicolons
                ", ",        # Commas
                " ",         # Words
                ""           # Characters (last resort)
            ],
        )
        
        chunks = splitter.split_documents(docs)
        print(f"✅ {category:15s}: {len(docs):4d} docs → {len(chunks):5d} chunks "
              f"(size={settings['chunk_size']}, overlap={settings['chunk_overlap']})")
        all_chunks.extend(chunks)
    
    return all_chunks


def load_all_data(base_path: str = "../data", split: bool = True) -> List[Document]:
    """
    Load all documents from all sources with intelligent parsing.
    
    OPTIMIZED DATA LOADING PIPELINE:
    1. Load from multiple sources (JSON, TXT, PDF)
    2. Auto-detect and parse format-specific structures
    3. Enrich with metadata (category, source, title, etc.)
    4. Apply category-aware chunking with optimized sizes
    5. Return ready-to-embed document chunks
    """
    
    print("=" * 70)
    print("🚀 LOADING DOCUMENTS FROM ALL SOURCES")
    print("=" * 70)
    
    text_docs = load_text_data(base_path)
    json_docs = load_json_data(base_path)
    pdf_docs = load_pdf_data(base_path)
    
    all_docs = text_docs + json_docs + pdf_docs
    print(f"\n📚 Total documents loaded: {len(all_docs)}")
    
    # Category statistics
    category_counts = {}
    for doc in all_docs:
        cat = doc.metadata.get('category', 'unknown')
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("\n📊 Documents by category:")
    for cat, count in sorted(category_counts.items()):
        print(f"   {cat:15s}: {count:4d} documents")
    
    if split:
        print("\n" + "=" * 70)
        print("✂️  CHUNKING DOCUMENTS (OPTIMIZED SETTINGS)")
        print("=" * 70)
        all_docs = split_documents_by_category(all_docs)
        print(f"\n✅ Total chunks created: {len(all_docs)}")
        print("=" * 70 + "\n")
    
    return all_docs


def get_chunking_config() -> Dict[str, Any]:
    """
    Return current chunking configuration for monitoring/debugging.
    """
    return {
        "strategy": "category-aware recursive splitting",
        "categories": {
            "ssa": {"chunk_size": 1000, "overlap": 200, "rationale": "Balanced for benefits info"},
            "uscis": {"chunk_size": 1200, "overlap": 250, "rationale": "Larger for complex procedures"},
            "travel_state": {"chunk_size": 800, "overlap": 150, "rationale": "Smaller for concise advisories"},
            "general": {"chunk_size": 1000, "overlap": 200, "rationale": "Default balanced setting"}
        },
        "separators": ["\\n\\n\\n", "\\n\\n", "\\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        "optimization_notes": [
            "Larger chunks (1000-1200) provide more context",
            "Higher overlap (200-250) preserves boundary information",
            "Category-specific tuning based on content complexity",
            "Recursive splitting respects natural text boundaries"
        ]
    }