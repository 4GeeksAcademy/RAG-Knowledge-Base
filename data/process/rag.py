"""Chunking and indexing for the HealthCore knowledge base (Qdrant)."""

import os

from dotenv import load_dotenv
from openai import OpenAI

# Read LLM_API_KEY, LLM_API_URL, EMBEDDING_MODEL ... from .env
load_dotenv()

# Embedding model: must be different from the generation model (LLM_MODEL)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# One OpenAI client, reused for every embed() call
openai_client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_API_URL", "https://api.openai.com/v1"),
)

# File name in docs/company-knowledge-base/ -> source_document ID (BRIEFING section 3).
SOURCE_DOCUMENTS = {
    "healthcore-insurance-coverage.en.md": "insurance-coverage",
    "healthcore-appointment-policy.en.md": "appointment-policy",
    "healthcore-referral-process.en.md": "referral-process",
    "healthcore-new-patient-checklist.en.md": "new-patient-checklist",
}

# source_document ID -> section label of each chunk, in document order.
# Unlabeled blocks in the source get a descriptive name.
SECTION_LABELS = {
    "insurance-coverage": [
        "United States (Texas, Florida, Georgia)",
        "United Kingdom (London and Manchester)",
        "Unlisted insurers",
    ],
    "appointment-policy": [
        "Booking appointments",
        "Cancellation policy",
        "Automated reminders",
        "Repeated no-shows",
    ],
    "referral-process": [
        "Standard referral process",
        "Target completed-referral time",
        "Escalation",
        "Out-of-network referrals",
    ],
    "new-patient-checklist": [
        "Before the first appointment",
        "Documents the patient should bring on the day of the appointment",
        "Incomplete medical history form",
    ],
}

# Split the document into individual blocks.
def split_blocks(text: str) -> list[str]:
    """Split a document into blocks separated by blank lines, without the '# ' title."""
    chunks = []

    # Step 1: Split document by blank lines
    blocks = text.split("\n\n")

    # Step 2: Process each block
    for block in blocks:
        cleaned_block = block.strip()

        if cleaned_block and not cleaned_block.startswith("# "):
            chunks.append(cleaned_block)

    return chunks

# Merge introductions with their corresponding content.
def merge_intro_blocks(blocks: list[str]) -> list[str]:
    """Join a block ending in ':' with the block after it (intro + its list)."""
    final_chunks = [] #stores both combined and unchanged blocks.
    pending = "" #stores introductions waiting to be combined.

    for block in blocks:
        # An intro sentence ends with ':' -> hold it until its list arrives
        if block.endswith(":"):
            pending = f"{pending}\n\n{block}" if pending else block
            continue

        # Put the held intro in front of the list it introduces
        if pending:
            block = f"{pending}\n\n{block}"
            pending = ""

        # Add blocks that do not end with ':'
        final_chunks.append(block)

    # Safety: keep any remaining intro blocks
    if pending:
        final_chunks.append(pending)

    return final_chunks

# It takes a knowledge-base file's name and text, splits the text into chunks, and assigns metadata to each chunk.
def chunk_document(filename: str, text: str) -> list[dict]:
    """Turn one knowledge-base file into labeled chunks."""
    source_document = SOURCE_DOCUMENTS[filename]
    labels = SECTION_LABELS[source_document]
    final_chunks = []

    blocks = merge_intro_blocks(split_blocks(text))

    # Fail loudly if the document changed and the labels no longer line up
    if len(blocks) != len(labels):
        raise ValueError(
            f"{filename}: got {len(blocks)} chunks but {len(labels)} section labels"
        )

    for index, (block, label) in enumerate(zip(blocks, labels)):

        chunk = {
            "source_document": source_document,
            "section": label,
            "chunk_index": index,
            "text": block,
        }

        final_chunks.append(chunk)

    return final_chunks



# Convert text into a vector. Used for both indexing and queries.
def embed(text: str) -> list[float]:
    """Return the embedding vector of a text, using the embedding model."""
    response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding
