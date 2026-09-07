from pathlib import Path
import re

if __package__:
    from .db import get_connection
    from .embeddings import generate_embedding, model
else:
    from db import get_connection
    from embeddings import generate_embedding, model

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def split_sections(text):
    """Return section topics and bodies, skipping the table of contents."""
    headings = list(re.finditer(r'^SECTION\s+\d+:[ \t]*([^\n]+)', text, re.MULTILINE))
    if not headings:
        return [(None, text)]
    sections = []
    for i, heading in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        sections.append((heading.group(1).strip(), text[heading.end():end]))
    return sections


def chunk_text(text: str, chunk_size: int = 200) -> list[str]:
    """Pack paragraphs, splitting oversized paragraphs by sentence then word."""
    tokenizer = model.tokenizer
    maximum = model.max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)
    if not 0 < chunk_size <= maximum:
        raise ValueError(f'chunk_size must be between 1 and {maximum}')

    def fits(value):
        # Measure the entire paragraph before splitting; it is not model input yet.
        return len(tokenizer.encode(value, add_special_tokens=False,
                                    truncation=False, verbose=False)) <= chunk_size

    def split_long(value):
        parts = []
        for sentence in re.split(r'(?<=[.!?])\s+', value):
            while sentence and not fits(sentence):
                offsets = tokenizer(sentence, add_special_tokens=False,
                                    truncation=False, verbose=False,
                                    return_offsets_mapping=True)['offset_mapping']
                end = offsets[chunk_size - 1][1]
                boundary = max(sentence.rfind(' ', 0, end), sentence.rfind('\n', 0, end))
                if boundary > 0:
                    end = boundary
                while end > 0 and not fits(sentence[:end]):
                    end -= 1
                if not end:
                    raise ValueError('Token budget too small to split text')
                parts.append(sentence[:end].strip())
                sentence = sentence[end:].strip()
            if sentence:
                parts.append(sentence)
        return parts

    text = re.sub(r'^[ \t]*[=-]{3,}[ \t]*$', '', text, flags=re.MULTILINE)
    paragraphs = [p.strip() for p in re.split(r'\n[ \t]*\n+', text) if p.strip()]
    chunks, current = [], ''
    for paragraph in paragraphs:
        for part in ([paragraph] if fits(paragraph) else split_long(paragraph)):
            candidate = current + '\n\n' + part if current else part
            if current and not fits(candidate):
                chunks.append(current)
                current = part
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks


def ingest_text_file(file_path: str | Path, document_id: int = 1):
    """Replace this document's chunks in one transaction; failures roll back."""
    path = Path(file_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    chunks = [(topic, chunk)
              for topic, body in split_sections(path.read_text(encoding='utf-8'))
              for chunk in chunk_text(body)]
    if not chunks:
        raise ValueError('Document contains no content to ingest')
    if any(topic and len(topic) > 100 for topic, _ in chunks):
        raise ValueError('Section topic exceeds the database 100-character limit')
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('DELETE FROM document_chunks WHERE document_id = %s', (document_id,))
            for index, (topic, chunk) in enumerate(chunks):
                cur.execute(
                    '''INSERT INTO document_chunks
                       (document_id, content, topic, source, chunk_index, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s)''',
                    (document_id, chunk, topic, path.name, index, generate_embedding(chunk)),
                )
    print(f'Successfully ingested {len(chunks)} chunks from {path.name}.')


if __name__ == '__main__':
    ingest_text_file('data/dsa.txt', document_id=1)
