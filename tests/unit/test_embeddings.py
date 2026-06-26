from core.embeddings.chunker import chunk_python, chunk_generic


SAMPLE_PYTHON = '''
def add(a, b):
    return a + b

class Calculator:
    def multiply(self, a, b):
        return a * b

    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
'''


def test_chunk_python_finds_functions():
    chunks = chunk_python(SAMPLE_PYTHON)
    types = [c.chunk_type for c in chunks]
    assert "function" in types or "class" in types


def test_chunk_python_line_numbers():
    chunks = chunk_python(SAMPLE_PYTHON)
    for c in chunks:
        assert c.start_line >= 1
        assert c.end_line >= c.start_line


def test_chunk_python_content_nonempty():
    chunks = chunk_python(SAMPLE_PYTHON)
    for c in chunks:
        assert len(c.content.strip()) > 0


def test_chunk_generic_sliding_window():
    source = "\n".join([f"line {i}" for i in range(100)])
    chunks = chunk_generic(source, "javascript", window=20, overlap=5)
    assert len(chunks) > 1
    for c in chunks:
        assert c.language == "javascript"


def test_chunk_invalid_python_falls_back():
    broken = "def foo(\n  this is not valid python"
    chunks = chunk_python(broken)
    assert len(chunks) >= 1
