import ast
from pathlib import Path


def _extract_supported_formats(path: Path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "SUPPORTED_FORMATS":
                    if isinstance(node.value, (ast.Set, ast.Tuple, ast.List)):
                        return {
                            elt.value
                            for elt in node.value.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                        }
    raise AssertionError(f"SUPPORTED_FORMATS not found in {path}")


def test_supported_audio_formats_include_m4a():
    for path in [
        Path("app.py"),
        Path("transcriber.py"),
        Path("transcriber_seg.py"),
        Path("ui_app.py"),
    ]:
        formats = _extract_supported_formats(path)
        assert ".m4a" in formats, f"{path} should accept M4A audio files"
