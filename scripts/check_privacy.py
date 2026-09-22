"""Conservative pre-upload audit; reports locations, never matched values."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "venv", "env", "__pycache__"}
SPECIAL = {".gitignore", ".gitattributes", "requirements.txt"}
PATTERNS = {
    "personal directory": r"(?:[/]home[/]|[/]Users[/]|[A-Za-z]:[\\/]Users[\\/])[^\s/\\]+",
    "email address": r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
    "private key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "access token": r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16})",
    "credential assignment": r'''(?i)(?:api[_-]?key|access[_-]?token|password|secret)\s*[=:]\s*["'][^"'\s]{8,}["']''',
    "phone number": r"(?<![\d.])(?:\+86[- ]?)?1[3-9]\d{9}(?![\d.])",
}


def main():
    findings = []
    checked = 0

    def walk(directory):
        for path in sorted(directory.iterdir()):
            relative = path.relative_to(ROOT)
            if path.is_symlink():
                findings.append((relative, 0, "symbolic link requires review"))
            elif path.is_dir():
                if path.name not in SKIP_DIRS:
                    yield from walk(path)
            else:
                yield path

    for path in walk(ROOT):
        relative = path.relative_to(ROOT)
        allowed = (path.name in SPECIAL or path.suffix in {".py", ".md"}
                   or relative.as_posix() == "examples/sphere-impact.ply")
        if not allowed:
            findings.append((relative, 0, "unexpected file requires review"))
            continue
        if path.stat().st_size > 2_000_000:
            findings.append((relative, 0, "large file requires review"))
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeError:
            findings.append((relative, 0, "non-UTF-8 content"))
            continue
        if "\x00" in content:
            findings.append((relative, 0, "binary content"))
            continue
        checked += 1
        for line_number, line in enumerate(content.splitlines(), 1):
            for label, pattern in PATTERNS.items():
                if re.search(pattern, line):
                    findings.append((relative, line_number, label))
    for relative, line_number, label in findings:
        print(f"REVIEW {relative}:{line_number}: {label}")
    print(f"Checked {checked} text files; {len(findings)} findings.")
    if findings:
        raise SystemExit(1)
    print("No pattern matches. Review staged files and commit identity before upload.")


if __name__ == "__main__":
    main()
