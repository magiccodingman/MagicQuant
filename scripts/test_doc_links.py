"""Protect active documentation links; archival documents retain their historical context."""
from pathlib import Path
import re
import unittest
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent


class DocumentationTests(unittest.TestCase):
    def test_local_documentation_targets_exist(self):
        files = [ROOT / name for name in ['README.md', 'CONTRIBUTING.md', 'THIRD-PARTY-NOTICES.md']]
        files += list((ROOT / 'docs').rglob('*.md')) + list((ROOT / 'wiki').rglob('*.md'))
        errors = []
        for path in files:
            for match in re.finditer(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
                url = match.group(1).split(' "')[0].strip('<>')
                if url.startswith(('https:', 'http:', 'mailto:', '#')):
                    continue
                target = unquote(url.split('#')[0])
                if target and not (path.parent / target).exists():
                    errors.append(f'{path.relative_to(ROOT)}: {url}')
        self.assertEqual([], errors)

    def test_current_docs_use_canonical_repository_url(self):
        files = [ROOT / 'README.md', *list((ROOT / 'docs').rglob('*.md')), *list((ROOT / 'wiki').rglob('*.md'))]
        for path in files:
            self.assertNotIn('github.com/magiccodingman/magicquant-wiki', path.read_text(encoding='utf-8').lower(), str(path))


if __name__ == '__main__':
    unittest.main()
