"""Read-only, content-addressed Markdown evidence. No vector service is required."""
from copy import deepcopy
from pathlib import Path
import re
from molsteer.common import digest, file_hash
from molsteer.molthinker.knowledge import KnowledgeBase, tokenize


class MarkdownCorpus:
    name = 'local'

    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        self.chunks = {}
        files = []
        for path in sorted(self.root.rglob('*.md')):
            if not path.resolve().is_relative_to(self.root):
                continue
            sha = file_hash(path)
            files.append((path.relative_to(self.root).as_posix(), sha))
            lines = path.read_text(encoding='utf-8-sig').splitlines()
            reviewed = {}
            if path.name == 'Molecular_Generation_Control_Functions_Representative_Table_2026-09-19.md':
                reviewed = {x['source']['line']: x for x in KnowledgeBase(path).entries}
            section, start, block = '', 1, []

            def add(begin, content, entry=None):
                excerpt = '\n'.join(content).strip()
                if not excerpt:
                    return
                relative = path.relative_to(self.root).as_posix()
                item = dict(source_key='local:' + relative + ':' + sha,
                            title=section or relative, file=relative, sha256=sha,
                            line_start=begin, line_end=begin+len(content)-1, excerpt=excerpt,
                            retrieval_scope='inspected local source passage')
                if entry:
                    item.update({k: entry[k] for k in ('function_id', 'formula', 'variables',
                                'gradient_target', 'prerequisites', 'sources', 'tags', 'name_en')})
                item['source_id'] = 'src_' + digest(item['source_key'])[:20]
                item['chunk_id'] = 'chunk_' + digest(item)[:24]
                self.chunks[item['chunk_id']] = item

            for number, line in enumerate(lines, 1):
                if line.startswith('#') or line.startswith('|') or not line.strip() or len(block) >= 24:
                    add(start, block)
                    block = []
                    start = number
                if line.startswith('#'):
                    section = line.lstrip('# ').strip()
                if line.startswith('|'):
                    if number in reviewed:
                        add(number, [line], reviewed[number])
                    elif not re.match(r'^\|[\s:|\-]+$', line):
                        add(number, [line])
                    start = number+1
                else:
                    if not block:
                        start = number
                    block.append(line)
            add(start, block)
        self.version = digest(files)

    def search(self, query, limit=5):
        if not isinstance(query, str) or not query.strip() or not 1 <= limit <= 10:
            raise ValueError('Nonempty query and bounded result count required')
        tokens = set(tokenize(query))
        ranked = []
        for item in self.chunks.values():
            content = ' '.join(str(item.get(k, '')) for k in
                               ('title', 'excerpt', 'tags', 'name_en', 'variables'))
            score = len(tokens & set(tokenize(content)))
            if score:
                ranked.append(dict(item, retrieval_score=score))
        ranked.sort(key=lambda x: (-x['retrieval_score'], x['chunk_id']))
        return {'status': 'ok' if ranked else 'zero_hits', 'query': query,
                'corpus_hash': self.version, 'records': deepcopy(ranked[:limit])}

    def fetch(self, source_id):
        if source_id not in self.chunks:
            raise ValueError('Unknown local chunk ID')
        item = self.chunks[source_id]
        path = (self.root/item['file']).resolve()
        if not path.is_relative_to(self.root) or file_hash(path) != item['sha256']:
            raise ValueError('Local evidence changed; rebuild corpus')
        return deepcopy(item)
