"""Resolve and snapshot human direction by source file, label, and scene."""

from copy import deepcopy
from dataclasses import replace
from itertools import groupby
from pathlib import Path, PurePosixPath
import tomllib

from .config import ConfigurationError
from .hashing import content_hash, file_hash


class Kantoku:
    def __init__(self, root, config, characters, source_root=None):
        self.root, self.characters = Path(root), characters
        self.files, hashes = {}, {}
        self.source_root, self.indexes = source_root, {}
        for path in sorted((self.root / config.kantoku.directory).rglob('*.toml')):
            with path.open('rb') as stream:
                node = tomllib.load(stream)
            self._validate(node, 0)
            name = node['name']
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts or '\\' in name:
                raise ConfigurationError(f'Unsafe Kantoku source name: {name}')
            if name in self.files:
                raise ConfigurationError(f'Duplicate Kantoku source: {name}')
            self.files[name] = node
            hashes[str(path.relative_to(self.root))] = file_hash(path)
        self.fingerprint = content_hash(hashes)

    def _validate(self, node, depth):
        allowed = {'name', 'prompts', 'characters', 'render'}
        if depth < 2:
            allowed.add('labels' if depth == 0 else 'scenes')
        if depth == 1:
            allowed.add('continuity_from')
        if depth == 2:
            allowed |= {'start_line', 'end_line'}
        if not isinstance(node, dict) or set(node) - allowed or not isinstance(node.get('name'), str) or not node['name']:
            raise ConfigurationError('Kantoku nodes require a name and supported fields only')
        for field in ('prompts', 'characters', 'render'):
            if not isinstance(node.get(field, {}), dict):
                raise ConfigurationError(f'Kantoku {field} must be a table')
        if set(node.get('prompts', {})) - {'background', 'direction'} or any(not isinstance(v, str) for v in node.get('prompts', {}).values()):
            raise ConfigurationError('Kantoku prompts supports background and direction strings')
        for character, values in node.get('characters', {}).items():
            if character not in self.characters or not isinstance(values, dict) or set(values) - {'state', 'acting', 'background'} or any(not isinstance(v, str) for v in values.values()):
                raise ConfigurationError(f'Invalid Kantoku character guidance: {character}')
        for character, values in node.get('render', {}).items():
            if character not in self.characters or not isinstance(values, dict) or set(values) - {'default_voice_profile', 'performance'}:
                raise ConfigurationError(f'Invalid Kantoku render override: {character}')
            if 'default_voice_profile' in values:
                profile = values['default_voice_profile']
                if not isinstance(profile, str) or not (self.root / profile).resolve().is_relative_to((self.root / 'profiles').resolve()) or not (self.root / profile).is_file():
                    raise ConfigurationError(f'Invalid Kantoku voice profile: {profile!r}')
            if 'performance' in values:
                performance = values['performance']
                if not isinstance(performance, dict) or 'cues' in performance:
                    raise ConfigurationError('Kantoku performance cannot override cleaning cues')
                from .annotation.validation import AnnotationValidator
                def invalid(code, message):
                    raise ConfigurationError(f'Kantoku {code}: {message}')
                AnnotationValidator._validate_performance(performance, '', invalid)
        if 'continuity_from' in node and (not isinstance(node['continuity_from'], str) or not node['continuity_from']):
            raise ConfigurationError('Kantoku continuity_from must name a preceding label')
        if depth == 2:
            start, end = node.get('start_line'), node.get('end_line')
            if (start is not None or end is not None) and (type(start) is not int or type(end) is not int or not 0 < start <= end):
                raise ConfigurationError('Kantoku scene ranges require positive start_line <= end_line')
        if depth < 2:
            children = node.get('labels' if depth == 0 else 'scenes', [])
            if not isinstance(children, list):
                raise ConfigurationError('Kantoku children must be arrays of tables')
            names, ranges = set(), []
            for child in children:
                self._validate(child, depth + 1)
                identity = (child['name'], child.get('start_line'))
                if identity in names:
                    raise ConfigurationError(f'Duplicate Kantoku scope: {identity}')
                names.add(identity)
                if 'start_line' in child:
                    span = (child['start_line'], child['end_line'])
                    if any(span[0] <= end and start <= span[1] for start, end in ranges):
                        raise ConfigurationError('Overlapping Kantoku scene ranges')
                    ranges.append(span)

    def resolve(self, record):
        source, label, scene = record['filename'], record.get('label', ''), record.get('scene', '')
        scene_line = 0
        if self.source_root is not None:
            from .galgame.renpy.context import RenPySourceContextIndex
            if source not in self.indexes:
                path = (Path(self.source_root) / source).resolve()
                if not path.is_relative_to(Path(self.source_root).resolve()):
                    raise ConfigurationError('Source path escapes the release')
                self.indexes[source] = RenPySourceContextIndex(path) if path.is_file() else None
            index = self.indexes[source]
            if index is not None:
                scene_line = index.at(record['line_number']).scene_line
        scope = [source, label, scene, scene_line]
        node = self.files.get(source, {})
        label_node = next((n for n in node.get('labels', []) if n['name'] == label), {})
        scenes = label_node.get('scenes', [])
        ranged = [n for n in scenes if 'start_line' in n and n['start_line'] <= record['line_number'] <= n['end_line']]
        scene_node = next(iter(ranged), next((n for n in scenes if 'start_line' not in n and n['name'] == scene), {}))
        if 'start_line' in scene_node:
            scope[2:] = [scene_node['name'], scene_node['start_line']]
        result = {'scope': scope, 'prompts': {}, 'characters': {}, 'render': {}, 'sources': [],
                  'continuity_from': label_node.get('continuity_from')}
        for node in (node, label_node, scene_node):
            if not node:
                continue
            result['sources'].append({'name': node['name'], 'hash': content_hash(node)})
            result['prompts'].update(node.get('prompts', {}))
            for field in ('characters', 'render'):
                for character, values in node.get(field, {}).items():
                    result[field].setdefault(character, {}).update(deepcopy(values))
        return result

    def profile_variants(self):
        def visit(node):
            for character, values in node.get('render', {}).items():
                if 'default_voice_profile' in values:
                    yield replace(self.characters[character], default_voice_profile=values['default_voice_profile'])
            for key in ('labels', 'scenes'):
                for child in node.get(key, []):
                    yield from visit(child)
        for node in self.files.values():
            yield from visit(node)


def bind_direction(request, director):
    """Snapshot all record scopes, including context-only speakers."""
    records = [row for section in ('context_before', 'targets', 'context_interleaved', 'context_after')
               for row in request['batch'].get(section, [])]
    request['direction_context'] = {row['id']: director.resolve(row) for row in records}
    request['kantoku_hash'] = director.fingerprint
    return request


def split_directed_request(request):
    """Split targets at scope changes without losing interleaved dialogue."""
    from .dialogue import DialogueBatch
    batch = DialogueBatch.from_dict(request['batch'])
    directions = request['direction_context']
    groups = [tuple(rows) for _, rows in groupby(batch.targets, key=lambda r: directions[r.id]['scope'])]
    if len(groups) == 1:
        yield request
        return
    ordered = sorted((*batch.context_before, *batch.targets, *batch.context_interleaved, *batch.context_after),
                     key=lambda r: (r.line_number, r.id))
    for targets in groups:
        ids = {r.id for r in targets}
        before = tuple(r for r in ordered if r.line_number < targets[0].line_number and r.id not in ids)
        after = tuple(r for r in ordered if r.line_number > targets[-1].line_number and r.id not in ids)
        interleaved = tuple(r for r in ordered if targets[0].line_number <= r.line_number <= targets[-1].line_number and r.id not in ids)
        child = replace(batch, id=content_hash({'parent': batch.id, 'targets': sorted(ids)})[:24],
                        targets=targets, context_before=before, context_after=after, context_interleaved=interleaved)
        result = {**request, 'batch': child.to_dict()}
        for key in ('director_notes', 'cleaned_annotations', 'original_texts'):
            if key in request:
                result[key] = {key_id: value for key_id, value in request[key].items() if key_id in ids}
        yield result
