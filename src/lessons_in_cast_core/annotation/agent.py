"""Resumable Chat Completions annotation with bounded story memory."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
from itertools import groupby
import json
from contextlib import ExitStack
import os
import re
from http.client import HTTPException
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, HTTPRedirectHandler

from ..dialogue import DialogueBatch
from ..hashing import content_hash, file_hash
from ..jsonl import read_jsonl, write_jsonl
from ..keyframes import extract_anchors
from ..locking import exclusive_lock
from ..speech_markup import SpeechSegment, emotion_markup, parse_emotion_markup
from .context import ContextOverflow, build_messages, scope_for, transition_memory
from .protocol import estimate_tokens, normalize_output
from .validation import AnnotationValidator
from .types import ValidationStatus


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('Annotation API redirects are disabled; configure the final endpoint')


def chat_completion(settings, messages, schema):
    key = os.environ.get(settings.api_key_environment)
    if not key:
        raise ValueError(f'Set {settings.api_key_environment} before running annotation')
    payload = {'model': settings.model, 'messages': messages,
               settings.completion_token_parameter: settings.max_completion_tokens,
               'response_format': ({'type': 'json_schema', 'json_schema': {
                   'name': 'dialogue_turn', 'strict': True, 'schema': schema}}
                   if settings.response_format == 'json_schema' else {'type': 'json_object'})}
    if settings.reasoning_split is not None:
        payload['reasoning_split'] = settings.reasoning_split
    if settings.response_format == 'json_object':
        payload['messages'] = [*messages, {'role': 'user', 'content': 'Required JSON schema: ' + json.dumps(schema)}]
    request = Request(settings.base_url.rstrip('/') + '/chat/completions',
                      data=json.dumps(payload, ensure_ascii=False).encode(), method='POST',
                      headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'})
    for attempt in range(3):
        try:
            with build_opener(NoRedirect()).open(request, timeout=settings.timeout_seconds) as response:
                raw = response.read(16 * 1024 * 1024 + 1)
                if len(raw) > 16 * 1024 * 1024:
                    raise ValueError('Annotation API response exceeds 16 MiB')
                return json.loads(raw)
        except HTTPError as exc:
            # Never echo response bodies: gateways can include credentials or request data.
            body = exc.read(65536).lower()
            if exc.code in (400, 413) and any(word in body for word in
                    (b'context_length', b'context window', b'too many tokens', b'maximum context')):
                raise ContextOverflow('Provider rejected the context size') from None
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise ValueError(f'Annotation API HTTP {exc.code}; check endpoint, model and protocol settings') from None
        except (URLError, TimeoutError, ConnectionError, HTTPException, json.JSONDecodeError) as exc:
            if attempt == 2:
                raise ValueError(f'Annotation API transport failed after three attempts ({type(exc).__name__})') from None
        time.sleep(2 ** attempt)


def available_voices(root, config, characters, request, targets):
    from ..model_registry import load_model_registry
    from ..synthesis.profiles import load_voice_profile
    if request['stage'] != 'polish':
        return {}
    result = {}
    for row in targets:
        character_id = row['character'] or 'narrator'
        character = characters[character_id].resolve(row['filename'], row.get('label', ''), row.get('scene', ''))
        direction = request.get('direction_context', {}).get(row['id'], {})
        routes = []
        for member in character.synthesis_members:
            definition = characters[member].resolve(row['filename'], row.get('label', ''), row.get('scene', ''))
            entrypoint = direction.get('render', {}).get(member, {}).get('default_voice_profile', definition.default_voice_profile)
            if not entrypoint:
                routes.append(set())
                continue
            profile = load_voice_profile(root, Path(entrypoint), definition, config,
                                         model_registry=load_model_registry(repository_root=root))
            try:
                try:
                    routes.append(set(profile.list_voice_tags()))
                except NotImplementedError:
                    routes.append(set())
            finally:
                profile.close()
        tags = sorted(set.intersection(*routes)) if routes else []
        if character_id in result and result[character_id] != tags:
            raise ValueError('Voice assets change inside a turn; split at the profile boundary')
        result[character_id] = tags
    return result


class AnnotationAgent:
    def __init__(self, root, config, characters, stage, *, completion=chat_completion, voice_lookup=available_voices):
        self.root, self.config, self.characters, self.stage = Path(root), config, characters, stage
        if stage not in ('cleaning', 'polish'):
            raise ValueError('Unknown annotation stage')
        self.settings = getattr(config, stage)
        self.completion, self.voice_lookup = completion, voice_lookup
        self.prompt_path = self.root / (config.codex.prompt_path if stage == 'cleaning' else config.codex.polish_prompt_path)
        self.prompt = self.prompt_path.read_text(encoding='utf-8')
        self.validator = AnnotationValidator(config.annotation)

    def run(self, layout, *, retry=False, preview=False, source_files=()):
        directory = layout.root / 'api' / ('retry' if retry else 'initial')
        directory.mkdir(parents=True, exist_ok=True)
        with ExitStack() as stack:
            try:
                stack.enter_context(exclusive_lock(directory / '.lock'))
            except BlockingIOError:
                raise ValueError('Another annotation process is already using this stage') from None
            return self._run(layout, retry=retry, preview=preview, source_files=source_files)

    def _run(self, layout, *, retry=False, preview=False, source_files=()):
        requests_path = layout.retry_requests if retry else layout.annotation_requests
        responses_path = layout.retry_responses if retry else layout.annotation_responses
        directory = layout.root / 'api' / ('retry' if retry else 'initial')
        self._failure_directory = directory / 'failed-attempts'
        requests = list(read_jsonl(requests_path))
        if any(r.get('stage') != self.stage for r in requests):
            raise ValueError('Request stage does not match configured agent')
        from ..kantoku import Kantoku
        director = Kantoku(self.root, self.config, self.characters)
        if any(r.get('kantoku_hash', director.fingerprint) != director.fingerprint for r in requests):
            raise ValueError('Kantoku changed; prepare a fresh run')
        pool = {}
        for request in requests:
            for section in ('context_before', 'targets', 'context_interleaved', 'context_after'):
                for row in request['batch'].get(section, []):
                    pool.setdefault(row['id'], row)
        if layout.raw_dialogue.is_file():
            pool.update({r['id']: r for r in read_jsonl(layout.raw_dialogue)})
        records = sorted(pool.values(), key=lambda r: (r['filename'], r['line_number'], r['id']))
        validation_config = asdict(self.config.annotation)
        for name in ('allowed_emotions', 'allowed_effects'):
            validation_config[name] = sorted(validation_config[name])
        settings = {'adapter': 'chat-completions-agent', 'version': 5,
                    'stage': self.stage, 'settings': asdict(self.settings),
                    'prompt_sha256': file_hash(self.prompt_path), 'requests_sha256': file_hash(requests_path),
                    'source_hash': content_hash(records), 'characters_hash': content_hash({k: asdict(v) for k, v in self.characters.items()}),
                    'source_files': list(source_files)}
        settings['validation_hash'] = content_hash(validation_config)
        if retry:
            settings['initial_checkpoints'] = {path.name: file_hash(path)
                for path in sorted((layout.root / 'api/initial').glob('turn-*.jsonl'))}
        manifest = directory / 'run.jsonl'
        if manifest.exists():
            if list(read_jsonl(manifest)) != [settings]:
                raise ValueError('API run inputs or settings changed; prepare a fresh run')
        elif not preview:
            if responses_path.exists():
                raise ValueError('Responses already exist without API checkpoints; use a fresh run')
            write_jsonl([settings], manifest)
        index = 0
        memory, scope, evidence, review_ids = [], None, set(), set()
        envelopes = []
        for request in requests:
            if source_files and request['batch']['targets'][0]['filename'] not in source_files:
                continue
            if request.get('independent_context'):
                memory, scope, evidence, review_ids = [], None, set(), set()
            batch = DialogueBatch.from_dict(request['batch'])
            output = []
            groups = [list(rows) for _, rows in groupby(request['batch']['targets'], key=lambda row:
                      (scope_for(request, row)[:2], request.get('direction_context', {}).get(row['id'], {}).get('render', {})))]
            labels = [scope_for(request, rows[0])[:2] for rows in groups]
            turn_target_limit = max(1, self.settings.api.max_completion_tokens // 320)
            if (not retry and not preview and self.settings.api.parallel_workers > 1
                    and len(groups) > 1 and len(set(map(tuple, labels))) == len(labels)
                    and all(len(rows) <= turn_target_limit for rows in groups)
                    and not any(request.get('direction_context', {}).get(rows[0]['id'], {}).get('continuity_from')
                                for rows in groups)):
                existing_checkpoints = set(directory.glob('turn-*.jsonl'))
                try:
                    commits = self._parallel_groups(request, groups, records, memory, scope, evidence,
                                                    review_ids, settings, directory, index)
                except ContextOverflow:
                    # Fall through to the serial path, which persists an adaptive split.
                    for checkpoint in set(directory.glob('turn-*.jsonl')) - existing_checkpoints:
                        checkpoint.unlink()
                else:
                    output = [annotation for committed in commits for annotation in committed['annotations']]
                    committed = commits[-1]
                    memory, scope = committed['memory_after'], committed['scope']
                    evidence, review_ids = set(committed['evidence_ids']), set(committed['review_ids'])
                    index += len(commits)
                    envelopes.append({'request_hash': content_hash(request), 'batch_id': batch.id,
                                      'prompt_version': request['prompt_version'], 'annotator_configuration': settings,
                                      'generated_at': committed['generated_at'],
                                      'response': {'batch_id': batch.id, 'annotations': output}})
                    write_jsonl(envelopes, responses_path)
                    continue
            pending = deque(groups)
            while pending:
                targets = pending.popleft()
                checkpoint = directory / f'turn-{index:06d}.jsonl'
                if not checkpoint.exists() and len(targets) > turn_target_limit:
                    pending.appendleft(targets[turn_target_limit:])
                    targets = targets[:turn_target_limit]
                current_scope = scope_for(request, targets[0])
                direction = request.get('direction_context', {}).get(targets[0]['id'], {})
                before_memory = transition_memory(memory, scope, current_scope, direction)
                if scope is None or current_scope[:2] != scope[:2]:
                    evidence = {identity for entry in before_memory for identity in entry['evidence_ids']}
                    review_ids &= evidence
                if retry:
                    # Focused retries never inherit memory from later dialogue or another retry.
                    before_memory = []
                    for initial_checkpoint in sorted((layout.root / 'api/initial').glob('turn-*.jsonl')):
                        old = next(read_jsonl(initial_checkpoint))
                        if old.get('commit_hash') != content_hash({k: v for k, v in old.items() if k != 'commit_hash'}):
                            raise ValueError('Initial API checkpoint was modified; use a fresh run')
                        if targets[0]['id'] in old['target_ids']:
                            before_memory = old['memory_before']
                            break
                    evidence = {identity for entry in before_memory for identity in entry['evidence_ids']}
                    review_ids = set()
                voices = self.voice_lookup(self.root, self.config, self.characters, request, targets)
                turn_id = content_hash({'request': content_hash(request), 'targets': [r['id'] for r in targets],
                                        'memory': before_memory, 'voices': voices, 'settings': settings})[:24]
                if checkpoint.exists():
                    committed = next(read_jsonl(checkpoint))
                    if committed.get('commit_hash') != content_hash({k: v for k, v in committed.items() if k != 'commit_hash'}):
                        raise ValueError('API checkpoint was modified; use a fresh run')
                    # Adaptive splits are replayed from committed target IDs, without another model call.
                    committed_ids = committed['target_ids']
                    if not committed_ids:
                        raise ValueError('API checkpoint contains no targets')
                    if committed_ids != [r['id'] for r in targets]:
                        if committed_ids != [r['id'] for r in targets[:len(committed_ids)]]:
                            raise ValueError('API checkpoint target sequence changed')
                        pending.appendleft(targets[len(committed_ids):])
                        targets = targets[:len(committed_ids)]
                        pending.appendleft(targets)
                        continue
                    if committed['turn_id'] != turn_id:
                        raise ValueError('API checkpoint context or voice assets changed; use a fresh run')
                else:
                    split_path = directory / f'split-{turn_id}.jsonl'
                    if split_path.exists():
                        if list(read_jsonl(split_path)) != [{'target_ids': [r['id'] for r in targets]}] or len(targets) < 2:
                            raise ValueError('Invalid saved API split; use a fresh run')
                        middle = len(targets) // 2
                        pending.appendleft(targets[middle:])
                        pending.appendleft(targets[:middle])
                        continue
                    try:
                        messages, aliases, schema, trims, visible_evidence = build_messages(
                            request, targets, records, before_memory, self.prompt,
                            {k: v.name for k, v in self.characters.items()}, voices, self.settings, turn_id)
                        if preview:
                            path = directory / 'preview.jsonl'
                            write_jsonl([{'messages': messages, 'response_format_schema': schema,
                                          'target_ids': aliases, 'trims': trims,
                                          'input_token_estimate': estimate_tokens({'messages': messages, 'schema': schema})}], path)
                            return {'backend': 'api', 'preview': str(path)}
                        committed = self._execute(request, targets, messages, aliases, schema, turn_id,
                                                  before_memory, evidence | visible_evidence, review_ids, voices)
                        committed.update(trims=trims, scope=current_scope, target_ids=[r['id'] for r in targets])
                        committed['commit_hash'] = content_hash(committed)
                        write_jsonl([committed], checkpoint)
                        (self._failure_directory / f'{turn_id}.jsonl').unlink(missing_ok=True)
                    except ContextOverflow as exc:
                        if len(targets) == 1:
                            write_jsonl([{'turn_id': turn_id, 'target_ids': [r['id'] for r in targets],
                                          'error': str(exc), 'memory_before': before_memory}], directory / 'failure.jsonl')
                            raise
                        middle = len(targets) // 2
                        write_jsonl([{'target_ids': [r['id'] for r in targets]}], split_path)
                        pending.appendleft(targets[middle:])
                        pending.appendleft(targets[:middle])
                        continue
                    except ValueError as exc:
                        write_jsonl([{'turn_id': turn_id, 'target_ids': [r['id'] for r in targets],
                                      'error': str(exc), 'memory_before': before_memory}], directory / 'failure.jsonl')
                        raise
                memory, scope = committed['memory_after'], current_scope
                evidence.update(committed['evidence_ids'])
                review_ids.update(committed['review_ids'])
                output.extend(committed['annotations'])
                index += 1
            envelopes.append({'request_hash': content_hash(request), 'batch_id': batch.id,
                              'prompt_version': request['prompt_version'], 'annotator_configuration': settings,
                              'generated_at': committed['generated_at'],
                              'response': {'batch_id': batch.id, 'annotations': output}})
            if not preview:
                # ponytail: rewrite the derived ledger per batch; use an indexed store if this becomes material.
                write_jsonl(envelopes, responses_path)
        if not preview:
            if not envelopes:
                write_jsonl([], responses_path)
            (directory / 'failure.jsonl').unlink(missing_ok=True)
        return {'backend': 'api', 'completed_batches': len(envelopes), 'completed_turns': index}

    def _parallel_groups(self, request, groups, records, memory, scope, evidence, review_ids,
                         settings, directory, index):
        prepared = []
        prior_scope = scope
        for offset, targets in enumerate(groups):
            current_scope = scope_for(request, targets[0])
            direction = request.get('direction_context', {}).get(targets[0]['id'], {})
            before_memory = transition_memory(memory if offset == 0 else [], prior_scope, current_scope, direction)
            current_evidence = set(evidence) if offset == 0 and current_scope[:2] == (prior_scope or [])[:2] else {
                identity for entry in before_memory for identity in entry['evidence_ids']}
            current_reviews = set(review_ids) & current_evidence
            voices = self.voice_lookup(self.root, self.config, self.characters, request, targets)
            turn_id = content_hash({'request': content_hash(request), 'targets': [r['id'] for r in targets],
                                    'memory': before_memory, 'voices': voices, 'settings': settings})[:24]
            checkpoint = directory / f'turn-{index + offset:06d}.jsonl'
            if checkpoint.exists():
                committed = next(read_jsonl(checkpoint))
                if (committed.get('commit_hash') != content_hash({k: v for k, v in committed.items() if k != 'commit_hash'})
                        or committed['turn_id'] != turn_id or committed['target_ids'] != [r['id'] for r in targets]):
                    raise ValueError('API checkpoint context changed; use a fresh run')
                prepared.append((None, committed))
            else:
                messages, aliases, schema, trims, visible = build_messages(
                    request, targets, records, before_memory, self.prompt,
                    {k: v.name for k, v in self.characters.items()}, voices, self.settings, turn_id)
                prepared.append(((request, targets, messages, aliases, schema, turn_id, before_memory,
                                  current_evidence | visible, current_reviews, voices, trims, current_scope,
                                  checkpoint), None))
            prior_scope = current_scope

        def execute(values):
            (child_request, targets, messages, aliases, schema, turn_id, before_memory,
             visible_evidence, current_reviews, voices, trims, current_scope, checkpoint) = values
            committed = self._execute(child_request, targets, messages, aliases, schema, turn_id,
                                      before_memory, visible_evidence, current_reviews, voices)
            committed.update(trims=trims, scope=current_scope, target_ids=[r['id'] for r in targets])
            committed['commit_hash'] = content_hash(committed)
            write_jsonl([committed], checkpoint)
            (self._failure_directory / f'{turn_id}.jsonl').unlink(missing_ok=True)
            return committed

        missing = [values for values, committed in prepared if committed is None]
        if not missing:
            return [committed for _, committed in prepared]
        # ponytail: independent label turns only; add dependency scheduling if continuity chains need parallelism.
        with ThreadPoolExecutor(max_workers=min(self.settings.api.parallel_workers, len(missing))) as pool:
            completed = iter(pool.map(execute, missing))
            return [next(completed) if committed is None else committed for _, committed in prepared]

    def _execute(self, request, targets, messages, aliases, schema, turn_id, memory_before, evidence, review_ids, voices):
        attempts = []
        correction = []
        for attempt in range(self.settings.api.max_repair_attempts + 1):
            actual_messages = messages + correction
            response = self.completion(self.settings.api, actual_messages, schema)
            attempts.append({'messages': actual_messages, 'response': response})
            write_jsonl(attempts, self._failure_directory / f'{turn_id}.jsonl')
            failed_content = None
            try:
                choice = response['choices'][0]
                if choice['message'].get('refusal'):
                    raise ValueError('Model refused this turn; human review is required')
                if choice.get('finish_reason') == 'length':
                    raise ContextOverflow('Output was truncated; split targets or increase completion budget')
                if choice.get('finish_reason') != 'stop':
                    raise ValueError('Model did not finish a structured response')
                failed_content = choice['message']['content']
                if not isinstance(failed_content, str):
                    raise ValueError('message.content must be a JSON string')
                content = failed_content.strip()
                fenced = re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```', content, re.DOTALL)
                document = json.loads(fenced[1] if fenced else content)
                annotations = normalize_output(document, schema, turn_id, aliases)
                if self.stage == 'cleaning':
                    originals = {target['id']: target['dialogue'] for target in targets}
                    for annotation in annotations:
                        annotation['spoken_text'] = re.sub(r'\{/?(?:i|b|u|s)\}', '', annotation['spoken_text'])
                        original = originals.get(annotation['id'], '')
                        heard = original.lstrip('. …')
                        if (heard and len(heard) < len(original)
                                and annotation['spoken_text'] == annotation['spoken_text'].lstrip('. …')
                                and len(annotation['spoken_text']) > len(heard)
                                and annotation['spoken_text'].casefold().endswith(heard.casefold())):
                            annotation['action'] = 'speak'
                            annotation['effects'] = []
                            annotation['performance']['cues'] = []
                        cues = annotation['performance']['cues']
                        annotation['performance']['cues'] = list({cue['offset']: cue for cue in cues
                            if cue['duration_seconds'] is not None and cue['duration_seconds'] > 0
                            and 0 <= cue['offset'] <= len(annotation['spoken_text'])
                            and not (0 < cue['offset'] < len(annotation['spoken_text'])
                                     and annotation['spoken_text'][cue['offset'] - 1].isalpha()
                                     and annotation['spoken_text'][cue['offset']].isalpha())}.values())
                        annotation['performance']['cues'].sort(key=lambda cue: cue['offset'])
                else:
                    allowed = self.config.annotation.allowed_emotions
                    for annotation in annotations:
                        if annotation['action'] not in ('speak', 'speak_with_effect'):
                            annotation['spoken_text'] = ''
                            annotation.pop('keyframe_effects', None)
                            continue
                        if ('<' not in annotation['spoken_text'] and '>' not in annotation['spoken_text']
                                and annotation['spoken_text'].strip()):
                            annotation['spoken_text'] = emotion_markup((
                                SpeechSegment(annotation['spoken_text'], 'neutral'),))
                        try:
                            spans = parse_emotion_markup(annotation['spoken_text'])
                        except ValueError:
                            continue
                        annotation['spoken_text'] = emotion_markup(tuple(
                            replace(span, emotion=None,
                                    arbitrary_emotion=span.emotion.replace('_', ' '))
                            if span.emotion is not None and span.emotion not in allowed else span
                            for span in spans))
                        effects = annotation.get('keyframe_effects')
                        if (annotation['id'] in request.get('keyframe_required', ())
                                and isinstance(effects, list) and len(effects) == 1
                                and isinstance(effects[0], dict)):
                            points = effects[0].get('keyframes')
                            if (isinstance(points, list) and len(points) == 2
                                    and all(isinstance(point, dict) and isinstance(point.get('anchor'), str)
                                            and re.fullmatch(r'[1-9][0-9]*', point['anchor']) for point in points)
                                    and points[0]['anchor'] != points[1]['anchor']):
                                spans = [replace(span, text=extract_anchors(span.text)[0])
                                         for span in parse_emotion_markup(annotation['spoken_text'])]
                                spans[0] = replace(spans[0], text='{' + points[0]['anchor'] + '}' + spans[0].text)
                                spans[-1] = replace(spans[-1], text=spans[-1].text + '{' + points[1]['anchor'] + '}')
                                annotation['spoken_text'] = emotion_markup(tuple(spans))
                child = {**request['batch'], 'targets': targets}
                result = self.validator.validate_batch(DialogueBatch.from_dict(child),
                    {'batch_id': child['batch_id'], 'annotations': annotations},
                    prompt_version=request['prompt_version'], annotator_configuration={},
                    schema_version=request['schema_version'], stage=self.stage,
                    cleaned_annotations=request.get('cleaned_annotations'),
                    keyframe_required=request.get('keyframe_required', ()))
                issues = [issue for row in result.records for issue in row.issues
                          if issue.severity == 'error' or issue.code == 'length_ratio']
                issues.extend(issue for issue in result.issues if issue.severity == 'error')
                if issues or any(row.status in (ValidationStatus.RETRYABLE, ValidationStatus.REJECTED) for row in result.records):
                    raise ValueError('; '.join(f'{issue.dialogue_id}: {issue.code}: {issue.message}' for issue in issues))
                if self.stage == 'polish':
                    for raw, target in zip(annotations, targets):
                        if raw['action'] not in ('speak', 'speak_with_effect'):
                            continue
                        for span in parse_emotion_markup(raw['spoken_text']):
                            if span.voice and span.voice not in voices.get(target['character'] or 'narrator', []):
                                raise ValueError(f'{raw["id"]}: unknown voice {span.voice}')
                current_review = {row.dialogue_id for row in result.records if row.status == ValidationStatus.REVIEW_REQUIRED}
                memory = []
                for entry in document['memory']:
                    cited = set(entry['evidence_ids'])
                    if (entry['character_id'] and entry['character_id'] not in self.characters
                            or not entry['text'].strip() or not cited or not cited <= evidence
                            or self.stage == 'cleaning' and entry['kind'] == 'acting_hypothesis'):
                        continue
                    if entry['kind'] != 'unresolved' and cited & (review_ids | current_review):
                        entry = {**entry, 'kind': 'unresolved'}
                    memory.append(entry)
                while estimate_tokens(memory) > self.settings.context.memory_tokens and memory:
                    memory.pop(0)
                return {'turn_id': turn_id, 'annotations': annotations, 'memory_before': memory_before,
                        'generated_at': datetime.now(timezone.utc).isoformat(),
                        'memory_after': memory, 'attempts': attempts, 'evidence_ids': sorted(evidence),
                        'review_ids': sorted(current_review)}
            except ContextOverflow:
                raise
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                error = str(exc)
                attempts[-1]['validation_error'] = error
                write_jsonl(attempts, self._failure_directory / f'{turn_id}.jsonl')
                if attempt == self.settings.api.max_repair_attempts and len(targets) > 1 and 'refused' not in error:
                    raise ContextOverflow('Repeated validation failure; split this turn') from None
                if attempt == self.settings.api.max_repair_attempts or 'refused' in error:
                    raise ValueError(f'Annotation turn {turn_id} failed validation: {error}') from None
                correction = ([{'role': 'assistant', 'content': failed_content}]
                              if isinstance(failed_content, str) and failed_content else [])
                feedback = {'role': 'user', 'content': 'Repair the failed response above. '
                    'Keep valid annotations and immutable cleaning unchanged. '
                    'Return a complete replacement JSON object, without Markdown. Validation errors: ' + error[:6000]}
                correction.append(feedback)
                if estimate_tokens({'messages': messages + correction, 'schema': schema}) + self.settings.api.max_completion_tokens + 1024 > self.settings.api.context_window_tokens:
                    correction = [{**feedback, 'content': 'The previous response failed validation. Regenerate the '
                                   'complete JSON object from the original request. ' + feedback['content']}]
                    if estimate_tokens({'messages': messages + correction, 'schema': schema}) + self.settings.api.max_completion_tokens + 1024 > self.settings.api.context_window_tokens:
                        raise ContextOverflow('Validation feedback exceeds context budget; split this turn')
