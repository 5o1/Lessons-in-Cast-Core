"""Closed API wire schemas and lossless normalization to core annotations."""

from copy import deepcopy
import json
import math


def object_schema(properties):
    return {'type': 'object', 'additionalProperties': False,
            'required': list(properties), 'properties': properties}


def wire_schema(request):
    annotation = deepcopy(request['response_schema']['properties']['annotations']['items'])
    properties = annotation['properties']
    if 'delivery' in properties:
        properties['delivery'] = {'type': 'array', 'items': object_schema({
            'key': {'type': 'string'}, 'value': {'type': 'string'}})}
    if 'keyframe_effects' in properties:
        properties['keyframe_effects']['type'] = ['array', 'null']
    schema = object_schema({
        'turn_id': {'type': 'string'},
        'annotations': {'type': 'array', 'items': annotation},
        'memory': {'type': 'array', 'items': object_schema({
            'character_id': {'type': 'string'},
            'kind': {'type': 'string', 'enum': (['fact', 'unresolved'] if request['stage'] == 'cleaning'
                                              else ['fact', 'acting_hypothesis', 'unresolved'])},
            'text': {'type': 'string'},
            'evidence_ids': {'type': 'array', 'items': {'type': 'string'}}})},
    })

    def close(node):
        # Provider-independent structural subset; core validation retains numeric constraints.
        for key in ('$schema', 'title', 'uniqueItems', 'minLength', 'maxLength', 'minimum',
                    'maximum', 'exclusiveMinimum', 'maxItems', 'minItems', 'pattern'):
            node.pop(key, None)
        if 'const' in node:
            node['enum'] = [node.pop('const')]
        if 'enum' in node and 'type' not in node:
            node['type'] = ['string', 'null'] if None in node['enum'] else 'string'
        if node.get('enum') == []:
            # No allowed effects: empty arrays remain valid; any returned item is rejected locally.
            node.pop('enum')
        if 'properties' in node:
            node['additionalProperties'] = False
            node['required'] = list(node['properties'])
            for child in node['properties'].values():
                close(child)
        if 'items' in node:
            close(node['items'])
    close(schema)
    return schema


def check_shape(value, schema, location='$'):
    """Validate the closed structural subset emitted by wire_schema (not arbitrary JSON Schema)."""
    kind = ('null' if value is None else 'boolean' if type(value) is bool else
            'integer' if type(value) is int else 'number' if type(value) is float else
            'string' if isinstance(value, str) else 'array' if isinstance(value, list) else
            'object' if isinstance(value, dict) else 'invalid')
    allowed = schema['type'] if isinstance(schema['type'], list) else [schema['type']]
    if kind not in allowed and not (kind == 'integer' and 'number' in allowed):
        raise ValueError(f'{location}: expected {allowed}, got {kind}')
    if kind == 'number' and not math.isfinite(value):
        raise ValueError(f'{location}: non-finite number')
    if 'enum' in schema and value not in schema['enum']:
        raise ValueError(f'{location}: invalid enum value')
    if kind == 'object':
        if set(value) != set(schema['properties']):
            raise ValueError(f'{location}: missing or extra fields; expected {list(schema["properties"])}')
        for key, item in value.items():
            check_shape(item, schema['properties'][key], f'{location}.{key}')
    if kind == 'array':
        for index, item in enumerate(value):
            check_shape(item, schema['items'], f'{location}[{index}]')


def normalize_output(output, schema, turn_id, aliases):
    check_shape(output, schema)
    if output['turn_id'] != turn_id:
        raise ValueError(f'$.turn_id: expected {turn_id!r}, got {output["turn_id"]!r}; copy the expected value exactly')
    annotations = deepcopy(output['annotations'])
    if [row['id'] for row in annotations] != list(aliases):
        raise ValueError(f'Return exactly these target IDs in order: {list(aliases)}')
    for row in annotations:
        row['id'] = aliases[row['id']]
        if 'delivery' in row:
            delivery = {entry['key']: entry['value'] for entry in row['delivery']}
            if len(delivery) != len(row['delivery']):
                raise ValueError('Duplicate delivery keys')
            row['delivery'] = delivery
        if row.get('keyframe_effects', False) is None:
            del row['keyframe_effects']
    return annotations


def estimate_tokens(value):
    # ponytail: conservative UTF-8 byte budget; use a model tokenizer if this wastes material context.
    # This is an estimate, not a claim about every compatible provider's tokenizer.
    return len(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
