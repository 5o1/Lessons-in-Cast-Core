# Kantoku direction

Add TOML files here to provide human background and acting guidance before annotation. Both Codex and API execution receive the prepared guidance. A file's `name` matches the complete release-relative source path, not its basename.

Example (fictional direction; replace the source, label, and line range with real values):

```toml
name = "game/route_a.rpy"

[prompts]
background = "Background needed to understand these exchanges."

[characters.a]
state = "Concealing anxiety."
acting = "Distinguish her private feelings from what she wants the listener to hear."

[[labels]]
name = "example_reconciliation"

[labels.prompts]
background = "The characters are trying to reconnect after an argument."

[labels.characters.a]
acting = "Begin restrained; do not resolve the tension before the dialogue does."

[[labels.scenes]]
name = "example_room"
start_line = 120
end_line = 180

[labels.scenes.characters.a]
acting = "Trying to sound cheerful while beginning to falter."

[labels.scenes.render.a.performance]
speed = 0.95
```

`prompts` supports `background` and `direction`. Each character supports `background`, `state`, and `acting`. Cleaning receives background and state for interpretation, without producing acting labels; polish receives full direction.

Merge file → label → scene. Matching prompt fields replace inherited values. Character fields merge by character ID, then by field. An empty string clears inherited guidance. A new scene is resolved from its parents rather than the preceding scene's merged result. Kantoku takes precedence over matching upstream audition notes.

A scene without a range matches its lexical scene name within the label. A range is inclusive and defines a story scene even if visual backgrounds change within it. Ranged scenes take precedence over name-only scenes. Repeated lexical backgrounds are distinguished by their source-statement line; ranges provide explicit control over their narrative meaning. Overlapping ranges are rejected.

By default, API memory resets across files and labels. An optional `continuity_from = "previous_label"` on a label permits memory to carry only when the immediately preceding processed label has that name in the same file. It does not traverse jumps or retrieve a remote label. Scene changes discard scene-local memory while retaining character entries. Source ordering is not a reconstructed runtime route.

`render.<character_id>` supports `default_voice_profile` and `performance`. A profile must be an existing entrypoint under `profiles/`. Performance uses existing portable fields such as speed, energy, pitch, and vocal mode; cues cannot be overridden. Production planning applies these values after upstream character and model defaults, without having the LLM regenerate them. Audition candidate profiles remain explicitly selected by the audition command; its rendering workflow is separate from production routing.

Preparing a run snapshots the effective direction, scope, and source hashes into its annotation requests. Polish and retries inherit that snapshot. Changing these TOML files requires a fresh run before API execution or polish validation. Empty directories have no effect.
