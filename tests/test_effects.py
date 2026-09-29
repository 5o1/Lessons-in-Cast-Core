from array import array
from dataclasses import replace
import json
import math
from pathlib import Path
import pickle
import shutil
import tempfile
import unittest
import zlib

from lessons_in_cast_core.annotation import DialogueAction
from lessons_in_cast_core.config import AudioConfig
from lessons_in_cast_core.effects import CoreEffectProcessor, EffectError, EffectLibrary, EffectSpec, load_effect_library
from lessons_in_cast_core.effects.pcm import Pcm, read_pcm, write_pcm
from lessons_in_cast_core.hashing import file_hash
from lessons_in_cast_core.synthesis.audio import WaveRenderer, AudioRenderError, AudioQualityChecker
from lessons_in_cast_core.synthesis.types import RenderTask, TtsJob
from lessons_in_cast_core.galgame.renpy.archive import RenPyArchiveWriter


class EffectsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source.wav"
        self.pcm = Pcm(array("h", (round(9000 * math.sin(2 * math.pi * 240 * i / 24000)) for i in range(48000))), 24000, 1)
        write_pcm(self.source, self.pcm)

    def processor(self, kind, **parameters):
        return CoreEffectProcessor(EffectLibrary({"test": EffectSpec(kind, parameters)}))

    def render(self, processor, names=("test",), name="out.wav", source=True):
        path = processor.process(self.source if source else None, self.root / name, names)
        return read_pcm(path), json.loads(path.with_suffix(".effects.json").read_text())

    def test_six_defaults_and_ordered_custom_chains(self):
        library = EffectLibrary()
        self.assertEqual(set(library.presets), {"fade_in", "fade_out", "telephone", "monster", "censor_beep", "glitch"})
        config = self.root / "effects.toml"
        config.write_text('[presets.soft]\ntype="fade_out"\nduration_seconds=0.8\n[chains]\nentity=["monster","soft","soft"]\n')
        library = load_effect_library(config)
        self.assertEqual([s.kind for s in library.resolve(("entity",))], ["monster", "fade_out", "fade_out"])
        self.assertEqual(library.resolve(("soft",))[0].resolved()["duration_seconds"], 0.8)

    def test_invalid_config_fails_before_rendering(self):
        for text in ('[chains]\na=["b"]\nb=["a"]', '[chains]\na=["missing"]', '[presets.glitch]\nrepeats=1.2',
                     '[presets.fade_in]\nduration_seconds=nan', '[presets.monster]\nsemitones=-30',
                     '[presets.telephone]\nunknown=42', '[chains]\nmonster=["fade_in"]'):
            with self.subTest(text=text):
                config = self.root / "bad.toml"
                config.write_text(text)
                with self.assertRaises(EffectError):
                    load_effect_library(config)
        with self.assertRaises(EffectError):
            load_effect_library(self.root / "missing.toml")

    def test_fades_preserve_duration_and_other_samples(self):
        for kind in ("fade_in", "fade_out"):
            with self.subTest(kind=kind):
                pcm, audit = self.render(self.processor(kind, duration_seconds=0.2), name=f"{kind}.wav")
                self.assertEqual(pcm.frames, self.pcm.frames)
                if kind == "fade_in":
                    self.assertEqual(pcm.samples[4800:], self.pcm.samples[4800:])
                    self.assertEqual(pcm.samples[0], 0)
                else:
                    self.assertEqual(pcm.samples[:-4800], self.pcm.samples[:-4800])
                    self.assertEqual(pcm.samples[-1], 0)
                self.assertEqual(audit["steps"][0]["effective_seconds"], 0.2)

    def test_long_fade_clamps_to_clip_and_stereo_is_preserved(self):
        self.pcm.samples = array("h", [1000, -1000]) * 48000
        self.pcm.channels = 2
        write_pcm(self.source, self.pcm)
        pcm, audit = self.render(self.processor("fade_out", duration_seconds=5))
        self.assertEqual(pcm.channels, 2)
        self.assertEqual(pcm.samples[-2:], array("h", [0, 0]))
        self.assertEqual(pcm.samples[100], -pcm.samples[101])
        self.assertEqual(audit["steps"][0]["effective_seconds"], 2)

    def test_beep_replaces_interval_instead_of_mixing_speech(self):
        processor = self.processor("censor_beep", start_seconds=0.5, end_seconds=1.0)
        pcm, audit = self.render(processor)
        first = pcm.samples[12000:24000]
        self.assertEqual(pcm.samples[:12000], self.pcm.samples[:12000])
        self.assertEqual(pcm.samples[24000:], self.pcm.samples[24000:])
        self.pcm.samples = array("h", [16000]) * 48000
        write_pcm(self.source, self.pcm)
        second, _ = self.render(processor, name="second.wav")
        self.assertEqual(first, second.samples[12000:24000])
        self.assertEqual(audit["steps"][0]["interval_seconds"], [0.5, 1.0])

    def test_effect_only_beep_and_subsequent_fade(self):
        pcm, audit = self.render(CoreEffectProcessor(), ("censor_beep", "fade_out"), source=False)
        self.assertEqual(pcm.seconds, 1)
        self.assertEqual(pcm.rate, 48000)
        self.assertIsNone(audit["source_sha256"])
        self.assertGreater(max(pcm.samples), 0)
        with self.assertRaises(EffectError):
            self.render(CoreEffectProcessor(), ("monster",), name="bad.wav", source=False)

    def test_unknown_effect_and_invalid_interval_leave_existing_output_untouched(self):
        output = self.root / "out.wav"
        output.write_bytes(b"existing")
        with self.assertRaises(EffectError):
            CoreEffectProcessor().process(self.source, output, ("unknown",))
        self.assertEqual(output.read_bytes(), b"existing")
        with self.assertRaises(EffectError):
            self.render(self.processor("censor_beep", start_seconds=20))
        self.assertEqual(output.read_bytes(), b"existing")
        with self.assertRaises(EffectError):
            CoreEffectProcessor().process(self.source, self.source, ("fade_out",))

    def test_glitch_is_deterministic_extends_duration_and_recovers_original_tail(self):
        processor = CoreEffectProcessor()
        before = file_hash(self.source)
        pcm, audit = self.render(processor, ("glitch",))
        second, other = self.render(processor, ("glitch",), name="second.wav")
        self.assertEqual(pcm.samples, second.samples)
        self.assertEqual(audit["output_sha256"], other["output_sha256"])
        self.assertEqual(file_hash(self.source), before)
        step = audit["steps"][0]
        self.assertAlmostEqual(step["inserted_seconds"], 4 * (0.16 + 0.045) + 0.8 + 0.55, places=4)
        self.assertEqual(pcm.samples[-12000:], self.pcm.samples[-12000:])
        recovery = round(step["recovery_output_seconds"] * pcm.rate)
        on = round(0.065 * pcm.rate)
        dropout_start = recovery - round(0.55 * pcm.rate)
        self.assertEqual(set(pcm.samples[dropout_start + on:dropout_start + on + 100]), {0})

    def test_glitch_rejects_short_input_and_unsafe_expansion(self):
        with self.assertRaises(EffectError):
            self.render(self.processor("glitch", position=0.99))
        with self.assertRaises(EffectError):
            self.render(self.processor("glitch", hold_seconds=120))

    def test_order_changes_audio_and_audit(self):
        processor = CoreEffectProcessor()
        first, audit = self.render(processor, ("fade_in", "censor_beep"))
        second, _ = self.render(processor, ("censor_beep", "fade_in"), name="second.wav")
        self.assertNotEqual(first.samples, second.samples)
        self.assertEqual([s["type"] for s in audit["steps"]], ["fade_in", "censor_beep"])

    def test_renderer_supports_sfx_only_and_retains_audit(self):
        task = RenderTask("dialogue", "identifier", DialogueAction.SFX_ONLY, (), "single", ("censor_beep",), "voice/test.wav", "voice/test.wav")
        output = WaveRenderer().render(task, {}, self.root)
        audit = json.loads(output.with_suffix(".effects.json").read_text())
        self.assertEqual(audit["dialogue_id"], "dialogue")
        self.assertEqual(audit["delivery"]["sha256"], file_hash(output))
        with self.assertRaises(AudioRenderError):
            WaveRenderer().render(replace(task, effects=("monster",)), {}, self.root)

    def test_missing_ffmpeg_is_an_explicit_error(self):
        with self.assertRaisesRegex(EffectError, "FFmpeg execution failed"):
            self.render(CoreEffectProcessor(ffmpeg_executable=str(self.root / "not-ffmpeg")), ("telephone",))


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg and ffprobe required for DSP/Opus integration")
class FfmpegEffectsTests(unittest.TestCase):
    setUp = EffectsTests.setUp
    render = EffectsTests.render

    def test_telephone_bandlimiting_and_monster_pitch(self):
        phone, audit = self.render(CoreEffectProcessor(), ("telephone",))
        self.assertEqual(phone.rate, self.pcm.rate)
        self.assertAlmostEqual(phone.seconds, self.pcm.seconds, delta=0.03)
        self.assertIn("ffmpeg_version", audit["steps"][0])
        monster, audit = self.render(CoreEffectProcessor(), ("monster",), name="monster.wav")
        samples = monster.samples[6000:36000]
        crossings = sum(a <= 0 < b for a, b in zip(samples, samples[1:]))
        frequency = crossings / (len(samples) / monster.rate)
        self.assertAlmostEqual(frequency, 240 * 2 ** (-6/12), delta=8)
        self.assertAlmostEqual(monster.seconds, self.pcm.seconds, delta=0.1)
        self.assertLessEqual(max(abs(sample) for sample in monster.samples), 32767)

    def test_abstract_entity_chain_opus_rpa_delivery(self):
        job = TtsJob("job", "dialogue", "a", "Test.", "neutral", {}, "source.wav", "key")
        task = RenderTask("dialogue", "identifier", DialogueAction.SPEAK_WITH_EFFECT, ("job",), "single",
                          ("monster", "glitch", "fade_out"), "voice/chapter/test.opus", "voice/chapter/test.opus")
        config = AudioConfig(format="opus", sample_rate=48000)
        output = WaveRenderer(audio_config=config).render(task, {"job": job}, self.root)
        self.assertTrue(AudioQualityChecker(config).check(task.dialogue_id, output).valid)
        audit = json.loads(output.with_suffix(".effects.json").read_text())
        self.assertEqual([s["type"] for s in audit["steps"]], list(task.effects))
        archive = self.root / "voice.rpa"
        RenPyArchiveWriter().write(archive, [(task.virtual_path, output)])
        with archive.open("rb") as stream:
            header = stream.readline().decode().split()
            stream.seek(int(header[1], 16))
            index = pickle.loads(zlib.decompress(stream.read()))
            self.assertEqual(list(index), [task.virtual_path])
            offset, size = index[task.virtual_path][0]
            key = int(header[2], 16)
            stream.seek(offset ^ key)
            payload = stream.read(size ^ key)
            self.assertIn(b"OpusHead", payload)
            self.assertNotIn(".wav", next(iter(index)))
