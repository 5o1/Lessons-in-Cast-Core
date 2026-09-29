"""Start a workspace-owned loopback worker; rendering uses normal profile APIs."""

import argparse
import os
from pathlib import Path
import subprocess
from urllib.parse import urlsplit

from ....config import find_repository_root
from ....model_registry import load_model_registry
from .client import save_json
from ....paths import venv_python
from .config import COMFY_REVISION, load_config
from .pipeline import verify_installation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Run an isolated ComfyUI worker in the foreground")
    serve.add_argument("--profile-config", type=Path, required=True)
    args = parser.parse_args(argv)
    root = (args.root or find_repository_root()).resolve()
    config = load_config(root / args.profile_config)
    source = (root / config.source_directory).resolve()
    if not source.is_relative_to(root):
        raise ValueError("H3 source must be inside the workspace")
    registry = load_model_registry(repository_root=root)
    models = registry.resolve_path(config.model_id)
    manifest = verify_installation(source, models)
    python = venv_python(source / ".venv")
    if not python.is_file():
        raise FileNotFoundError(f"Install the isolated ComfyUI environment first: {python}")
    endpoint = urlsplit(config.endpoint)
    runtime = root / "build/backends/minimax-h3" / str(endpoint.port)
    for directory in ("input", "output", "temp", "user", "cache"):
        (runtime / directory).mkdir(parents=True, exist_ok=True)
    # JSON is a YAML subset and quotes model paths without YAML interpolation.
    model_paths = runtime / "model-paths.yaml"
    save_json(model_paths, {"lessons_in_cast_h3": {"base_path": str(models),
               "diffusion_models": "diffusion_models", "text_encoders": "text_encoders", "vae": "vae"}})
    command = [str(python), str(source / "main.py"), "--listen", endpoint.hostname, "--port", str(endpoint.port),
               "--extra-model-paths-config", str(model_paths), "--disable-all-custom-nodes", "--disable-api-nodes",
               "--disable-auto-launch", "--preview-method", "none", "--disable-metadata"]
    for kind in ("input", "output", "temp", "user"):
        command.extend([f"--{kind}-directory", str(runtime / kind)])
    environment = os.environ.copy()
    environment.update({"PYTHONUNBUFFERED": "1", "HF_HOME": str(root / "build/cache/huggingface"),
                        "TORCH_HOME": str(runtime / "cache/torch"), "XDG_CACHE_HOME": str(runtime / "cache"),
                        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    freeze = subprocess.check_output([str(python), "-c",
        "import importlib.metadata as m; print('\\n'.join(sorted(f'{d.metadata[\"Name\"]}=={d.version}' for d in m.distributions())))"], text=True, encoding="utf-8")
    (runtime / "environment.txt").write_text(freeze, encoding="utf-8")
    save_json(runtime / "worker.json", {"endpoint": config.endpoint, "source_revision": COMFY_REVISION,
              "model": registry.require(config.model_id).to_dict(), "verified": manifest, "command": command})
    print(f"Local H3 worker: {config.endpoint}\nArtifacts: {runtime}", flush=True)
    return subprocess.call(command, cwd=source, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
