"""Native ComfyUI graph: minimal video latent, audio VAE decode only."""

from pathlib import PurePosixPath

from .config import COMPONENTS


def build_workflow(adaptation, reference_names, *, output_prefix):
    p = adaptation.parameters
    if not 1 <= len(reference_names) <= 3 or len(reference_names) != len(p["references"]):
        raise ValueError("Every H3 reference must be uploaded exactly once")
    diffusion, text, audio_vae, video_vae = (PurePosixPath(name).name for name in COMPONENTS)
    def node(kind, **inputs):
        return {"class_type": kind, "inputs": inputs}
    graph = {
        "92": node("SaveAudio", audio=["121", 0], filename_prefix=output_prefix),
        "119": node("VAELoader", vae_name=video_vae),
        "120": node("VAELoader", vae_name=audio_vae),
        "121": node("VAEDecodeAudio", samples=["125", 0], vae=["120", 0]),
        "123": node("KSamplerSelect", sampler_name="res_multistep"),
        "124": node("BasicScheduler", model=["127", 0], scheduler="simple", steps=p["steps"], denoise=1.),
        "125": node("SamplerCustomAdvanced", noise=["129", 0], guider=["126", 0], sampler=["123", 0],
                    sigmas=["124", 0], latent_image=["136", 1]),
        "126": node("BasicGuider", model=["127", 0], conditioning=["136", 0]),
        "127": node("UNETLoader", unet_name=diffusion, weight_dtype="default"),
        "128": node("CLIPLoader", clip_name=text, type="minimax", device="default"),
        "129": node("RandomNoise", noise_seed=p["seed"]),
        "136": node("MiniMaxH3ReferenceToVideo", clip=["128", 0], vae=["119", 0], audio_vae=["120", 0],
                    prompt=p["prompt"], width=32, height=32, length=p["length"], ref_image_size="match"),
    }
    for index, name in enumerate(reference_names):
        key = str(200 + index)
        graph[key] = node("LoadAudio", audio=name)
        graph["136"]["inputs"][f"ref_audios.ref_audio_{index}"] = [key, 0]
    return graph
