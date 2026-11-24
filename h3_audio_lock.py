import torch

import comfy.audio
import comfy.nested_tensor
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
from comfy_extras.nodes_minimax_h3 import AUDIO_LATENT_FPS, FPS as H3_FPS, _encode_ref_audio


class H3AudioLock(ComfyNodeABC):
    """Holds a song piece in the audio half of a MiniMax H3 AV latent.

    H3 does not keep a reference song: reference audio and guides only steer
    the audio, and the sampler still generates it. This node encodes the piece
    with the audio VAE, writes it into the latent's audio half and lowers the
    audio noise mask over it, so the sampler keeps it.

    The audio latent runs at 40 frames per second, which is 5/3 per pixel frame
    at H3's 24 fps. An AV latent of ``17n + 5`` pixel frames carries
    ``round(frames / 24 * 40)`` audio frames.
    """

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "latent": (IO.LATENT, {"tooltip": "The AV latent the sampler will denoise."}),
                "audio_vae": (IO.VAE, {"tooltip": "The H3 audio VAE."}),
                "audio": (IO.AUDIO, {"tooltip": "This clip's song piece."}),
                "lock_strength": (IO.FLOAT, {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "How firmly the piece is held. 0 leaves the latent unchanged."}),
            },
            "optional": {
                "trim_frames": (IO.INT, {"default": 0, "min": 0, "max": 100000, "tooltip": "Overlap pixel frames from the previous clip. The piece starts after them."}),
            },
        }

    RETURN_TYPES = (IO.LATENT, IO.AUDIO)
    RETURN_NAMES = ("latent", "audio")
    FUNCTION = "lock"
    CATEGORY = "latent"

    def lock(self, latent, audio_vae, audio, lock_strength, trim_frames=0):
        """
        Args:
            latent: An AV latent whose ``samples`` is a NestedTensor of (video, audio).
            audio_vae: The H3 audio VAE.
            audio: The song piece, starting at the clip's first new frame.
            lock_strength: 0 to 1; the audio mask is ``1 - lock_strength`` over the piece.
            trim_frames: Overlap pixel frames before the piece. Their audio stays unlocked.

        Returns:
            The latent with the piece written in and masked, and ``audio`` as given.
        """
        if lock_strength <= 0:
            return (latent, audio)

        video, audio_latent = latent["samples"].tensors
        offset = round(trim_frames / H3_FPS * AUDIO_LATENT_FPS)
        if offset >= audio_latent.shape[-1]:
            return (latent, audio)

        encoded = _encode_piece(audio_vae, audio)
        count = min(audio_latent.shape[-1] - offset, encoded.shape[-1])
        audio_latent = audio_latent.clone()
        audio_latent[..., offset:offset + count] = encoded[..., :count].to(audio_latent)

        audio_mask = torch.ones_like(audio_latent)
        audio_mask[..., offset:offset + count] = 1.0 - lock_strength

        out = latent.copy()
        out["samples"] = comfy.nested_tensor.NestedTensor((video, audio_latent))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((_video_mask(latent, video), audio_mask))
        return (out, audio)


def _video_mask(latent, video):
    mask = latent.get("noise_mask")
    if isinstance(mask, comfy.nested_tensor.NestedTensor):
        return mask.tensors[0]
    if mask is not None:
        return mask
    return torch.ones_like(video)


def _encode_piece(audio_vae, audio):
    waveform = audio["waveform"]
    if waveform.shape[1] == 1:
        waveform = waveform.expand(-1, 2, -1)
    sample_rate = getattr(audio_vae, "audio_sample_rate", 32000)
    waveform = comfy.audio.resample(waveform, audio["sample_rate"], sample_rate)
    encoded, _ = _encode_ref_audio(audio_vae, {"waveform": waveform, "sample_rate": sample_rate})
    return encoded


NODE_CLASS_MAPPINGS: dict[str, type[ComfyNodeABC]] = {
    "H3AudioLock": H3AudioLock,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "H3AudioLock": "H3 Audio Lock",
}
