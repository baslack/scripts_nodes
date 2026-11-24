"""H3 Audio Lock contract.

The song piece is written into the audio half of the AV latent from the clip's
first new frame, and the audio noise mask is lowered over it. A fake audio VAE
gives each encoded frame its own value, so the tests can see where it landed.
"""
import pytest
import torch

import comfy.nested_tensor
from comfy.ldm.minimax.model import FRAME_PER_TOKEN
from comfy_extras.nodes_minimax_h3 import temporal_shape

import scripts_nodes
from scripts_nodes.h3_audio_lock import H3AudioLock

VAE_RATE = 32000
AUDIO_FRAMES = 200  # a 5 s clip at 40 audio frames per second


class FakeAudioVAE:
    audio_sample_rate = VAE_RATE

    def __init__(self):
        self.seen = []

    def encode(self, waveform):
        """[1, L, C] -> [1, 32, 2, T]; frame t holds the value t + 1."""
        self.seen.append(waveform)
        frames = round(waveform.shape[1] / VAE_RATE * 40)
        values = torch.arange(1, frames + 1, dtype=torch.float32)
        return values.expand(1, 32, 2, frames).clone()


def _latent(video_mask=None):
    video = torch.zeros(1, 24, 3, 2, 2)
    audio = torch.zeros(1, 32, 2, AUDIO_FRAMES)
    latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
    if video_mask is not None:
        latent["noise_mask"] = video_mask
    return latent


def _piece(seconds, channels=2, rate=VAE_RATE):
    return {"waveform": torch.randn(1, channels, int(seconds * rate)), "sample_rate": rate}


def _lock(latent, audio, vae=None, strength=1.0):
    return H3AudioLock().lock(latent, vae or FakeAudioVAE(), audio, strength)


def _streams(latent):
    return latent["samples"].tensors[1], latent["noise_mask"].tensors[1]


def test_window_is_written_and_locked_from_the_first_audio_frame():
    out, _ = _lock(_latent(), _piece(2))
    audio, mask = _streams(out)
    assert audio[0, 0, 0, 0] == 1
    assert (mask[..., :80] == 0).all()



def test_long_window_is_cropped_to_the_clip():
    out, _ = _lock(_latent(), _piece(8))
    audio, mask = _streams(out)
    assert audio.shape[-1] == AUDIO_FRAMES
    assert audio[0, 0, 0, -1] == AUDIO_FRAMES
    assert (mask == 0).all()


def test_short_window_leaves_the_tail_unlocked():
    out, _ = _lock(_latent(), _piece(1))
    audio, mask = _streams(out)
    assert (mask[..., :40] == 0).all()
    assert (mask[..., 40:] == 1).all()
    assert (audio[..., 40:] == 0).all()


def test_mono_becomes_stereo():
    vae = FakeAudioVAE()
    _lock(_latent(), _piece(1, channels=1), vae)
    assert vae.seen[0].shape[-1] == 2


def test_other_sample_rate_is_resampled_to_the_vae_rate():
    vae = FakeAudioVAE()
    _lock(_latent(), _piece(1, rate=16000), vae)
    assert vae.seen[0].shape[1] == VAE_RATE


@pytest.mark.parametrize("strength", [1.0, 0.75, 0.25])
def test_audio_mask_is_one_minus_strength_over_the_piece(strength):
    out, _ = _lock(_latent(), _piece(1), strength=strength)
    _, mask = _streams(out)
    assert torch.allclose(mask[..., :40], torch.full_like(mask[..., :40], 1 - strength))
    assert (mask[..., 40:] == 1).all()


def test_strength_zero_returns_the_latent_unchanged():
    latent = _latent()
    out, _ = _lock(latent, _piece(1), strength=0.0)
    assert out is latent


def test_video_mask_defaults_to_ones():
    out, _ = _lock(_latent(), _piece(1))
    assert (out["noise_mask"].tensors[0] == 1).all()


def test_existing_video_mask_is_kept():
    video_mask = torch.zeros(1, 1, 3, 2, 2)
    video_mask[..., 0, :, :] = 0.5
    out, _ = _lock(_latent(video_mask), _piece(1))
    assert torch.equal(out["noise_mask"].tensors[0], video_mask)

    nested = comfy.nested_tensor.NestedTensor((video_mask, torch.zeros(1, 32, 2, AUDIO_FRAMES)))
    out, _ = _lock(_latent(nested), _piece(1))
    assert torch.equal(out["noise_mask"].tensors[0], video_mask)


def test_input_latent_and_audio_are_not_modified():
    latent = _latent()
    audio = _piece(1)
    out, audio_out = _lock(latent, audio)
    assert (latent["samples"].tensors[1] == 0).all()
    assert "noise_mask" not in latent
    assert audio_out is audio


@pytest.mark.parametrize("length", [5, 22, 39, 73, 121, 241, 481])
def test_audio_latent_runs_at_40_per_second_over_the_clips_pixel_frames(length):
    frame_count, latent_t, audio_t = temporal_shape(length)
    pixel_frames = sum(FRAME_PER_TOKEN[k % 5] for k in range(latent_t))
    assert pixel_frames == frame_count
    assert abs(audio_t - pixel_frames / 24 * 40) <= 0.5


def test_node_is_registered():
    assert scripts_nodes.NODE_CLASS_MAPPINGS["H3AudioLock"] is H3AudioLock
    assert scripts_nodes.NODE_DISPLAY_NAME_MAPPINGS["H3AudioLock"] == "H3 Audio Lock"
