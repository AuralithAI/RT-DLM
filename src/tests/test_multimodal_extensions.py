"""Tests for multimodal extensions: STFT audio, spatiotemporal video, document,
point cloud, biosignal, tactile, action, spectrogram decoder, VQ-VAE, streaming
buffer, RGBD vision, sync loss and timestamp embeddings."""

import math
import unittest

import jax
import jax.numpy as jnp
import haiku as hk


def _run(fn, *args, rng_seed: int = 0):
    rng = jax.random.PRNGKey(rng_seed)
    transformed = hk.transform(fn)
    params = transformed.init(rng, *args)
    return transformed.apply(params, rng, *args)


class TestAudioSTFT(unittest.TestCase):
    def test_stft_peak_at_440hz(self):
        from src.modules.multimodal.hybrid_audio_module import stft

        sr = 16000
        sine = jnp.sin(2 * jnp.pi * 440 * jnp.arange(sr) / sr)[None, :]
        mag = stft(sine)  # [B, T, F]
        avg_spec = mag.mean(axis=1)[0]
        peak_bin = int(jnp.argmax(avg_spec))
        # 440 * 512 / 16000 ≈ 14.08
        self.assertIn(peak_bin, (13, 14, 15))

    def test_learnable_filterbank_dynamic_range(self):
        from src.modules.multimodal.hybrid_audio_module import stft, LearnableFilterbank

        sr = 16000
        sine = jnp.sin(2 * jnp.pi * 440 * jnp.arange(sr) / sr)[None, :]
        mag = stft(sine)
        mel = _run(lambda x: LearnableFilterbank(n_mels=80)(x), mag)
        self.assertEqual(mel.shape[-1], 80)
        self.assertEqual(mel.shape[0], 1)
        self.assertGreater(float(mel.max() - mel.min()), 1.0)


class TestVideoSpatiotemporal(unittest.TestCase):
    def test_video_encoder_token_count(self):
        from src.modules.multimodal.fusion_module import VideoEncoder

        v = jax.random.normal(jax.random.PRNGKey(1), (1, 4, 32, 32, 3))
        out = _run(lambda x: VideoEncoder(d_model=64, patch_size=16, max_frames=4)(x), v)
        # 4 frames × (32/16)^2 = 4 × 4 = 16 patches/frame, 64 tokens total
        self.assertEqual(out.shape, (1, 64, 64))

    def test_3d_rope_shape_and_finite(self):
        from src.modules.multimodal.fusion_module import apply_3d_rope

        x = jax.random.normal(jax.random.PRNGKey(2), (1, 2 * 4 * 4, 24))
        y = apply_3d_rope(x, axis_lens=(2, 4, 4))
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(bool(jnp.all(jnp.isfinite(y))))
        # positional injection alters values
        self.assertGreater(float(jnp.abs(y - x).mean()), 0.0)


class TestVisionRGBD(unittest.TestCase):
    def test_rgbd_input(self):
        from src.modules.multimodal.fusion_module import VisionEncoder

        img = jax.random.normal(jax.random.PRNGKey(3), (1, 32, 32, 4))
        out = _run(lambda x: VisionEncoder(d_model=64, in_channels=4, num_layers=2)(x), img)
        self.assertEqual(out.shape[-1], 64)
        self.assertEqual(out.shape[0], 1)

    def test_rgb_into_rgbd_encoder_pads(self):
        from src.modules.multimodal.fusion_module import VisionEncoder

        img = jax.random.normal(jax.random.PRNGKey(4), (1, 32, 32, 3))
        out = _run(lambda x: VisionEncoder(d_model=64, in_channels=4, num_layers=2)(x), img)
        self.assertEqual(out.shape[-1], 64)


class TestDocumentEncoder(unittest.TestCase):
    def test_forward(self):
        from src.modules.multimodal.document_encoder import DocumentEncoder

        img = jax.random.normal(jax.random.PRNGKey(5), (1, 64, 64, 3))
        out = _run(lambda x: DocumentEncoder(d_model=64, num_layers=2)(x), img)
        self.assertIn("tokens", out)
        self.assertEqual(out["tokens"].shape[-1], 64)

    def test_table_grid_hook(self):
        from src.modules.multimodal.document_encoder import (
            DocumentEncoder,
            TableStructureEncoder,
        )

        img = jax.random.normal(jax.random.PRNGKey(6), (1, 64, 64, 3))

        def fwd(x):
            enc = DocumentEncoder(d_model=64, num_layers=2)
            tbl = TableStructureEncoder(d_model=64, max_rows=4, max_cols=4)
            out = enc(x)
            out["tokens"] = tbl(out["tokens"], rows=4, cols=4)
            return out

        out = _run(fwd, img)
        self.assertEqual(out["tokens"].shape[-1], 64)


class TestPointCloudEncoder(unittest.TestCase):
    def test_forward(self):
        from src.modules.multimodal.point_cloud_encoder import PointCloudEncoder

        pts = jax.random.normal(jax.random.PRNGKey(7), (2, 1024, 3))
        out = _run(lambda x: PointCloudEncoder(d_model=64)(x), pts)
        self.assertEqual(out["global"].shape, (2, 64))
        self.assertEqual(out["local"].shape[-1], 64)
        self.assertEqual(out["local"].shape[0], 2)


class TestBiosignalEncoder(unittest.TestCase):
    def test_forward(self):
        from src.modules.multimodal.biosignal_encoder import BiosignalEncoder

        sig = jax.random.normal(jax.random.PRNGKey(8), (2, 200, 32))
        out = _run(lambda x: BiosignalEncoder(d_model=64, downsample=4)(x), sig)
        self.assertEqual(out["features"].shape[-1], 64)
        self.assertEqual(out["features"].shape[0], 2)
        # downsampled time
        self.assertLess(out["features"].shape[1], 200)


class TestTactileEncoder(unittest.TestCase):
    def test_forward_global_shape(self):
        from src.modules.multimodal.tactile_encoder import TactileEncoder

        sig = jax.random.normal(jax.random.PRNGKey(9), (2, 50, 128))
        out = _run(lambda x: TactileEncoder(d_model=64)(x), sig)
        self.assertEqual(out["global"].shape, (2, 64))


class TestActionEncoder(unittest.TestCase):
    def test_forward_and_logits(self):
        from src.modules.multimodal.action_encoder import ActionEncoder

        acts = jax.random.uniform(jax.random.PRNGKey(10), (2, 10, 15), minval=-1, maxval=1)
        out = _run(lambda x: ActionEncoder(d_model=64, num_axes=15, num_bins=256)(x), acts)
        self.assertEqual(out["features"].shape, (2, 10, 64))
        self.assertEqual(out["action_logits"].shape, (2, 10, 15, 256))


class TestSpectrogramDecoder(unittest.TestCase):
    def test_upsample_ratio(self):
        from src.modules.multimodal.spectrogram_decoder import SpectrogramDecoder

        h = jax.random.normal(jax.random.PRNGKey(11), (1, 16, 64))
        out = _run(
            lambda x: SpectrogramDecoder(d_model=64, n_mels=80, upsample_factors=(2, 2))(x),
            h,
        )
        self.assertEqual(out["mel"].shape, (1, 64, 80))


class TestVQVAEImageTokenizer(unittest.TestCase):
    def test_recon_and_indices(self):
        from src.modules.multimodal.vqvae_image_tokenizer import VQVAEImageTokenizer

        img = jax.random.normal(jax.random.PRNGKey(12), (1, 32, 32, 3))
        out = _run(
            lambda x: VQVAEImageTokenizer(num_codes=256, code_dim=64, downsample_factor=16)(x),
            img,
        )
        self.assertEqual(out["reconstruction"].shape, img.shape)
        self.assertEqual(out["indices"].shape, (1, 2, 2))
        self.assertGreaterEqual(int(out["indices"].min()), 0)
        self.assertLess(int(out["indices"].max()), 256)
        self.assertGreater(float(out["total_loss"]), 0.0)


class TestStreamingVideoBuffer(unittest.TestCase):
    def test_circular_eviction(self):
        from src.modules.multimodal.streaming_video_buffer import StreamingVideoBuffer

        buf = StreamingVideoBuffer(d_model=64, max_frames=4, patches_per_frame=4)
        for i in range(6):
            buf.append(jnp.ones((4, 64)) * i, timestamp=i * 0.1)
        tokens, ts = buf.view()
        self.assertEqual(tokens.shape, (16, 64))
        self.assertEqual(ts.shape, (4,))
        for actual, expected in zip(ts.tolist(), [0.2, 0.3, 0.4, 0.5]):
            self.assertTrue(math.isclose(actual, expected, abs_tol=1e-3))

    def test_reset(self):
        from src.modules.multimodal.streaming_video_buffer import StreamingVideoBuffer

        buf = StreamingVideoBuffer(d_model=64, max_frames=4, patches_per_frame=4)
        buf.append(jnp.ones((4, 64)), timestamp=0.0)
        buf.reset()
        tokens, ts = buf.view()
        self.assertEqual(tokens.shape[0], 0)
        self.assertEqual(ts.shape[0], 0)


class TestSyncLossAndTimestampEmbedding(unittest.TestCase):
    def test_sync_loss_zero_for_aligned(self):
        from src.modules.multimodal.fusion_module import modality_synchronization_loss

        a = jnp.ones((1, 3, 8))
        b = jnp.ones((1, 3, 8))
        ta = jnp.array([[0.0, 0.1, 0.2]])
        tb = jnp.array([[0.05, 0.15, 0.25]])
        loss = float(modality_synchronization_loss(a, b, ta, tb, tolerance=0.1))
        self.assertLess(loss, 1e-4)

    def test_timestamp_embedding_shape(self):
        from src.modules.multimodal.fusion_module import sinusoidal_timestamp_embedding

        te = sinusoidal_timestamp_embedding(jnp.array([0.0, 0.5, 1.0]), 16)
        self.assertEqual(te.shape, (3, 16))


if __name__ == "__main__":
    unittest.main()
