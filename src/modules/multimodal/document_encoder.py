import haiku as hk
import jax
import jax.numpy as jnp
from typing import Dict, Optional, Any


def _2d_sinusoidal_bias(rows: int, cols: int, dim: int) -> jnp.ndarray:
    """2D sinusoidal positional bias [rows*cols, dim] for arbitrary page layouts."""
    half = dim // 2
    quarter = half // 2
    if quarter == 0:
        return jnp.zeros((rows * cols, dim))
    div = jnp.exp(-jnp.log(10000.0) * jnp.arange(quarter, dtype=jnp.float32) / quarter)
    r = jnp.arange(rows, dtype=jnp.float32)[:, None] * div
    c = jnp.arange(cols, dtype=jnp.float32)[:, None] * div
    row_emb = jnp.concatenate([jnp.sin(r), jnp.cos(r)], axis=-1)
    col_emb = jnp.concatenate([jnp.sin(c), jnp.cos(c)], axis=-1)
    grid = jnp.concatenate(
        [jnp.repeat(row_emb, cols, axis=0), jnp.tile(col_emb, (rows, 1))], axis=-1
    )
    if grid.shape[-1] < dim:
        grid = jnp.concatenate([grid, jnp.zeros((grid.shape[0], dim - grid.shape[-1]))], axis=-1)
    return grid[:, :dim]


class TableStructureEncoder(hk.Module):
    """Adds row/column embeddings per cell. cells [B, R*C, d] with grid (R,C)."""

    def __init__(self, d_model: int, max_rows: int = 64, max_cols: int = 32, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.max_rows = max_rows
        self.max_cols = max_cols

    def __call__(self, cells: jnp.ndarray, rows: int, cols: int) -> jnp.ndarray:
        row_emb = hk.get_parameter(
            "row_emb", [self.max_rows, self.d_model], init=hk.initializers.TruncatedNormal(0.02)
        )
        col_emb = hk.get_parameter(
            "col_emb", [self.max_cols, self.d_model], init=hk.initializers.TruncatedNormal(0.02)
        )
        grid_rows = jnp.repeat(row_emb[:rows], cols, axis=0)
        grid_cols = jnp.tile(col_emb[:cols], (rows, 1))
        bias = (grid_rows + grid_cols)[None, : rows * cols, :]
        return cells + bias


class ChartDecoder(hk.Module):
    """Multi-head: chart-type classifier + axis bbox + value extraction via cross-attn."""

    NUM_CHART_TYPES = 5

    def __init__(self, d_model: int, num_heads: int = 4, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_heads = num_heads
        self.type_head = hk.Sequential(
            [hk.Linear(d_model), jax.nn.silu, hk.Linear(self.NUM_CHART_TYPES)],
            name="chart_type",
        )
        self.bbox_head = hk.Linear(8, name="axis_bbox")
        self.value_attn = hk.MultiHeadAttention(
            num_heads=num_heads, key_size=d_model // num_heads,
            w_init=hk.initializers.TruncatedNormal(0.02), name="value_attn",
        )
        self.value_proj = hk.Linear(1, name="value_proj")

    def __call__(self, tokens: jnp.ndarray, queries: Optional[jnp.ndarray] = None) -> Dict[str, jnp.ndarray]:
        pooled = tokens.mean(axis=1)
        chart_logits = self.type_head(pooled)
        bbox = self.bbox_head(pooled)
        if queries is None:
            queries = pooled[:, None, :]
        attended = self.value_attn(queries, tokens, tokens)
        values = self.value_proj(attended).squeeze(-1)
        return {"chart_type_logits": chart_logits, "axis_bbox": bbox, "values": values}


class DocumentEncoder(hk.Module):
    """Document encoder: patch embed + 2D sinusoidal reading-order bias + transformer."""

    def __init__(
        self,
        d_model: int,
        patch_size: int = 16,
        num_layers: int = 4,
        num_heads: int = 8,
        name=None,
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_embed = hk.Conv2D(
            d_model, kernel_shape=patch_size, stride=patch_size, padding="VALID",
            name="patch_embed",
        )
        self.layers = []
        self.norms = []
        for i in range(num_layers):
            self.layers.append(
                hk.MultiHeadAttention(
                    num_heads=num_heads, key_size=d_model // num_heads,
                    w_init=hk.initializers.TruncatedNormal(0.02), name=f"doc_attn_{i}",
                )
            )
            self.norms.append(
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=f"doc_ln_{i}")
            )
        self.table_encoder = TableStructureEncoder(d_model, name="table_encoder")
        self.chart_decoder = ChartDecoder(d_model, name="chart_decoder")

    def __call__(
        self,
        page_image: jnp.ndarray,
        table_grid: Optional[tuple] = None,
        decode_chart: bool = False,
    ) -> Dict[str, Any]:
        """page_image [B,H,W,C] -> {tokens, chart_decoded?}."""
        b = page_image.shape[0]
        patches = self.patch_embed(page_image)
        ph, pw = patches.shape[1], patches.shape[2]
        tokens = patches.reshape(b, ph * pw, self.d_model)
        bias = _2d_sinusoidal_bias(ph, pw, self.d_model)[None, ...]
        tokens = tokens + bias
        if table_grid is not None:
            r, c = table_grid
            if r * c <= tokens.shape[1]:
                head = self.table_encoder(tokens[:, : r * c, :], r, c)
                tokens = jnp.concatenate([head, tokens[:, r * c :, :]], axis=1)
        for layer, norm in zip(self.layers, self.norms):
            tokens = norm(tokens + layer(tokens, tokens, tokens))
        out: Dict[str, Any] = {"tokens": tokens, "patch_grid": (ph, pw)}
        if decode_chart:
            out["chart"] = self.chart_decoder(tokens)
        return out
