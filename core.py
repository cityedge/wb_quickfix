from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Tuple

import numpy as np


# sRGB / XYZ (D65)
_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ)
_D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
_BRADFORD = np.array(
    [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ],
    dtype=np.float64,
)
_BRADFORD_INV = np.linalg.inv(_BRADFORD)
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


@dataclass
class EditParams:
    """All visible edit parameters, always interpreted from the untouched source image.

    There is intentionally no hidden white-balance matrix. Eyedropper operations
    estimate values for the same visible Temperature/Tint/Naturalize controls that
    the user can edit manually.
    """

    temperature: int = 0       # -200 ... +200, relative
    tint: int = 0              # -100 green ... +100 magenta
    naturalize: int = 0        # 0 ... 100
    brightness: int = 0        # -100 ... +100
    contrast: int = 0          # -100 ... +100
    saturation: int = 0        # -100 ... +100
    hue: int = 0               # -180 ... +180 degrees
    gamma: float = 1.0         # UI 0 ... 3, internal epsilon at zero

    def clone(self) -> "EditParams":
        return EditParams(
            temperature=int(self.temperature),
            tint=int(self.tint),
            naturalize=int(self.naturalize),
            brightness=int(self.brightness),
            contrast=int(self.contrast),
            saturation=int(self.saturation),
            hue=int(self.hue),
            gamma=float(self.gamma),
        )


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32, copy=False)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    # Negative linear values can occur after a matrix transform. Clip only the
    # negative side here; values >1 are retained until later gamut handling.
    rgb = np.maximum(rgb, 0.0)
    return np.where(
        rgb <= 0.0031308,
        12.92 * rgb,
        1.055 * np.power(rgb, 1.0 / 2.4) - 0.055,
    ).astype(np.float32, copy=False)


def white_balance_matrix_from_sample(sample_srgb: np.ndarray) -> np.ndarray:
    """Return a Bradford chromatic-adaptation matrix in linear-sRGB space.

    sample_srgb is a 3-vector in the 0..1 sRGB range and is assumed to be a
    surface that should be neutral (white or gray).
    """
    sample = np.clip(np.asarray(sample_srgb, dtype=np.float64), 1e-5, 1.0)
    sample_lin = srgb_to_linear(sample.astype(np.float32)).astype(np.float64)
    xyz = _RGB_TO_XYZ @ sample_lin
    if xyz[1] <= 1e-8:
        return np.eye(3, dtype=np.float64)

    src_white = xyz / xyz[1]
    src_lms = _BRADFORD @ src_white
    dst_lms = _BRADFORD @ _D65
    src_lms = np.where(np.abs(src_lms) < 1e-6, 1e-6, src_lms)
    adapt_xyz = _BRADFORD_INV @ np.diag(dst_lms / src_lms) @ _BRADFORD
    adapt_rgb = _XYZ_TO_RGB @ adapt_xyz @ _RGB_TO_XYZ

    # Very pathological samples can produce huge matrices. A conservative
    # soft fallback keeps this usable as an interactive retouching tool.
    if not np.all(np.isfinite(adapt_rgb)) or np.max(np.abs(adapt_rgb)) > 8.0:
        target = float(np.dot(sample_lin, _LUMA.astype(np.float64)))
        gains = target / np.maximum(sample_lin, 1e-5)
        gains = np.clip(gains, 0.25, 4.0)
        adapt_rgb = np.diag(gains)
    return adapt_rgb.astype(np.float64)


def skin_correction_gains_from_sample(sample_srgb: np.ndarray) -> np.ndarray:
    """Experimental, conservative global correction inferred from a skin sample.

    Unlike white-point correction, skin has no single correct RGB value. This
    chooses the nearest chromaticity from a small family of plausible skin
    references, then applies only about half of the implied correction and
    clamps per-channel gains tightly.
    """
    sample = np.clip(np.asarray(sample_srgb, dtype=np.float64), 1e-4, 1.0)
    sample_lin = srgb_to_linear(sample.astype(np.float32)).astype(np.float64)
    chroma = sample_lin / max(float(sample_lin.sum()), 1e-6)

    # Reference samples are deliberately varied rather than defining one
    # "standard skin". Values are sRGB-like exemplars; only chromaticity matters.
    refs_srgb = np.array(
        [
            [0.88, 0.70, 0.61],  # light / neutral-warm
            [0.78, 0.59, 0.48],  # medium / warm
            [0.68, 0.50, 0.39],  # medium / neutral
            [0.55, 0.37, 0.29],  # deeper / warm
            [0.44, 0.29, 0.23],  # deeper / neutral
            [0.82, 0.66, 0.58],  # slightly cooler/pinker
        ],
        dtype=np.float32,
    )
    refs_lin = srgb_to_linear(refs_srgb).astype(np.float64)
    refs_chr = refs_lin / np.maximum(refs_lin.sum(axis=1, keepdims=True), 1e-6)
    idx = int(np.argmin(np.sum((refs_chr - chroma[None, :]) ** 2, axis=1)))
    target_chr = refs_chr[idx]

    raw = target_chr / np.maximum(chroma, 1e-5)
    # Conservative correction: skin sampling is an inference, not a known neutral.
    gains = np.exp(np.log(np.maximum(raw, 1e-5)) * 0.50)
    gains = np.clip(gains, 0.85, 1.18)

    # Keep approximate luminance neutral so the operation mostly changes color.
    norm = float(np.dot(gains, _LUMA.astype(np.float64)))
    if norm > 1e-6:
        gains /= norm
    return np.clip(gains, 0.82, 1.22).astype(np.float64)


def _temperature_tint_from_gains(gains: np.ndarray) -> tuple[int, int]:
    """Convert relative RGB gains to this app's visible Temperature/Tint controls."""
    g = np.maximum(np.asarray(gains, dtype=np.float64), 1e-6)
    # Inverse of _temperature_tint_gains, using channel ratios so luminance
    # normalization cancels out.
    temperature = math.log(g[0] / g[2]) / 0.0070
    tint = math.log((g[0] * g[2]) / (g[1] * g[1])) / 0.0108
    return (
        int(round(np.clip(temperature, -200.0, 200.0))),
        int(round(np.clip(tint, -100.0, 100.0))),
    )


def white_params_from_sample(sample_srgb: np.ndarray) -> tuple[int, int, int]:
    """Estimate visible WB controls from an ORIGINAL-image neutral sample.

    The picked patch is assumed to be neutral. We derive diagonal linear-RGB
    gains, then express those gains as the UI's Temperature/Tint pair. Naturalize
    is increased modestly for stronger corrections to suppress clipping and
    over-saturation, but remains a visible/editable parameter rather than a
    hidden operation.
    """
    sample = np.clip(np.asarray(sample_srgb, dtype=np.float64), 1e-5, 1.0)
    lin = srgb_to_linear(sample.astype(np.float32)).astype(np.float64)
    # Any common target luminance cancels when converted to gain ratios.
    gains = 1.0 / np.maximum(lin, 1e-5)
    temperature, tint = _temperature_tint_from_gains(gains)
    severity = max(abs(temperature), abs(tint))
    naturalize = 0 if severity < 5 else int(round(np.clip(severity * 0.35, 0, 45)))
    return temperature, tint, naturalize


def skin_params_from_sample(sample_srgb: np.ndarray) -> tuple[int, int, int]:
    """Estimate visible WB controls from an ORIGINAL-image skin sample.

    Skin has no single correct color, so this reuses the conservative family of
    skin chromaticity references and maps the inferred weak RGB correction into
    the same visible Temperature/Tint controls.
    """
    gains = skin_correction_gains_from_sample(sample_srgb)
    temperature, tint = _temperature_tint_from_gains(gains)
    severity = max(abs(temperature), abs(tint))
    naturalize = 12 if severity < 8 else int(round(np.clip(12 + severity * 0.25, 12, 40)))
    return temperature, tint, naturalize


def sample_median_rgb(rgb_u8: np.ndarray, x: int, y: int, size: int = 11) -> np.ndarray:
    """Median RGB sample from a square region, returned as sRGB float 0..1."""
    h, w = rgb_u8.shape[:2]
    r = max(0, size // 2)
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    patch = rgb_u8[y0:y1, x0:x1, :3]
    if patch.size == 0:
        return np.array([0.5, 0.5, 0.5], dtype=np.float32)
    med = np.median(patch.reshape(-1, 3), axis=0)
    return (med / 255.0).astype(np.float32)



def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Return a weighted median for one-dimensional finite data."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    good = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(good):
        finite = v[np.isfinite(v)]
        return float(np.median(finite)) if finite.size else 0.0
    v = v[good]
    w = w[good]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cutoff = 0.5 * float(w.sum())
    idx = int(np.searchsorted(np.cumsum(w), cutoff, side="left"))
    return float(v[min(idx, v.size - 1)])


def auto_white_sample(
    rgb_u8: np.ndarray,
    alpha_u8: np.ndarray | None = None,
    max_work_dim: int = 768,
    block_size: int = 12,
) -> np.ndarray:
    """Estimate a global white reference from a *group* of white-like regions.

    The original 1.0.1 implementation deliberately chose a very neutral-looking
    bright patch and therefore tended to be conservative when the whole image had
    a strong color cast.  This version instead gathers many plausible white/gray
    blocks, rejects obvious colored/outlier blocks, then estimates the robust
    *center of their chromaticity distribution*.

    This means that if all plausible neutral surfaces are shifted red/orange by the
    illuminant, the red/orange center itself becomes the white reference and Auto
    WB applies a correspondingly stronger correction toward neutral white.

    The operation is lightweight and deterministic.  Analysis is performed on a
    reduced-resolution view of the untouched source image; the returned value is
    an sRGB 0..1 triplet suitable for ``white_params_from_sample``.
    """
    src = np.asarray(rgb_u8, dtype=np.uint8)
    if src.ndim != 3 or src.shape[2] < 3 or src.shape[0] < 1 or src.shape[1] < 1:
        raise ValueError("RGB image is empty or invalid")
    src = src[..., :3]
    h, w = src.shape[:2]

    # Analyze a reduced-resolution view.  The source image itself is never altered
    # or replaced by the current preview/edit state.
    step = max(1, int(math.ceil(max(h, w) / max(64, int(max_work_dim)))))
    work = src[::step, ::step]

    if alpha_u8 is not None:
        alpha = np.asarray(alpha_u8, dtype=np.uint8)
        if alpha.shape[:2] != (h, w):
            raise ValueError("Alpha channel size does not match RGB image")
        valid_work = alpha[::step, ::step] >= 64
    else:
        valid_work = np.ones(work.shape[:2], dtype=bool)

    bh = max(4, int(block_size))
    medians: list[np.ndarray] = []
    luminances: list[float] = []
    chromas: list[float] = []
    textures: list[float] = []
    clipped: list[float] = []

    luma = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    wh, ww = work.shape[:2]
    for y0 in range(0, wh, bh):
        for x0 in range(0, ww, bh):
            patch = work[y0:min(y0 + bh, wh), x0:min(x0 + bh, ww)]
            mask = valid_work[y0:min(y0 + bh, wh), x0:min(x0 + bh, ww)]
            if patch.size == 0:
                continue
            flat = patch.reshape(-1, 3)
            mflat = mask.reshape(-1)
            needed = max(8, int(math.ceil(flat.shape[0] * 0.35)))
            if int(mflat.sum()) < needed:
                continue
            pix = flat[mflat]
            med = np.median(pix, axis=0).astype(np.float32) / 255.0
            mean_rgb = max(float(med.mean()), 1e-6)
            chroma = float((med.max() - med.min()) / mean_rgb)
            lum = float(np.dot(med, luma))

            pixf = pix.astype(np.float32) / 255.0
            pl = pixf @ luma
            texture = float(np.percentile(pl, 90) - np.percentile(pl, 10))
            clip_fraction = float(np.mean(np.all(pix >= 250, axis=1)))

            medians.append(med)
            luminances.append(lum)
            chromas.append(chroma)
            textures.append(texture)
            clipped.append(clip_fraction)

    if not medians:
        pix = work[valid_work]
        if pix.size == 0:
            pix = work.reshape(-1, 3)
        return (np.median(pix.reshape(-1, 3), axis=0) / 255.0).astype(np.float32)

    med = np.asarray(medians, dtype=np.float32)
    lum = np.asarray(luminances, dtype=np.float32)
    chroma = np.asarray(chromas, dtype=np.float32)
    texture = np.asarray(textures, dtype=np.float32)
    clip = np.asarray(clipped, dtype=np.float32)

    # 1) Find reasonably bright regions.  Percentile-based selection allows the
    # same logic to work in dim indoor scenes as well as daylight photographs.
    bright_cut = max(0.14, float(np.percentile(lum, 60)))
    bright = lum >= bright_cut
    if int(bright.sum()) < 6:
        bright = lum >= max(0.10, float(np.percentile(lum, 45)))
    if not np.any(bright):
        bright = np.ones_like(lum, dtype=bool)

    # 2) Among bright blocks, keep the lower-chroma majority rather than only the
    # single "most neutral" patch.  The relative percentile is the key: a white
    # wall under red light may have substantial raw chroma but can still be less
    # chromatic than genuinely red/green/blue scene objects.
    bright_chroma = chroma[bright]
    chroma_cut = float(np.percentile(bright_chroma, 72)) if bright_chroma.size else float(np.median(chroma))
    chroma_cut = float(np.clip(max(chroma_cut, 0.22), 0.22, 1.20))
    candidate = bright & (chroma <= chroma_cut)

    # Prefer information-bearing (not fully clipped) blocks whenever possible.
    nonclipped = candidate & (clip < 0.55)
    if int(nonclipped.sum()) >= 4:
        candidate = nonclipped

    # If the first pass is too sparse, relax chroma selection while still avoiding
    # only the most extreme saturated blocks.
    if int(candidate.sum()) < 4:
        relaxed_cut = float(np.percentile(bright_chroma, 82)) if bright_chroma.size else 1.20
        relaxed_cut = float(np.clip(max(relaxed_cut, 0.35), 0.35, 1.40))
        candidate = bright & (chroma <= relaxed_cut) & (clip < 0.80)
    if not np.any(candidate):
        candidate = bright

    idx = np.flatnonzero(candidate)
    cmed = med[idx]
    clum = lum[idx]
    cchroma = chroma[idx]
    ctexture = texture[idx]
    cclip = clip[idx]

    # Convert candidate colors to linear-RGB log chromaticity.  Common brightness
    # factors disappear in R/G and B/G, which is exactly what Auto WB needs.
    lin = srgb_to_linear(np.clip(cmed, 1e-4, 1.0)).astype(np.float64)
    eps = 1e-7
    u = np.log((lin[:, 0] + eps) / (lin[:, 1] + eps))  # red vs green
    v = np.log((lin[:, 2] + eps) / (lin[:, 1] + eps))  # blue vs green

    # Bright, locally stable blocks count somewhat more, but we intentionally do
    # *not* heavily reward raw neutrality: doing so would reintroduce the old
    # conservative bias toward the least-cast white patch.
    max_lum = max(float(np.max(clum)), bright_cut + 1e-6)
    br = np.clip((clum - bright_cut) / (max_lum - bright_cut + 1e-6), 0.0, 1.0)
    weights = (0.20 + 0.80 * br) ** 2.2
    weights *= 1.0 / (1.0 + 2.0 * ctexture)
    weights *= np.clip(1.0 - 0.70 * cclip, 0.15, 1.0)
    weights *= 1.0 / (1.0 + 0.10 * (cchroma / max(chroma_cut, 0.10)) ** 2)
    weights = np.maximum(weights.astype(np.float64), 1e-4)

    # 3) Robust center.  Start from weighted coordinate medians, measure each
    # block's distance from that center, and reject chromatic outliers.  This keeps
    # a blue sky patch or a pale colored wall from dragging the estimate too far.
    u0 = _weighted_median(u, weights)
    v0 = _weighted_median(v, weights)
    dist = np.sqrt((u - u0) ** 2 + (v - v0) ** 2)
    dmed = _weighted_median(dist, weights)
    dmad = _weighted_median(np.abs(dist - dmed), weights)
    robust_sigma = max(1.4826 * dmad, 0.018)
    dist_cut = max(0.055, dmed + 2.8 * robust_sigma)
    inlier = dist <= dist_cut

    # Guarantee a meaningful group even in unusual images with several distinct
    # lighting/color clusters.  Use *weight mass*, not a fixed fraction of block
    # count: a large patch of moderately bright sky should not force itself into
    # the white cluster merely because it occupies many blocks.
    inlier_weight = float(weights[inlier].sum())
    total_weight = float(weights.sum())
    if int(inlier.sum()) < 3 or inlier_weight < 0.45 * total_weight:
        order = np.argsort(dist)
        cumulative = np.cumsum(weights[order])
        need = max(3, int(np.searchsorted(cumulative, 0.55 * total_weight, side="left") + 1))
        need = min(need, len(order))
        inlier = np.zeros_like(dist, dtype=bool)
        inlier[order[:need]] = True

    iw = weights[inlier]
    # After outlier rejection, use the weighted mean: unlike choosing one patch,
    # this genuinely targets the center of the plausible white/gray population.
    u_center = float(np.average(u[inlier], weights=iw))
    v_center = float(np.average(v[inlier], weights=iw))

    # Reconstruct a representative white sample from chromaticity only.  Absolute
    # luminance is irrelevant to white_params_from_sample; choose a safe scale so
    # no channel clips before conversion back to sRGB.
    ratio = np.array([math.exp(u_center), 1.0, math.exp(v_center)], dtype=np.float64)
    ratio = np.clip(ratio, 0.12, 8.0)
    ratio *= 0.78 / max(float(ratio.max()), 1e-6)
    sample = linear_to_srgb(ratio.astype(np.float32))
    return np.clip(sample, 0.0, 1.0).astype(np.float32)

def _temperature_tint_gains(temperature: float, tint: float) -> np.ndarray:
    # Relative UI controls, intentionally not pretending to be absolute Kelvin.
    t = float(np.clip(temperature, -200.0, 200.0))
    q = float(np.clip(tint, -100.0, 100.0))
    gains = np.array(
        [
            math.exp(0.0035 * t + 0.0018 * q),
            math.exp(-0.0036 * q),
            math.exp(-0.0035 * t + 0.0018 * q),
        ],
        dtype=np.float32,
    )
    norm = float(np.dot(gains, _LUMA))
    if norm > 1e-6:
        gains /= norm
    return np.clip(gains, 0.25, 4.00)


def _skin_mask_srgb(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.5 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 + 0.5 * r - 0.418688 * g - 0.081312 * b
    # Broad detector, used only for weak protection inside Naturalize.
    return (
        (y > 0.10)
        & (y < 0.97)
        & (cb > 0.26)
        & (cb < 0.56)
        & (cr > 0.50)
        & (cr < 0.75)
        & (r > b * 0.92)
    )


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _naturalize(original_lin: np.ndarray, corrected_lin: np.ndarray, strength: int) -> np.ndarray:
    n = float(np.clip(strength, 0, 100)) / 100.0
    if n <= 0.0:
        return corrected_lin

    out = corrected_lin.copy()
    y = np.sum(original_lin * _LUMA[None, None, :], axis=2)

    # Deep shadows and near-highlights are less reliable WB references and more
    # prone to clipping/noise, so gently reduce correction there.
    shadow = 1.0 - _smoothstep(0.025, 0.22, y)
    highlight = _smoothstep(0.72, 0.98, y)
    protect = n * (0.28 * shadow + 0.20 * highlight)
    out = out * (1.0 - protect[..., None]) + original_lin * protect[..., None]

    # Prevent WB from creating excessive chroma relative to the source.
    orig_max = original_lin.max(axis=2)
    orig_min = original_lin.min(axis=2)
    out_max = out.max(axis=2)
    out_min = out.min(axis=2)
    orig_chroma = orig_max - orig_min
    out_chroma = out_max - out_min
    allowed = orig_chroma * 1.30 + 0.035
    over = np.clip((out_chroma - allowed) / np.maximum(out_chroma, 1e-5), 0.0, 1.0)
    gray = np.sum(out * _LUMA[None, None, :], axis=2, keepdims=True)
    chroma_reduce = (n * 0.55 * over)[..., None]
    out = out * (1.0 - chroma_reduce) + gray * chroma_reduce

    # Weak skin protection. This does not "make skin a standard color"; it only
    # reduces aggressive global shifts in pixels already resembling skin.
    original_srgb = np.clip(linear_to_srgb(original_lin), 0.0, 1.0)
    smask = _skin_mask_srgb(original_srgb).astype(np.float32)
    skin_protect = (n * 0.16 * smask)[..., None]
    out = out * (1.0 - skin_protect) + original_lin * skin_protect

    # Hue-preserving headroom compression for channels pushed above display gamut.
    mx = out.max(axis=2, keepdims=True)
    scale = np.where(mx > 1.0, 1.0 / (1.0 + n * 0.85 * (mx - 1.0)), 1.0)
    out *= scale
    return out


def _apply_hue_rotation(rgb: np.ndarray, degrees: float) -> np.ndarray:
    if abs(degrees) < 1e-6:
        return rgb
    a = math.radians(float(degrees))
    c, s = math.cos(a), math.sin(a)

    # YIQ hue rotation. It is compact, fast and adequate for a beginner-level
    # global hue control.
    rgb_to_yiq = np.array(
        [[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]],
        dtype=np.float32,
    )
    yiq_to_rgb = np.linalg.inv(rgb_to_yiq).astype(np.float32)
    rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    mat = yiq_to_rgb @ rot @ rgb_to_yiq
    return np.einsum("...c,dc->...d", rgb, mat, optimize=True)


def apply_pipeline(rgb_u8: np.ndarray, params: EditParams) -> np.ndarray:
    """Apply all corrections to an RGB uint8 image and return RGB uint8."""
    src = np.asarray(rgb_u8, dtype=np.uint8)[..., :3]
    srgb = src.astype(np.float32) / 255.0
    original_lin = srgb_to_linear(srgb)

    # All white-balance correction is represented by visible parameters.
    # The untouched source is always the input; picker operations merely replace
    # these parameter values and never bake or accumulate a prior correction.
    lin = original_lin * _temperature_tint_gains(params.temperature, params.tint)[None, None, :]

    # Composite "make it natural" control.
    lin = _naturalize(original_lin, lin, params.naturalize)

    # Beginner-level brightness. Darkening is intentionally much stronger than
    # in 1.0.1: -100 reaches -4 EV (1/16 linear light), while the positive side
    # keeps the existing +1.5 EV maximum to avoid excessive highlight clipping.
    brightness = float(np.clip(params.brightness, -100, 100))
    if brightness < 0.0:
        ev = brightness / 100.0 * 4.0
    else:
        ev = brightness / 100.0 * 1.5
    lin *= np.float32(2.0 ** ev)

    rgb = linear_to_srgb(lin)
    rgb = np.clip(rgb, 0.0, 1.0)

    # Gamma: conventional correction y = x^(1/gamma). UI may show 0; use epsilon.
    gamma = max(float(params.gamma), 0.01)
    if abs(gamma - 1.0) > 1e-6:
        rgb = np.power(rgb, 1.0 / gamma, dtype=np.float32)

    # Contrast around middle gray, factor range 0.5..2.0.
    contrast_factor = 2.0 ** (float(np.clip(params.contrast, -100, 100)) / 100.0)
    rgb = (rgb - 0.5) * contrast_factor + 0.5

    # Hue rotation followed by saturation around luminance.
    rgb = _apply_hue_rotation(rgb, params.hue)
    sat_factor = 1.0 + float(np.clip(params.saturation, -100, 100)) / 100.0
    gray = np.sum(rgb * _LUMA[None, None, :], axis=2, keepdims=True)
    rgb = gray + (rgb - gray) * sat_factor

    rgb = np.clip(rgb, 0.0, 1.0)
    return np.rint(rgb * 255.0).astype(np.uint8)


def sample_warning(sample_srgb: np.ndarray) -> str:
    """Machine-readable warning key for a sampled point."""
    s = np.asarray(sample_srgb, dtype=np.float32)
    lum = float(np.dot(s, np.array([0.299, 0.587, 0.114], dtype=np.float32)))
    if lum < 0.05:
        return "too_dark"
    if float(s.max()) > 0.985 and float(s.min()) > 0.93:
        return "near_clipped"
    return ""
