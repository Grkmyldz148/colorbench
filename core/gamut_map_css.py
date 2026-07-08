"""CSS Color 4 gamut mapping algorithms, space-agnostic (G1 prereg harness).

Faithful ports of the three CSS gamut mapping algorithms, generalized so the
working color space is a parameter (any object with numpy `from_XYZ(xyz)->Lab`
and `to_XYZ(lab)->xyz`, L in [0,1]):

  · binsearch_minde — spec §"Binary Search Gamut Mapping with Local MINDE"
    (drafts.csswg.org/css-color-4/#binsearch), pseudocode followed step by step.
  · edge_seeker     — Alexey Ardov's LUT method, ported from
    color-js/apps/gamut-mapping/edge-seeker (research/fortress/refs/).
  · raytrace        — Isaac Muse's method, spec §"Ray Trace Gamut Mapping".
    The spec pseudocode's ray-cast has an inverted branch condition
    (`if abs(d) < 1E-12` guards the NON-parallel slab case; the comment and the
    reference implementations show it must be `>=`). We implement the correct
    logic, matching colorjs.io/coloraide behavior.

All algorithms take/return colors as XYZ (source) -> linear RGB (destination);
RGB<->XYZ matrices are parameters so the harness controls the white convention.
deltaE is Euclidean distance in the working space's own Lab coordinates (for
OKLab this IS deltaEOK; JND=0.02 assumes an L-range of [0,1], true for both
G1 contestants — declared in the prereg addendum).
"""
import numpy as np

# ── generic LCh helpers ─────────────────────────────────────────────────────

def _to_lch(lab):
    L, a, b = lab
    return L, float(np.hypot(a, b)), float(np.degrees(np.arctan2(b, a)) % 360.0)


def _from_lch(L, C, h):
    hr = np.radians(h)
    return np.array([L, C * np.cos(hr), C * np.sin(hr)])


def _in_gamut(rgb, eps):
    return bool(np.all(rgb >= -eps) and np.all(rgb <= 1.0 + eps))


# ── OKLab exactly as CSS / colorjs.io ───────────────────────────────────────

class OKLabCSS:
    """OKLab with the colorjs.io / CSS Color 4 matrices (XYZ relative to D65)."""
    _XYZ2LMS = np.array([[0.8190224379967030, 0.3619062600528904, -0.1288737815209879],
                         [0.0329836539323885, 0.9292868615863434,  0.0361446663506424],
                         [0.0481771893596242, 0.2642395317527308,  0.6335478284694309]])
    _LMS2XYZ = np.array([[ 1.2268798758459243, -0.5578149944602171,  0.2813910456659647],
                         [-0.0405757452148008,  1.1122868032803170, -0.0717110580655164],
                         [-0.0763729366746601, -0.4214933324022432,  1.5869240198367816]])
    _LMS2LAB = np.array([[0.2104542683093140,  0.7936177747023054, -0.0040720430116193],
                         [1.9779985324311684, -2.4285922420485799,  0.4505937096174110],
                         [0.0259040424655478,  0.7827717124575296, -0.8086757549230774]])
    _LAB2LMS = np.array([[1.0, 0.3963377773761749,  0.2158037573099136],
                         [1.0, -0.1055613458156586, -0.0638541728258133],
                         [1.0, -0.0894841775298119, -1.2914855480194092]])

    def from_XYZ(self, xyz):
        lms = np.asarray(xyz, float) @ self._XYZ2LMS.T
        return np.cbrt(lms) @ self._LMS2LAB.T

    def to_XYZ(self, lab):
        lms = np.asarray(lab, float) @ self._LAB2LMS.T
        return (lms ** 3) @ self._LMS2XYZ.T


# ── 1) Binary search + local MINDE (spec pseudocode, step by step) ──────────

def binsearch_minde(xyz, M_dst, M_dst_inv, space,
                    jnd=0.02, epsilon=1e-4, gamut_eps=1e-6):
    """Map one XYZ color into the destination RGB gamut. Returns linear dst RGB."""
    lab = space.from_XYZ(np.asarray(xyz, float))
    L, C, h = _to_lch(lab)
    if L >= 1.0:
        return np.clip(space.to_XYZ(np.array([1.0, 0, 0])) @ M_dst_inv.T, 0, 1)
    if L <= 0.0:
        return np.zeros(3)

    def to_rgb(lch):
        return space.to_XYZ(_from_lch(*lch)) @ M_dst_inv.T

    def delta(lch1, rgb2):                       # deltaE(space) of lch vs an RGB color
        lab2 = space.from_XYZ(np.asarray(rgb2, float) @ M_dst.T)
        return float(np.linalg.norm(_from_lch(*lch1) - lab2))

    rgb = to_rgb((L, C, h))
    if _in_gamut(rgb, gamut_eps):
        return np.clip(rgb, 0, 1)

    clipped = np.clip(rgb, 0, 1)
    if delta((L, C, h), clipped) < jnd:
        return clipped

    lo, hi = 0.0, C
    lo_in_gamut = True
    while hi - lo > epsilon:
        chroma = (lo + hi) / 2
        cur = (L, chroma, h)
        rgb = to_rgb(cur)
        if lo_in_gamut and _in_gamut(rgb, gamut_eps):
            lo = chroma
            continue
        clipped = np.clip(rgb, 0, 1)
        E = delta(cur, clipped)
        if E < jnd:
            if jnd - E < epsilon:
                return clipped
            lo_in_gamut = False
            lo = chroma
        else:
            hi = chroma
    return clipped


# ── 2) EdgeSeeker (Ardov; port of color-js/apps edge-seeker) ────────────────

def _hsl_to_rgb(h, s, l):
    h = h % 360.0
    m1 = l + s * (l if l < 0.5 else 1 - l)
    m2 = m1 - (m1 - l) * 2 * abs(((h / 60.0) % 2) - 1)
    if h < 60:   return np.array([m1, m2, 2 * l - m1])
    if h < 120:  return np.array([m2, m1, 2 * l - m1])
    if h < 180:  return np.array([2 * l - m1, m1, m2])
    if h < 240:  return np.array([2 * l - m1, m2, m1])
    if h < 300:  return np.array([m2, 2 * l - m1, m1])
    return np.array([m1, 2 * l - m1, m2])


def _is_cw(h1, h2):
    d = (h2 % 360.0) - (h1 % 360.0)
    if d < 0:
        d += 360.0
    return 0 <= d <= 180.0


def _lerp_color(c1, c2, hue):
    if hue == c1['h']: return dict(c1, h=hue)
    if hue == c2['h']: return dict(c2, h=hue)
    t = (hue - c1['h']) / (c2['h'] - c1['h'])
    out = {'h': hue,
           'l': c1['l'] * (1 - t) + c2['l'] * t,
           'c': c1['c'] * (1 - t) + c2['c'] * t}
    if 'curvature' in c1 and 'curvature' in c2:
        out['curvature'] = c1['curvature'] * (1 - t) + c2['curvature'] * t
    return out


def _fix_section(section):
    result, has_error = [], False
    for idx, color in enumerate(section):
        if idx == 0:
            result.append(color)
            continue
        will_cw = idx == len(section) - 1 or _is_cw(color['h'], section[idx + 1]['h'])
        if will_cw:
            if has_error:
                first_bad = next(i for i, a in enumerate(result)
                                 if not _is_cw(a['h'], color['h']))
                repl = _lerp_color(result[first_bad - 1], result[first_bad],
                                   color['h'] - 0.0001)
                result = result[:first_bad] + [repl, color]
                has_error = False
            else:
                result.append(color)
        else:
            if not has_error:
                has_error = True
                result.append(color)
    return result


def _filter_interp(colors, threshold=1e-5):
    quad = threshold ** 2
    result = colors
    while True:
        keep = []
        for i, c in enumerate(result):
            if i == 0 or i == len(result) - 1 or i % 2 == 0:
                keep.append(c)
                continue
            interp = _lerp_color(result[i - 1], result[i + 1], c['h'])
            if (c['l'] - interp['l']) ** 2 + (c['c'] - interp['c']) ** 2 > quad:
                keep.append(c)
        if len(keep) == len(result):
            return result
        result = keep


def _curvature(peak, curve):
    x = (1 - curve['l']) / (1 - peak['l'])
    y = curve['c'] / peak['c']
    if x == y:
        return 0.0
    def bisector(p1, p2):
        return (-1.0 / ((p2[1] - p1[1]) / (p2[0] - p1[0])),
                ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2))
    m1, p1 = bisector((0, 0), (x, y))
    m2, p2 = bisector((0, 0), (1, 1))
    cx = (m1 * p1[0] - m2 * p2[0] + p2[1] - p1[1]) / (m1 - m2)
    cy = m1 * (cx - p1[0]) + p1[1]
    curv = 1.0 / np.sqrt(cx ** 2 + cy ** 2)
    return -curv if cx < cy else curv


def srgb_decode(rgb):
    """sRGB/Display-P3 transfer decode (encoded -> linear)."""
    rgb = np.asarray(rgb, float)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


class EdgeSeekerLUT:
    """Max-chroma-per-hue LUT for (space, destination gamut). SLICES=400 as ref.

    `decode` is the destination gamut's transfer decode: the reference builds the
    LUT from an HSL sweep over the gamma-ENCODED destination RGB cube
    (Color("p3", hslToRgb(...))), so encoded values must be linearized before M_dst.
    """
    def __init__(self, space, M_dst, slices=400, decode=srgb_decode):
        def rgb_to_lch(rgb):
            lin = decode(rgb) if decode is not None else np.asarray(rgb, float)
            L, C, h = _to_lch(space.from_XYZ(lin @ M_dst.T))
            return {'l': L, 'c': C, 'h': h}
        def color_list(lightness):
            segments = []
            for h0, h1 in ((0, 60), (60, 120), (120, 180),
                           (180, 240), (240, 300), (300, 360)):
                seg = [rgb_to_lch(_hsl_to_rgb(h0 + (h1 - h0) * i / (slices - 1),
                                              1.0, lightness))
                       for i in range(slices)]
                segments.append(_filter_interp(_fix_section(seg)))
            merged = sorted((c for seg in segments for c in seg[:-1]),
                            key=lambda c: c['h'])
            end = _lerp_color(merged[-1], dict(merged[0], h=merged[0]['h'] + 360), 360.0)
            return [dict(end, h=0.0)] + merged + [end]
        peaks = color_list(0.5)
        curves = color_list(0.75)
        self.lut = []
        for pk in peaks:
            cv = self._get(pk['h'], curves)
            self.lut.append(dict(pk, curvature=_curvature(pk, cv)))

    @staticmethod
    def _get(h, lut):
        lo, hi = 0, len(lut) - 1
        mid = (lo + hi) // 2
        while lo <= hi:
            if lut[mid]['h'] == h:
                return lut[mid]
            if lut[mid]['h'] < h:
                lo = mid + 1
            else:
                hi = mid - 1
            mid = (lo + hi) // 2
        return _lerp_color(lut[mid], lut[mid + 1], h)

    def max_chroma(self, l, h):
        if l <= 0 or l >= 1:
            return 0.0
        item = self._get(h % 360.0, self.lut)
        if l <= item['l']:                       # linear bottom section
            return (l / item['l']) * item['c']
        x = (1 - l) / (1 - item['l'])            # arc top section
        return item['c'] * _arc_intersection(x, item['curvature'])


def _arc_intersection(x, curvature):
    if curvature == 0:
        return x
    radius = abs(1.0 / curvature)
    half_diag = np.sqrt(0.5)
    dist = np.sqrt(radius ** 2 - half_diag ** 2)
    off = dist / np.sqrt(2)
    cx = (off if curvature > 0 else -off) + 0.5
    cy = (-off if curvature > 0 else off) + 0.5
    under = radius ** 2 - (x - cx) ** 2
    if under < 0:
        return 0.0
    s = np.sqrt(under)
    r1 = cy + s
    return r1 if 0 <= r1 <= 1 else cy - s


def edge_seeker(xyz, M_dst, M_dst_inv, space, lut):
    """Map one XYZ color via EdgeSeeker. `lut` = EdgeSeekerLUT(space, M_dst)."""
    L, C, h = _to_lch(space.from_XYZ(np.asarray(xyz, float)))
    if L <= 0.0:
        return np.zeros(3)
    if L >= 1.0:
        return np.clip(space.to_XYZ(np.array([1.0, 0, 0])) @ M_dst_inv.T, 0, 1)
    C = min(C, lut.max_chroma(L, h))
    rgb = space.to_XYZ(_from_lch(L, C, h)) @ M_dst_inv.T
    return np.clip(rgb, 0, 1)


# ── 3) Ray trace (Muse; spec pseudocode with corrected ray-cast branch) ─────

def _cast_ray(start, end):
    tnear, tfar = -np.inf, np.inf
    direction = end - start
    for i in range(3):
        a, d = start[i], direction[i]
        if abs(d) >= 1e-12:                      # non-parallel slab (spec typo fixed)
            t1 = (0.0 - a) / d
            t2 = (1.0 - a) / d
            tnear = max(min(t1, t2), tnear)
            tfar = min(max(t1, t2), tfar)
        elif a < 0.0 or a > 1.0:
            return None
    if tnear > tfar or tfar < 0:
        return None
    if tnear < 0:
        tnear = tfar
    if not np.isfinite(tnear):
        return None
    return start + direction * tnear


def raytrace(xyz, M_dst, M_dst_inv, space, gamut_eps=1e-6):
    """Map one XYZ color via Ray Trace. Returns linear dst RGB."""
    lab = space.from_XYZ(np.asarray(xyz, float))
    L, C, h = _to_lch(lab)
    if L >= 1.0:
        return np.clip(space.to_XYZ(np.array([1.0, 0, 0])) @ M_dst_inv.T, 0, 1)
    if L <= 0.0:
        return np.zeros(3)

    anchor = space.to_XYZ(_from_lch(L, 0.0, h)) @ M_dst_inv.T
    rgb = space.to_XYZ(lab) @ M_dst_inv.T
    if not _in_gamut(rgb, gamut_eps):
        low, high = 1e-6, 1.0 - 1e-6
        last = rgb
        for i in range(4):
            if i > 0:
                lch = _to_lch(space.from_XYZ(rgb @ M_dst.T))
                rgb = space.to_XYZ(_from_lch(L, lch[1], h)) @ M_dst_inv.T
            hit = _cast_ray(anchor, rgb)
            if hit is None:
                rgb = last
                break
            if i > 0 and np.all(rgb > low) and np.all(rgb < high):
                anchor = rgb
            rgb = hit
            last = hit
    return np.clip(rgb, 0, 1)


# ── batch driver ────────────────────────────────────────────────────────────

ALGORITHMS = ('binsearch_minde', 'edge_seeker', 'raytrace')


def gamut_map_batch(xyz_array, M_dst, M_dst_inv, space, algorithm, lut=None):
    """Map (N,3) XYZ colors; returns (N,3) linear destination RGB."""
    xyz_array = np.atleast_2d(np.asarray(xyz_array, float))
    if algorithm == 'edge_seeker' and lut is None:
        lut = EdgeSeekerLUT(space, M_dst)
    out = np.empty_like(xyz_array)
    for i, xyz in enumerate(xyz_array):
        if algorithm == 'binsearch_minde':
            out[i] = binsearch_minde(xyz, M_dst, M_dst_inv, space)
        elif algorithm == 'edge_seeker':
            out[i] = edge_seeker(xyz, M_dst, M_dst_inv, space, lut)
        elif algorithm == 'raytrace':
            out[i] = raytrace(xyz, M_dst, M_dst_inv, space)
        else:
            raise ValueError(algorithm)
    return out


# ── vectorized implementations (G1 addendum-2 §C: must pass 1e-12 self-parity
#    against the per-color versions above before use) ────────────────────────

def _to_lch_v(lab):
    L = lab[:, 0]
    C = np.hypot(lab[:, 1], lab[:, 2])
    h = np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) % 360.0
    return L, C, h


def _from_lch_v(L, C, h):
    hr = np.radians(h)
    return np.stack([L, C * np.cos(hr), C * np.sin(hr)], axis=-1)


def _endpoint_masks(space, L, M_dst_inv, n):
    """White/black returns shared by all three algorithms."""
    out = np.zeros((n, 3))
    white_mask = L >= 1.0
    black_mask = L <= 0.0
    if white_mask.any():
        white = np.clip(space.to_XYZ(np.array([1.0, 0, 0])) @ M_dst_inv.T, 0, 1)
        out[white_mask] = white
    return out, white_mask, black_mask


def binsearch_minde_v(xyz, M_dst, M_dst_inv, space,
                      jnd=0.02, epsilon=1e-4, gamut_eps=1e-6):
    xyz = np.atleast_2d(np.asarray(xyz, float))
    n = len(xyz)
    lab = space.from_XYZ(xyz)
    L, C, h = _to_lch_v(lab)
    out, wm, bm = _endpoint_masks(space, L, M_dst_inv, n)
    done = wm | bm

    def to_rgb(Lv, Cv, hv):
        return space.to_XYZ(_from_lch_v(Lv, Cv, hv)) @ M_dst_inv.T

    def delta(Lv, Cv, hv, rgb2):
        lab2 = space.from_XYZ(rgb2 @ M_dst.T)
        return np.linalg.norm(_from_lch_v(Lv, Cv, hv) - lab2, axis=-1)

    rgb = to_rgb(L, C, h)
    ing = np.all(rgb >= -gamut_eps, axis=1) & np.all(rgb <= 1 + gamut_eps, axis=1)
    m = ~done & ing
    out[m] = np.clip(rgb[m], 0, 1)
    done |= m

    clipped = np.clip(rgb, 0, 1)
    E = delta(L, C, h, clipped)
    m = ~done & (E < jnd)
    out[m] = clipped[m]
    done |= m

    lo = np.zeros(n)
    hi = C.copy()
    lo_in = np.ones(n, bool)
    last_clipped = clipped.copy()
    active = ~done
    while True:
        work = active & (hi - lo > epsilon)
        if not work.any():
            break
        chroma = np.where(work, (lo + hi) / 2, 0.0)
        rgb_w = to_rgb(L[work], chroma[work], h[work])
        ing_w = (np.all(rgb_w >= -gamut_eps, axis=1)
                 & np.all(rgb_w <= 1 + gamut_eps, axis=1))
        idx = np.where(work)[0]
        # branch 1: min still in gamut and current in gamut -> lo = chroma
        b1 = lo_in[idx] & ing_w
        lo[idx[b1]] = chroma[idx[b1]]
        # branch 2: clip & compare
        b2 = ~b1
        if b2.any():
            i2 = idx[b2]
            cl = np.clip(rgb_w[b2], 0, 1)
            E2 = delta(L[i2], chroma[i2], h[i2], cl)
            last_clipped[i2] = cl
            near = E2 < jnd
            ret = near & (jnd - E2 < epsilon)
            out[i2[ret]] = cl[ret]
            active[i2[ret]] = False
            cont = near & ~ret
            lo_in[i2[cont]] = False
            lo[i2[cont]] = chroma[i2[cont]]
            far = ~near
            hi[i2[far]] = chroma[i2[far]]
    rem = active
    out[rem] = last_clipped[rem]
    return out


class _LUTArrays:
    """Vectorized view over EdgeSeekerLUT (same data, array lookups)."""
    def __init__(self, lut):
        self.h = np.array([x['h'] for x in lut.lut])
        self.l = np.array([x['l'] for x in lut.lut])
        self.c = np.array([x['c'] for x in lut.lut])
        self.k = np.array([x['curvature'] for x in lut.lut])

    def max_chroma(self, L, h):
        h = h % 360.0
        j = np.searchsorted(self.h, h)            # findClosest: lut[mid], lut[mid+1]
        j = np.clip(j, 1, len(self.h) - 1)
        lo, hi = j - 1, j
        exact = np.isclose(h, self.h[np.clip(j, 0, len(self.h) - 1)], rtol=0, atol=0)
        t = (h - self.h[lo]) / (self.h[hi] - self.h[lo])
        t = np.where(exact, 1.0, t)
        li = self.l[lo] * (1 - t) + self.l[hi] * t
        ci = self.c[lo] * (1 - t) + self.c[hi] * t
        ki = self.k[lo] * (1 - t) + self.k[hi] * t
        outside = (L <= 0) | (L >= 1)
        bottom = L <= li
        mc_bottom = np.where(li > 0, (L / np.where(li == 0, 1, li)) * ci, 0.0)
        x = (1 - L) / (1 - li)
        mc_top = ci * _arc_intersection_v(x, ki)
        return np.where(outside, 0.0, np.where(bottom, mc_bottom, mc_top))


def _arc_intersection_v(x, k):
    straight = k == 0
    ks = np.where(straight, 1.0, k)
    radius = np.abs(1.0 / ks)
    under_r = radius ** 2 - 0.5
    dist = np.sqrt(np.maximum(under_r, 0.0))
    off = dist / np.sqrt(2)
    cx = np.where(ks > 0, off, -off) + 0.5
    cy = np.where(ks > 0, -off, off) + 0.5
    under = radius ** 2 - (x - cx) ** 2
    s = np.sqrt(np.maximum(under, 0.0))
    r1 = cy + s
    res = np.where((r1 >= 0) & (r1 <= 1), r1, cy - s)
    res = np.where(under < 0, 0.0, res)
    return np.where(straight, x, res)


def edge_seeker_v(xyz, M_dst, M_dst_inv, space, lut):
    xyz = np.atleast_2d(np.asarray(xyz, float))
    n = len(xyz)
    arr = lut if isinstance(lut, _LUTArrays) else _LUTArrays(lut)
    L, C, h = _to_lch_v(space.from_XYZ(xyz))
    out, wm, bm = _endpoint_masks(space, L, M_dst_inv, n)
    mid = ~(wm | bm)
    Cc = np.minimum(C[mid], arr.max_chroma(L[mid], h[mid]))
    rgb = space.to_XYZ(_from_lch_v(L[mid], Cc, h[mid])) @ M_dst_inv.T
    out[mid] = np.clip(rgb, 0, 1)
    return out


def _cast_ray_v(start, end):
    """Vectorized slab method. Returns (hit_points, found_mask)."""
    d = end - start
    parallel = np.abs(d) < 1e-12
    safe_d = np.where(parallel, 1.0, d)
    t1 = (0.0 - start) / safe_d
    t2 = (1.0 - start) / safe_d
    t1m = np.where(parallel, -np.inf, np.minimum(t1, t2))
    t2m = np.where(parallel, np.inf, np.maximum(t1, t2))
    tnear = t1m.max(axis=1)
    tfar = t2m.min(axis=1)
    bad_parallel = (parallel & ((start < 0) | (start > 1))).any(axis=1)
    found = ~bad_parallel & ~(tnear > tfar) & ~(tfar < 0)
    tnear = np.where(tnear < 0, tfar, tnear)
    found &= np.isfinite(tnear)
    hit = start + d * tnear[:, None]
    return hit, found


def raytrace_v(xyz, M_dst, M_dst_inv, space, gamut_eps=1e-6):
    xyz = np.atleast_2d(np.asarray(xyz, float))
    n = len(xyz)
    lab = space.from_XYZ(xyz)
    L, C, h = _to_lch_v(lab)
    out, wm, bm = _endpoint_masks(space, L, M_dst_inv, n)
    mid = ~(wm | bm)
    idx = np.where(mid)[0]
    if len(idx):
        Lm, hm = L[idx], h[idx]
        anchor = space.to_XYZ(_from_lch_v(Lm, np.zeros(len(idx)), hm)) @ M_dst_inv.T
        rgb = space.to_XYZ(lab[idx]) @ M_dst_inv.T
        ing = (np.all(rgb >= -gamut_eps, axis=1)
               & np.all(rgb <= 1 + gamut_eps, axis=1))
        act = ~ing
        if act.any():
            low, high = 1e-6, 1.0 - 1e-6
            cur = rgb[act].copy()
            anc = anchor[act].copy()
            La, ha = Lm[act], hm[act]
            last = cur.copy()
            alive = np.ones(len(cur), bool)
            for i in range(4):
                if i > 0:
                    lch_c = _to_lch_v(space.from_XYZ(cur @ M_dst.T))
                    cur = np.where(alive[:, None],
                                   space.to_XYZ(_from_lch_v(La, lch_c[1], ha))
                                   @ M_dst_inv.T, cur)
                hit, found = _cast_ray_v(anc, cur)
                lost = alive & ~found
                cur[lost] = last[lost]
                alive &= found
                if i > 0:
                    upd = alive & np.all(cur > low, axis=1) & np.all(cur < high, axis=1)
                    anc[upd] = cur[upd]
                cur = np.where((alive & found)[:, None], hit, cur)
                last = np.where((alive & found)[:, None], hit, last)
            rgb[act] = cur
        out[idx] = np.clip(rgb, 0, 1)
    return out


VECTORIZED = {'binsearch_minde': binsearch_minde_v,
              'edge_seeker': edge_seeker_v,
              'raytrace': raytrace_v}
