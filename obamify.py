"""obamify.py -- turn any image into Obama with a numpy genetic algorithm.

Python port of the `process_genetic` core from Spu7Nix/obamify (the Rust
+wgpu tool behind the Geometry Dash "obamify" video). Every pixel of your
image is a gene; the optimizer keeps swapping pixels until the population
spells Obama. Fully deterministic given a seed.

Usage:
    python obamify.py input.jpg output.png
    python obamify.py input.jpg output.png --side 256 --seed 7 --quiet

Requires: numpy, pillow  (see requirements.txt)
"""

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

# --- algorithm constants, copied exactly from the Rust original (mod.rs) ---
SWAPS_PER_GENERATION_PER_PIXEL = 128   # 2,097,152 swap attempts per generation at side=128
PROXIMITY_IMPORTANCE = 13              # spatial weight from util.rs
DECAY = 0.99                           # max_dist decay per generation (floor 2, int truncation)
CONVERGE_MAX_DIST = 4
CONVERGE_SWAPS = 10
SAFETY_CAP = 2000                      # generations; original relies on convergence only
BATCH = 4096                           # vectorized lane size (implementation detail)


def resize_center(img, sidelen):
    """CropScale::identity() -- center square crop, then Lanczos3."""
    w, h = img.size
    base = min(w, h)
    x0, y0 = (w - base) // 2, (h - base) // 2
    return img.crop((x0, y0, x0 + base, y0 + base)).resize(
        (sidelen, sidelen), Image.LANCZOS
    )


def obamify(src_img, tgt_img, w_img, seed=12345, save_every=0, quiet=False,
            progress_cb=None):
    """Run the genetic rearrangement. Returns (PIL.Image, generations).

    src_img/tgt_img/w_img: PIL RGB images, any size (center-cropped to side).
    The weight image's red channel sets per-slot color importance; background
    pixels in the bundled weights256.png are 1, face pixels are 255.
    """
    sidelen = tgt_img.size[0]
    n = sidelen * sidelen

    tgt = np.asarray(tgt_img, dtype=np.int64).reshape(n, 3)
    TR, TG, TB = tgt[:, 0].copy(), tgt[:, 1].copy(), tgt[:, 2].copy()
    W = np.asarray(w_img, dtype=np.int64)[:, :, 0].reshape(n)

    xs, ys = np.meshgrid(np.arange(sidelen), np.arange(sidelen))
    TX = xs.reshape(n)  # slot i's target position
    TY = ys.reshape(n)

    # per-slot state: source pixel identity (sx, sy) + its color
    SX = TX.copy()
    SY = TY.copy()
    src = np.asarray(src_img, dtype=np.int64).reshape(n, 3)
    SR = src[:, 0].copy()
    SG = src[:, 1].copy()
    SB = src[:, 2].copy()

    def heur(sr, sg, sb, sx, sy, slot):
        """Heuristic of pixel (rgb, source-coords) placed at slot.

        color_sq * color_weight + (spatial * 13)^2   -- exact Rust formula.
        """
        dx = sx - TX[slot]
        dy = sy - TY[slot]
        spatial = (dx * dx + dy * dy) * PROXIMITY_IMPORTANCE
        dr = sr - TR[slot]
        dg = sg - TG[slot]
        db = sb - TB[slot]
        return (dr * dr + dg * dg + db * db) * W[slot] + spatial * spatial

    # initial heuristics: pixel i at slot i (spatial = 0)
    H = ((SR - TR) ** 2 + (SG - TG) ** 2 + (SB - TB) ** 2) * W

    rng = np.random.default_rng(seed)
    max_dist = float(sidelen)
    gen = 0
    t0 = time.time()

    while True:
        swaps = 0
        md = int(max_dist)
        attempts = SWAPS_PER_GENERATION_PER_PIXEL * n
        done = 0
        while done < attempts:
            m = min(BATCH, attempts - done)
            apos = rng.integers(0, n, size=m)
            bx = np.clip(TX[apos] + rng.integers(-md, md + 1, size=m), 0, sidelen - 1)
            by = np.clip(TY[apos] + rng.integers(-md, md + 1, size=m), 0, sidelen - 1)
            bpos = by * sidelen + bx

            a_on_b = heur(SR[apos], SG[apos], SB[apos], SX[apos], SY[apos], bpos)
            b_on_a = heur(SR[bpos], SG[bpos], SB[bpos], SX[bpos], SY[bpos], apos)

            ok = (H[apos] + H[bpos] - a_on_b - b_on_a) > 0

            ap_ok, bp_ok = apos[ok], bpos[ok]
            ba, bb = a_on_b[ok], b_on_a[ok]
            touched = np.zeros(n, dtype=bool)
            # fold accepted swaps in sequentially; a lane that touches a slot
            # already swapped this batch re-evaluates against current state so
            # results are identical to the scalar loop
            for k in range(len(ap_ok)):
                a, b = int(ap_ok[k]), int(bp_ok[k])
                if touched[a] or touched[b]:
                    ha = heur(SR[b], SG[b], SB[b], SX[b], SY[b], a)
                    hb = heur(SR[a], SG[a], SB[a], SX[a], SY[a], b)
                    if H[a] + H[b] - ha - hb > 0:
                        SR[a], SR[b] = SR[b], SR[a]
                        SG[a], SG[b] = SG[b], SG[a]
                        SB[a], SB[b] = SB[b], SB[a]
                        SX[a], SX[b] = SX[b], SX[a]
                        SY[a], SY[b] = SY[b], SY[a]
                        H[a], H[b] = ha, hb
                        swaps += 1
                    touched[a] = touched[b] = True
                    continue
                SR[a], SR[b] = SR[b], SR[a]
                SG[a], SG[b] = SG[b], SG[a]
                SB[a], SB[b] = SB[b], SB[a]
                SX[a], SX[b] = SX[b], SX[a]
                SY[a], SY[b] = SY[b], SY[a]
                H[a], H[b] = bb[k], ba[k]
                touched[a] = touched[b] = True
                swaps += 1
            done += m

        gen += 1
        if max_dist < CONVERGE_MAX_DIST and swaps < CONVERGE_SWAPS:
            if not quiet:
                print(f"converged at generation {gen} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            break
        if not quiet and (gen % 50 == 0 or gen == 1):
            print(f"gen {gen:4d}: swaps={swaps:7,} max_dist={md:3d} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if progress_cb:
            progress_cb(gen, swaps, md)
        if save_every and gen % save_every == 0:
            _render(SR, SG, SB, SX, SY, sidelen, src,
                    f"obamify_gen_{gen:04d}.png")
        max_dist = max(2.0, max_dist * DECAY)
        if gen >= SAFETY_CAP:
            print("safety cap reached", file=sys.stderr, flush=True)
            break

    return _render(SR, SG, SB, SX, SY, sidelen, src), gen


def _render(SR, SG, SB, SX, SY, sidelen, src, path=None):
    """Slot i displays the source pixel that currently lives there.

    v4/Rust semantics: look up the pixel's ORIGINAL color via its source
    coordinates. (SR/SG/SB are slot-indexed current colors; indexing them
    with the assignment map applies the permutation twice and produces a
    subtly double-shuffled image.)
    """
    assignments = (SY * sidelen + SX).astype(np.int64)
    out = Image.new("RGB", (sidelen, sidelen))
    out.putdata([tuple(src[i]) for i in assignments])
    if path:
        out.save(path)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Turn any image into Obama with a numpy genetic algorithm."
    )
    ap.add_argument("input", help="source image (any size; center-cropped)")
    ap.add_argument("output", help="output png path")
    ap.add_argument("--side", type=int, default=128,
                    help="grid side length (default 128 = 16,384 pixels)")
    ap.add_argument("--seed", type=int, default=12345,
                    help="RNG seed; same seed + same input = same Obama")
    ap.add_argument("--save-every", type=int, default=0, metavar="N",
                    help="also save obamify_gen_NNNN.png every N generations")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    tgt_path = os.path.join(here, "assets", "target256.png")
    wgt_path = os.path.join(here, "assets", "weights256.png")
    if not (os.path.exists(tgt_path) and os.path.exists(wgt_path)):
        sys.exit("missing assets/target256.png or assets/weights256.png")

    src_img = resize_center(Image.open(args.input).convert("RGB"), args.side)
    tgt_img = resize_center(Image.open(tgt_path).convert("RGB"), args.side)
    w_img = resize_center(Image.open(wgt_path).convert("RGB"), args.side)

    out, _ = obamify(src_img, tgt_img, w_img, seed=args.seed,
                     save_every=args.save_every, quiet=args.quiet)
    out = out.resize((args.side * 4, args.side * 4), Image.NEAREST)
    out.save(args.output)
    if not args.quiet:
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
