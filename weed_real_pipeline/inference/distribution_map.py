"""
Weed distribution map generator.

Takes the list of WeedInstance objects produced by the inference pipeline
and generates publication-quality spatial distribution maps:

  1. density_heatmap.png    – Gaussian KDE over all detection centres,
                              shows overall weed pressure across the field.

  2. category_map.png       – Per-species dot map. Each detected weed is
                              plotted as a coloured circle. A legend lists
                              species and count.

  3. per_species_heatmaps/  – One heatmap per species (only for species
                              with ≥ 3 detections).

  4. summary_stats.txt      – Detection counts and spatial statistics.

Coordinate system
─────────────────
The (cx, cy) stored in WeedInstance are normalised to [0, 1] relative
to each source image. For a multi-image field survey the images need to
be georeferenced. For this demo all images are tiled into a virtual
field grid so that each image occupies one cell in an NxM mosaic.
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Optional, Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from inference.inference_pipeline import WeedInstance, ImageResult

WEED_CLASSES: list = []
CLASS_COLORS_BGR: dict = {}


# ── Colour helpers ────────────────────────────────────────────────────────────

def bgr_to_rgb_f(bgr: Tuple[int,int,int]) -> Tuple[float,float,float]:
    return bgr[2]/255, bgr[1]/255, bgr[0]/255


CLASS_RGB_F: Dict[str, Tuple[float,float,float]] = {
    name: bgr_to_rgb_f(bgr)
    for name, bgr in CLASS_COLORS_BGR.items()
}

GREEN_HEATMAP = LinearSegmentedColormap.from_list(
    "weed_heat", ["#1a1a2e", "#16213e", "#0f3460", "#53bd8c", "#f5f0c8"]
)


# ── Mosaic layout ─────────────────────────────────────────────────────────────

def _tile_layout(n_images: int) -> Tuple[int, int]:
    """Return (n_cols, n_rows) for a roughly square tile grid."""
    n_cols = int(np.ceil(np.sqrt(n_images)))
    n_rows = int(np.ceil(n_images / n_cols))
    return n_cols, n_rows


def _global_coords(
    instances: List[WeedInstance],
    image_ids: List[str],
    n_cols: int, n_rows: int,
) -> List[Tuple[float, float, WeedInstance]]:
    """
    Map per-image normalised (cx, cy) → global field coordinates in [0, 1]².

    Each image occupies a 1/n_cols × 1/n_rows cell in the virtual field.
    """
    id_to_cell: Dict[str, Tuple[int, int]] = {}
    for idx, img_id in enumerate(image_ids):
        col = idx % n_cols
        row = idx // n_cols
        id_to_cell[img_id] = (col, row)

    coords = []
    for inst in instances:
        col, row = id_to_cell.get(inst.image_id, (0, 0))
        gx = (col + inst.center_xy[0]) / n_cols
        gy = (row + inst.center_xy[1]) / n_rows
        coords.append((gx, gy, inst))
    return coords


# ── Map generators ────────────────────────────────────────────────────────────

class WeedDistributionMapper:

    def __init__(self, class_names: List[str] = WEED_CLASSES, grid_px: int = 256):
        self.class_names = class_names
        self.grid_px     = grid_px

    # ── 1. Overall density heatmap ────────────────────────────────────────────

    def density_heatmap(
        self,
        global_coords: List[Tuple[float, float, WeedInstance]],
        output_path: str,
        sigma: float = 10.0,
    ):
        G = self.grid_px
        density = np.zeros((G, G), dtype=np.float32)

        for gx, gy, inst in global_coords:
            px = int(np.clip(gx * G, 0, G - 1))
            py = int(np.clip(gy * G, 0, G - 1))
            density[py, px] += inst.confidence

        density_smooth = gaussian_filter(density, sigma=sigma)

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(
            density_smooth, origin="upper", cmap=GREEN_HEATMAP,
            extent=[0, 1, 1, 0], interpolation="bilinear"
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Weighted detection density", fontsize=11)

        ax.set_title("Overall Weed Density Heatmap", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Field X (normalised)", fontsize=11)
        ax.set_ylabel("Field Y (normalised)", fontsize=11)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.tick_params(labelsize=9)
        ax.grid(color="white", alpha=0.15, linewidth=0.5)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Map] Density heatmap → {output_path}")

    # ── 2. Per-species category dot map ──────────────────────────────────────

    def category_map(
        self,
        global_coords: List[Tuple[float, float, WeedInstance]],
        output_path: str,
    ):
        counts = Counter(inst.class_name for _, _, inst in global_coords)
        present_classes = [c for c in self.class_names if counts.get(c, 0) > 0]

        fig, ax = plt.subplots(figsize=(9, 8))
        ax.set_facecolor("#1a1a1a")
        fig.patch.set_facecolor("#111111")

        # Background field outline
        field_rect = mpatches.FancyBboxPatch(
            (0, 0), 1, 1, linewidth=1.5, edgecolor="#888888",
            facecolor="none", boxstyle="round,pad=0.01"
        )
        ax.add_patch(field_rect)

        for gx, gy, inst in global_coords:
            color = CLASS_RGB_F.get(inst.class_name, (0, 1, 0))
            radius = 0.006 + inst.confidence * 0.008
            circle = plt.Circle((gx, gy), radius, color=color,
                                 alpha=0.75, linewidth=0)
            ax.add_patch(circle)

        # Legend
        legend_patches = []
        for cls_name in present_classes:
            color = CLASS_RGB_F.get(cls_name, (0, 1, 0))
            cnt   = counts[cls_name]
            patch = mpatches.Patch(color=color, label=f"{cls_name} (n={cnt})")
            legend_patches.append(patch)

        legend = ax.legend(
            handles=legend_patches, loc="upper right",
            facecolor="#222222", edgecolor="#555555",
            labelcolor="white", fontsize=8,
            title="Weed Species", title_fontsize=9,
        )
        legend.get_title().set_color("white")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("Weed Species Distribution Map", fontsize=14,
                     fontweight="bold", color="white", pad=12)
        ax.set_xlabel("Field X (normalised)", fontsize=11, color="#cccccc")
        ax.set_ylabel("Field Y (normalised)", fontsize=11, color="#cccccc")
        ax.tick_params(colors="#aaaaaa", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#444444")

        total = len(global_coords)
        ax.text(0.02, 0.97, f"Total detections: {total}",
                transform=ax.transAxes, fontsize=9, color="#dddddd",
                verticalalignment="top")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"  [Map] Category map → {output_path}")

    # ── 3. Per-species heatmaps ───────────────────────────────────────────────

    def per_species_heatmaps(
        self,
        global_coords: List[Tuple[float, float, WeedInstance]],
        output_dir: str,
        min_detections: int = 3,
        sigma: float = 8.0,
    ):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        G = self.grid_px

        by_species: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)
        for gx, gy, inst in global_coords:
            by_species[inst.class_name].append((gx, gy, inst.confidence))

        for cls_name, pts in by_species.items():
            if len(pts) < min_detections:
                continue

            density = np.zeros((G, G), dtype=np.float32)
            for gx, gy, conf in pts:
                px = int(np.clip(gx * G, 0, G - 1))
                py = int(np.clip(gy * G, 0, G - 1))
                density[py, px] += conf
            density_smooth = gaussian_filter(density, sigma=sigma)

            color = CLASS_RGB_F.get(cls_name, (0, 0.8, 0))
            cmap  = LinearSegmentedColormap.from_list(
                "sp", ["#0a0a0a", color]
            )

            fig, ax = plt.subplots(figsize=(6, 5.5))
            ax.imshow(density_smooth, origin="upper", cmap=cmap,
                      extent=[0, 1, 1, 0], interpolation="bilinear")
            ax.set_title(f"{cls_name}\n(n={len(pts)} detections)",
                         fontsize=12, fontweight="bold", pad=10)
            ax.set_xlabel("Field X", fontsize=10)
            ax.set_ylabel("Field Y", fontsize=10)
            plt.tight_layout()

            safe_name = cls_name.replace(" ", "_")
            out_path  = str(Path(output_dir) / f"{safe_name}_heatmap.png")
            plt.savefig(out_path, dpi=130, bbox_inches="tight")
            plt.close()
            print(f"  [Map] {cls_name} heatmap → {out_path}")

    # ── 4. Summary statistics ─────────────────────────────────────────────────

    def summary_stats(
        self,
        global_coords: List[Tuple[float, float, WeedInstance]],
        n_images: int,
        output_path: str,
    ):
        counts   = Counter(inst.class_name for _, _, inst in global_coords)
        total    = len(global_coords)
        conf_arr = np.array([inst.confidence for _, _, inst in global_coords])

        lines = [
            "=" * 50,
            "  WEED DISTRIBUTION SUMMARY",
            "=" * 50,
            f"  Images processed : {n_images}",
            f"  Total detections : {total}",
            f"  Avg confidence   : {conf_arr.mean():.3f}" if total else "  Avg confidence   : N/A",
            "",
            "  Species breakdown:",
        ]
        for cls_name in self.class_names:
            cnt = counts.get(cls_name, 0)
            pct = cnt / total * 100 if total else 0.0
            lines.append(f"    {cls_name:<25}  {cnt:4d}  ({pct:5.1f}%)")

        if total > 0:
            xs = np.array([gx for gx, _, _ in global_coords])
            ys = np.array([gy for _, gy, _ in global_coords])
            lines += [
                "",
                "  Spatial statistics (normalised field coords):",
                f"    Centroid X : {xs.mean():.3f} ± {xs.std():.3f}",
                f"    Centroid Y : {ys.mean():.3f} ± {ys.std():.3f}",
                f"    Hot-spot X : {xs[xs.size//2]:.3f}",
                f"    Hot-spot Y : {ys[ys.size//2]:.3f}",
            ]

        lines.append("=" * 50)
        text = "\n".join(lines)
        print(text)
        Path(output_path).write_text(text)
        print(f"  [Map] Summary stats → {output_path}")

    # ── 5. Generate all maps ──────────────────────────────────────────────────

    def generate_all(
        self,
        results: List[ImageResult],
        output_dir: str,
    ):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        all_instances = [inst for r in results for inst in r.instances]
        image_ids     = [r.image_id for r in results]

        if not all_instances:
            print("[Map] No detections found – skipping map generation.")
            (out / "summary_stats.txt").write_text("No detections.\n")
            return

        n_cols, n_rows = _tile_layout(len(results))
        global_coords  = _global_coords(all_instances, image_ids, n_cols, n_rows)

        print(f"\n[Map] Generating distribution maps for {len(all_instances)} detections…")
        self.density_heatmap(global_coords, str(out / "density_heatmap.png"))
        self.category_map(global_coords,   str(out / "category_map.png"))
        self.per_species_heatmaps(global_coords, str(out / "per_species_heatmaps"))
        self.summary_stats(global_coords,  len(results), str(out / "summary_stats.txt"))
        print(f"\n[Map] All maps saved to: {out}")


# ── CLI standalone ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.sample_generator import generate_dataset, WEED_CLASSES

    print("[Demo] Generating demo dataset…")
    root = Path("outputs/sample_dataset")
    generate_dataset(str(root))

    # Build fake ImageResults for demo (no trained models needed)
    rng = np.random.default_rng(42)
    demo_results: List[ImageResult] = []
    for img_idx in range(20):
        img_id = f"test_{img_idx:04d}"
        instances = []
        n = rng.integers(2, 8)
        for _ in range(n):
            cls_name = rng.choice(WEED_CLASSES[:-1])
            cls_id   = WEED_CLASSES.index(cls_name)
            conf     = float(rng.uniform(0.4, 0.95))
            cx, cy   = float(rng.uniform(0.05, 0.95)), float(rng.uniform(0.05, 0.95))
            instances.append(WeedInstance(
                image_id  = img_id,
                class_name= cls_name,
                class_id  = cls_id,
                confidence= conf,
                box_xyxy  = (100, 100, 200, 200),
                center_xy = (cx, cy),
                proba     = np.zeros(len(WEED_CLASSES)),
            ))
        demo_results.append(ImageResult(img_id, (640, 640), instances))

    mapper = WeedDistributionMapper(class_names=WEED_CLASSES)
    mapper.generate_all(demo_results, output_dir="outputs/demo_maps")
    print("\n[Demo] Done! Check outputs/demo_maps/")
