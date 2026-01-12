import os
import numpy as np


def to_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to uint8 [0,255]. Supports float [0,1] or [0,255]."""
    rgb = np.asarray(rgb)
    if rgb.dtype == np.uint8:
        return rgb
    rgb_f = rgb.astype(np.float32)
    if np.nanmax(rgb_f) <= 1.0:
        rgb_f *= 255.0
    rgb_f = np.clip(np.round(rgb_f), 0, 255)
    return rgb_f.astype(np.uint8)


def estimate_normals_pca(xyz: np.ndarray, k: int = 20) -> np.ndarray:
    """
    Estimate normals via local PCA on k-nearest neighbors.
    - Uses scipy.spatial.cKDTree if available; otherwise falls back to brute force.
    - Returns Nx3 float32 normals.
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    n = xyz.shape[0]
    if n < 10:
        # Too few points to estimate robust normals
        normals = np.zeros((n, 3), dtype=np.float32)
        normals[:, 2] = 1.0
        return normals

    k = int(max(5, min(k, n - 1)))

    # Build neighbor search
    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(xyz)
        _, idx = tree.query(xyz, k=k + 1)  # includes itself at idx[:,0]
        nbr_idx = idx[:, 1:]
    except Exception:
        # Brute force KNN (O(N^2)) - OK for ~2k points
        d2 = np.sum((xyz[:, None, :] - xyz[None, :, :]) ** 2, axis=-1)
        np.fill_diagonal(d2, np.inf)
        nbr_idx = np.argpartition(d2, kth=k, axis=1)[:, :k]

    normals = np.zeros((n, 3), dtype=np.float32)

    # PCA per point
    for i in range(n):
        pts = xyz[nbr_idx[i]]
        mu = pts.mean(axis=0, keepdims=True)
        X = pts - mu
        # Covariance
        C = (X.T @ X) / max(1, X.shape[0])
        # Smallest eigenvector is normal direction
        w, v = np.linalg.eigh(C)
        nrm = v[:, 0].astype(np.float32)
        # Normalize
        norm = np.linalg.norm(nrm) + 1e-12
        nrm = nrm / norm
        normals[i] = nrm

    # Optional: make normals more consistent (rough heuristic)
    # Align normals to point outward in a weak sense (use centroid direction)
    center = xyz.mean(axis=0)
    vec = xyz - center
    flip = (np.sum(normals * vec, axis=1) < 0)
    normals[flip] *= -1.0

    # Ensure no NaNs
    bad = ~np.isfinite(normals).all(axis=1)
    if bad.any():
        normals[bad] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    return normals.astype(np.float32)


def write_ascii_ply_xyz_nxyz_rgb(ply_path: str, xyz: np.ndarray, nxyz: np.ndarray, rgb_u8: np.ndarray) -> None:
    """
    Write ASCII PLY with vertex fields:
      x y z (float)
      nx ny nz (float)
      red green blue (uchar)
    This matches E-D3DGS loader expectations (no field missing).
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    nxyz = np.asarray(nxyz, dtype=np.float32)
    rgb_u8 = np.asarray(rgb_u8, dtype=np.uint8)

    n = xyz.shape[0]
    assert xyz.shape == (n, 3)
    assert nxyz.shape == (n, 3)
    assert rgb_u8.shape == (n, 3)

    header = "\n".join([
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ])

    with open(ply_path, "w") as f:
        f.write(header + "\n")
        for p, nn, c in zip(xyz, nxyz, rgb_u8):
            f.write(
                f"{p[0]} {p[1]} {p[2]} "
                f"{nn[0]} {nn[1]} {nn[2]} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )


def main():
    # Run inside: hypernerf/vrig-3dprinter/
    npy_path = "points.npy"
    out_ply = "points3D_downsample.ply"

    if not os.path.isfile(npy_path):
        raise FileNotFoundError(f"Cannot find {npy_path} in: {os.getcwd()}")

    pts = np.load(npy_path)
    if pts.ndim != 2 or pts.shape[1] not in (3, 6):
        raise ValueError(f"Unexpected points.npy shape: {pts.shape}. Expected Nx3 or Nx6.")

    xyz = pts[:, :3].astype(np.float32)

    # RGB: required by E-D3DGS loader
    if pts.shape[1] == 6:
        rgb = to_uint8_rgb(pts[:, 3:6])
    else:
        rgb = np.full((xyz.shape[0], 3), 127, dtype=np.uint8)  # neutral gray

    # Normals: estimate from geometry (more reasonable than constant normals)
    normals = estimate_normals_pca(xyz, k=20)

    write_ascii_ply_xyz_nxyz_rgb(out_ply, xyz, normals, rgb)

    print("[OK] Wrote:", os.path.abspath(out_ply))
    print("     points:", xyz.shape[0])
    print("     xyz min:", xyz.min(axis=0))
    print("     xyz max:", xyz.max(axis=0))
    print("     rgb unique (up to 5):", np.unique(rgb, axis=0)[:5])
    print("     normals sample:", normals[0])


if __name__ == "__main__":
    main()
