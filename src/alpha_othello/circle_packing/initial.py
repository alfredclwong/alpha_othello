# https://github.com/codelion/openevolve/blob/main/examples/circle_packing/initial_program.py

def pack_26() -> list[tuple[float, float, float]]:
    import numpy as np
    def compute_max_radii(centers):
        n = centers.shape[0]
        radii = np.ones(n)
        for i in range(n):
            x, y = centers[i]
            radii[i] = min(x, y, 1 - x, 1 - y)
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if radii[i] + radii[j] > dist:
                    scale = dist / (radii[i] + radii[j])
                    radii[i] *= scale
                    radii[j] *= scale
        return radii

    n = 26
    centers = np.zeros((n, 2))

    centers[0] = [0.5, 0.5]
    for i in range(8):
        angle = 2 * np.pi * i / 8
        centers[i + 1] = [0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)]
    for i in range(16):
        angle = 2 * np.pi * i / 16
        centers[i + 9] = [0.5 + 0.7 * np.cos(angle), 0.5 + 0.7 * np.sin(angle)]
    centers = np.clip(centers, 0.001, 0.999)
    radii = compute_max_radii(centers)
    return [(centers[i, 0].item(), centers[i, 1].item(), radii[i].item()) for i in range(n)]
