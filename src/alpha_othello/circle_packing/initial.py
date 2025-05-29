# Inspired by https://github.com/codelion/openevolve/blob/main/examples/circle_packing/initial_program.py

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
    centers = np.zeros((26, 2))
    centers[0] = [0.5, 0.5]
    rings = [(9, 0.3), (16, 0.5)]
    i = 1
    for count, radius in rings:
        angle_step = 2 * np.pi / count
        for j in range(count):
            angle = i * angle_step
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            centers[i] = [x, y]
            i += 1
    centers = np.clip(centers, 0.01, 0.99)
    radii = compute_max_radii(centers)
    return [(x, y, r) for (x, y), r in zip(centers.tolist(), radii.tolist())]
