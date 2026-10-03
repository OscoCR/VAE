import csv
import numpy as np
from scipy.stats import wilcoxon




def load_errors(path):
    with open(path) as r:
        rows = list(csv.DictReader(r))

    columnas = [c for c in rows[0] if c not in ("index", "is_hard")]
    errors = {}

    for columna in columnas:
        errors[columna] = np.array([float(r[columna]) for r in rows])

    errors = {k: v for k, v in errors.items() if not np.isnan(v).any()}

    is_hard = np.array([bool(int(r["is_hard"])) for r in rows])

    return errors, is_hard

def relative_change (model_x, model_y):
    change = (np.mean(model_x) - np.mean(model_y)) / np.mean(model_y) * 100

    return change

def wilcoxon_p(model_x, model_y):
    # devuelve el p-value del Wilcoxon pareado entre los dos arreglos
    _, p_value = wilcoxon(model_x, model_y)
    return p_value

def bootstrap_ci(model_x, model_y, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(model_x)
    changes = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        changes.append(relative_change(model_x[idx], model_y[idx]))

    return np.percentile(changes, [2.5, 97.5])

def pct_improved(model_x, model_y):
    improved = np.sum(model_x < model_y)
    total = len(model_x)
    return improved / total * 100



if __name__ == "__main__":
    errors, is_hard = load_errors("/Users/osco/Downloads/test_per_image_mse.csv")
    groups = {"all": np.ones_like(is_hard, dtype=bool), "hard": is_hard, "rest": ~is_hard}

    for name, mask in groups.items():
        print(f"Group: {name}")
        for model_x in errors.keys():
            for model_y in errors.keys():
                if model_x == model_y:
                    continue
                change = relative_change(errors[model_x][mask], errors[model_y][mask])
                p_value = wilcoxon_p(errors[model_x][mask], errors[model_y][mask])
                bootstrap_ci_ = bootstrap_ci(errors[model_x][mask], errors[model_y][mask])
                pct_improved_ = pct_improved(errors[model_x][mask], errors[model_y][mask])
                print(f"Relative change from {model_y} to {model_x}: {change:.2f}% (p-value: {p_value:.3g}, 95% CI: [{bootstrap_ci_[0]:.2f}%, {bootstrap_ci_[1]:.2f}%], % improved: {pct_improved_:.2f}%)")