"""Independent bond evidence, before global structural conditioning."""
import numpy as np
from scipy.special import logsumexp

def independent(model):
    scores = np.asarray(model["log_potentials"], dtype=float)
    z = logsumexp(scores, axis=1)
    return {"log_partition": float(z.sum()),
            "bond_marginals": np.exp(scores - z[:, None]).tolist(),
            "map_orders": scores.argmax(axis=1).tolist()}

def joint_log_weight(model, orders):
    return float(sum(model["log_potentials"][i][order]
                     for i, order in enumerate(orders)))

