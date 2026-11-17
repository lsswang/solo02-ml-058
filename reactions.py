"""Ordinary net-stoichiometric integer planning and individual reaction replay."""
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint

def fire(problem, stock, reaction_index):
    reaction = problem["reactions"][reaction_index]
    if any(s < c for s, c in zip(stock, reaction["consume"])):
        return None
    after = [s-c+p for s, c, p in
             zip(stock, reaction["consume"], reaction["produce"])]
    if any(s < 0 or s > cap for s, cap in zip(after, problem["capacity"])):
        return None
    return after

def net_plan(problem, target, quantity=1):
    initial = np.asarray(problem["initial"], dtype=float)
    net = np.asarray([[p-c for p, c in zip(r["produce"], r["consume"])]
                      for r in problem["reactions"]], dtype=float).T
    lower = np.zeros(len(initial))
    lower[problem["species"].index(target)] = quantity
    matrix = np.vstack([net, np.ones((1, net.shape[1]))])
    result = milp(
        np.asarray([r["cost"] for r in problem["reactions"]]),
        integrality=np.ones(net.shape[1]), bounds=Bounds(0, problem["step_bound"]),
        constraints=LinearConstraint(
            matrix, np.r_[lower-initial, 0],
            np.r_[np.asarray(problem["capacity"])-initial, problem["step_bound"]]))
    if result.x is None:
        return {"status": "infeasible"}
    counts = np.rint(result.x).astype(int)
    return {"status": "ok", "target": target, "cost": float(result.fun),
            "reaction_counts": counts.tolist(),
            "final_stock": (initial + net @ counts).astype(int).tolist()}

