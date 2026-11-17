import json
from pathlib import Path
import graph_model
import reactions

if __name__ == "__main__":
    cases = json.loads((Path(__file__).parent / "fixtures/cases.json").read_text())
    print(json.dumps({"ordinary_bonds": graph_model.independent(cases["ordinary_graph"]),
                      "ordinary_net_plan": reactions.net_plan(
                          cases["ordinary_problem"], "A")}, indent=2))

