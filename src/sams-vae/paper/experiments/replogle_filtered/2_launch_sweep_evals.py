from typing import Optional

import redun
import yaml

from launch_sweep import evaluate_sweep


@redun.task(cache=False)
def launch_all_sweep_evals(
    jobs_per_agent: Optional[int] = None,
):
    with open("paper/experiments/replogle_filtered/sweep_ids.yaml") as f:
        sweep_ids = yaml.safe_load(f)

    out = []
    for name, sweep_id in sweep_ids.items():
        ret = evaluate_sweep(
            sweep_id=sweep_id,
            perturbseq=1,
            ate_n_particles=2500,
            jobs_per_agent=jobs_per_agent,
            executor="sweep_agent_batch",
        )
        out.append(ret)

    return out
