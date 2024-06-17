from typing import Optional

import redun

from launch_sweep import launch_sweep


@redun.task(cache=False)
def launch_all_train_sweeps(
    num_agents_per_sweep: Optional[int] = None,
    jobs_per_agent: Optional[int] = None,
):
    config_paths = redun.file.glob_file(
        "paper/experiments/norman_data_efficiency/configs/*.yaml"
    )
    out = []
    for path in config_paths:
        ret = launch_sweep(
            config_path=path,
            num_agents=num_agents_per_sweep,
            jobs_per_agent=jobs_per_agent,
            executor="sweep_agent_batch_large",
        )
        out.append(ret)
    return out
