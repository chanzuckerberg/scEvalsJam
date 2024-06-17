import os
from os.path import join
from typing import List, Literal, Optional

import numpy as np
import redun
import yaml

import wandb
from eval import evaluate_wandb_experiment


@redun.task(cache=False)
def launch_sweep(
    config_path: str,
    num_agents: Optional[int] = None,
    jobs_per_agent: Optional[int] = None,
    executor: str = "sweep_agent",
):
    assert (num_agents is None) != (
        jobs_per_agent is None
    ), "Must specify exactly one of num_agents or jobs_per_agent"

    # get wandb login info
    api = wandb.Api()
    host = api.settings["base_url"]
    entity = api.default_entity
    api_key = api.api_key

    sweep_path = initialize_sweep(config_path, entity)

    if num_agents is None:
        assert jobs_per_agent is not None
        sweep = api.sweep(sweep_path)
        num_jobs = sweep.expected_run_count
        num_agents = int(np.ceil(float(num_jobs) / jobs_per_agent))

    agent_results = []
    for i in range(num_agents):
        if executor == "sweep_agent":
            ret = launch_agent_local(sweep_path, host, api_key, agent_idx=i)
        elif executor == "sweep_agent_batch":
            ret = launch_agent_batch(sweep_path, host, api_key, agent_idx=i)
        else:
            ret = launch_agent_batch_large(sweep_path, host, api_key, agent_idx=i)
        agent_results.append(ret)

    return agent_results


def initialize_sweep(config_path: str, entity: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    sweep_id = wandb.sweep(config)
    sweep_path = join(entity, config["project"], sweep_id)
    return sweep_path


@redun.task(cache=False, executor="sweep_agent")
def launch_agent_local(sweep_path: str, host: str, api_key: str, agent_idx: int):
    cmd = f"wandb login --host {host} {api_key}; wandb agent {sweep_path}"
    return os.system(cmd)


@redun.task(cache=False, executor="sweep_agent_batch")
def launch_agent_batch(sweep_path: str, host: str, api_key: str, agent_idx: int):
    cmd = f"wandb login --host {host} {api_key}; wandb agent {sweep_path}"
    return os.system(cmd)


@redun.task(cache=False, executor="sweep_agent_batch_large")
def launch_agent_batch_large(sweep_path: str, host: str, api_key: str, agent_idx: int):
    cmd = f"wandb login --host {host} {api_key}; wandb agent {sweep_path}"
    return os.system(cmd)


@redun.task(cache=False)
def evaluate_sweep(
    sweep_id: str,
    jobs_per_agent: int,
    perturbseq: int,
    batch_size: int = 128,
    ate_n_particles: int = 2500,
    executor: str = "sweep_agent",
):
    api = wandb.Api()
    sweep = api.sweep(sweep_id)

    host = api.settings["base_url"]
    api_key = api.api_key

    run_paths = ["/".join(run.path) for run in sweep.runs]
    print(len(run_paths))

    if executor == "sweep_agent":
        evaluate_fn = evaluate_run_list_local
    elif executor == "sweep_agent_batch":
        evaluate_fn = evaluate_run_list_batch
    else:
        evaluate_fn = evaluate_run_list_batch_large

    ret = []
    for i in range(0, len(run_paths), jobs_per_agent):
        ret.append(
            evaluate_fn(
                run_path_list=run_paths[i : i + jobs_per_agent],
                perturbseq=perturbseq,
                batch_size=batch_size,
                ate_n_particles=ate_n_particles,
                host=host,
                api_key=api_key,
            )
        )
    return ret


@redun.task(cache=False, executor="sweep_agent")
def evaluate_run_list_local(
    run_path_list: List[str],
    perturbseq: int,
    host: str,
    api_key: str,
    batch_size: int = 128,
    ate_n_particles: int = 2500,
):
    return evaluate_run_list(
        run_path_list=run_path_list,
        perturbseq=perturbseq,
        host=host,
        api_key=api_key,
        batch_size=batch_size,
        ate_n_particles=ate_n_particles,
    )


@redun.task(cache=False, executor="sweep_agent_batch")
def evaluate_run_list_batch(
    run_path_list: List[str],
    perturbseq: int,
    host: str,
    api_key: str,
    batch_size: int = 128,
    ate_n_particles: int = 2500,
):
    return evaluate_run_list(
        run_path_list=run_path_list,
        perturbseq=perturbseq,
        host=host,
        api_key=api_key,
        batch_size=batch_size,
        ate_n_particles=ate_n_particles,
    )


@redun.task(cache=False, executor="sweep_agent_batch_large")
def evaluate_run_list_batch_large(
    run_path_list: List[str],
    perturbseq: int,
    host: str,
    api_key: str,
    batch_size: int = 128,
    ate_n_particles: int = 2500,
):
    return evaluate_run_list(
        run_path_list=run_path_list,
        perturbseq=perturbseq,
        host=host,
        api_key=api_key,
        batch_size=batch_size,
        ate_n_particles=ate_n_particles,
    )


def evaluate_run_list(
    run_path_list: List[str],
    perturbseq: int,
    host: str,
    api_key: str,
    batch_size: int = 128,
    ate_n_particles: int = 2500,
):
    wandb.login(host=host, key=api_key)
    ate_method: Literal["perturbseq", "mean"]
    if perturbseq:
        ate_method = "perturbseq"
    else:
        ate_method = "mean"

    for run_path in run_path_list:
        evaluate_wandb_experiment(
            experiment_path=run_path,
            average_treatment_effect_method=ate_method,
            batch_size=batch_size,
            ate_n_particles=ate_n_particles,
        )
    return 0
