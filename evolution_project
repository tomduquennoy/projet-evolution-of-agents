# -*- coding: utf-8 -*-
"""
Neuroevolution on EvoGym — Course Project
==========================================
Tasks: Walker-v0 (easy) | Thrower-v0 (medium) | Climb-v2 (hard)
Budget: 5,000,000 steps total per task

Algorithms implemented:
  - (1+λ)-EA  (Genetic Algorithm baseline)
  - Evolution Strategy (ES) with CMA
  - CMA-ES via pycma (recommended for best performance)

Author: Generated for course project
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import evogym.envs
from evogym import sample_robot
from evogym.utils import get_full_connectivity
from tqdm import tqdm
import copy
import json
import csv
import os
import time
import argparse
from datetime import datetime

# ─────────────────────────────────────────────
# Robot morphology
# ─────────────────────────────────────────────

WALKER = np.array([
    [3, 3, 3, 3, 3],
    [3, 3, 3, 0, 3],
    [3, 3, 0, 3, 3],
    [3, 3, 0, 3, 3],
    [3, 3, 0, 3, 3]
])

# ─────────────────────────────────────────────
# Neural Network
# ─────────────────────────────────────────────

class Network(nn.Module):
    def __init__(self, n_in, h_size, n_out):
        super().__init__()
        self.fc1 = nn.Linear(n_in, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, n_out)
        self.n_out = n_out

    def reset(self):
        pass

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ─────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────

class Agent:
    def __init__(self, config, genes=None):
        self.config = config
        self.fitness = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Network(config["n_in"], config["h_size"], config["n_out"]
                             ).to(self.device).double()
        if genes is not None:
            self.genes = genes

    @property
    def genes(self):
        with torch.no_grad():
            vec = torch.nn.utils.parameters_to_vector(self.model.parameters())
        return vec.cpu().numpy().astype(np.float64)

    @genes.setter
    def genes(self, params):
        params = np.array(params, dtype=np.float64)
        assert len(params) == len(self.genes)
        torch.nn.utils.vector_to_parameters(
            torch.tensor(params, device=self.device, dtype=torch.float64),
            self.model.parameters()
        )
        self.fitness = None

    def act(self, obs):
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float64).unsqueeze(0).to(self.device)
            return self.model(x).cpu().numpy()[0]

    def genome_size(self):
        return len(self.genes)


# ─────────────────────────────────────────────
# Environment helpers
# ─────────────────────────────────────────────

def make_env(env_name, robot=None):
    if robot is None:
        env = gym.make(env_name)
    else:
        env = gym.make(env_name, body=robot)
    env.robot = robot
    return env


def get_cfg(env_name, robot=None, h_size=32):
    env = make_env(env_name, robot=robot)
    cfg = {
        "n_in": env.observation_space.shape[0],
        "h_size": h_size,
        "n_out": env.action_space.shape[0],
    }
    env.close()
    return cfg


def evaluate(agent, env, max_steps=500):
    obs, _ = env.reset()
    agent.model.reset()
    total_reward = 0.0
    steps = 0
    done = False
    while not done and steps < max_steps:
        action = agent.act(obs)
        obs, r, done, trunc, _ = env.step(action)
        total_reward += r
        steps += 1
        if trunc:
            break
    return total_reward, steps


# ─────────────────────────────────────────────
# Budget tracker
# ─────────────────────────────────────────────

class BudgetTracker:
    """Tracks total steps used across the evolution."""
    def __init__(self, total_budget=5_000_000):
        self.total_budget = total_budget
        self.used = 0

    def consume(self, steps):
        self.used += steps

    @property
    def remaining(self):
        return self.total_budget - self.used

    @property
    def exhausted(self):
        return self.used >= self.total_budget

    def __str__(self):
        pct = 100 * self.used / self.total_budget
        return f"Steps: {self.used:,}/{self.total_budget:,} ({pct:.1f}%)"


# ─────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────

class Logger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["generation", "total_steps", "best_fitness",
                             "mean_fitness", "std_fitness", "wall_time_s"])
        self.start_time = time.time()

    def log(self, gen, total_steps, best, mean, std):
        elapsed = time.time() - self.start_time
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([gen, total_steps, best, mean, std, f"{elapsed:.1f}"])


# ─────────────────────────────────────────────
# (1+λ)-EA
# ─────────────────────────────────────────────

def one_plus_lambda(config, run_id=0, log_dir="logs"):
    """Simple (1+lambda) EA with Gaussian mutation."""
    cfg = {**config, **get_cfg(config["env_name"], robot=config["robot"],
                                h_size=config.get("h_size", 32))}
    budget = BudgetTracker(config.get("budget", 5_000_000))
    tag = f"{config['env_name']}_1pluslambda_run{run_id}"
    logger = Logger(f"{log_dir}/{tag}.csv")

    env = make_env(cfg["env_name"], robot=cfg["robot"])
    elite = Agent(cfg)
    elite.fitness, steps = evaluate(elite, env, max_steps=cfg["max_steps"])
    budget.consume(steps)

    fits_history = []
    steps_history = []
    t = tqdm(range(cfg["generations"]), desc=tag)

    for gen in t:
        if budget.exhausted:
            print(f"\n⚠ Budget exhausted at generation {gen}")
            break

        population = []
        pop_fitness = []

        for _ in range(cfg["lambda"]):
            if budget.exhausted:
                break
            # Gaussian mutation
            sigma = cfg.get("sigma", 0.1)
            new_genes = elite.genes + np.random.randn(elite.genome_size()) * sigma
            child = Agent(cfg, genes=new_genes)
            fit, steps = evaluate(child, env, max_steps=cfg["max_steps"])
            budget.consume(steps)
            child.fitness = fit
            population.append(child)
            pop_fitness.append(fit)

        if pop_fitness:
            best_idx = np.argmax(pop_fitness)
            if pop_fitness[best_idx] > elite.fitness:
                elite.genes = population[best_idx].genes
                elite.fitness = pop_fitness[best_idx]

        fits_history.append(elite.fitness)
        steps_history.append(budget.used)
        logger.log(gen, budget.used, elite.fitness,
                   np.mean(pop_fitness) if pop_fitness else elite.fitness,
                   np.std(pop_fitness) if pop_fitness else 0)
        t.set_description(f"{tag} | best={elite.fitness:.2f} | {budget}")

    env.close()
    save_solution(elite, cfg, f"{log_dir}/{tag}_solution.json")
    return elite, fits_history, steps_history


# ─────────────────────────────────────────────
# Evolution Strategy (ES) — rank-based CMA-lite
# ─────────────────────────────────────────────

def ES(config, run_id=0, log_dir="logs"):
    """
    (μ/μ_w, λ)-ES with weighted recombination.
    This is the natural gradient ascent direction used in OpenAI ES.
    """
    cfg = {**config, **get_cfg(config["env_name"], robot=config["robot"],
                                h_size=config.get("h_size", 32))}
    budget = BudgetTracker(config.get("budget", 5_000_000))
    tag = f"{config['env_name']}_ES_run{run_id}"
    logger = Logger(f"{log_dir}/{tag}.csv")

    lam = cfg["lambda"]
    mu  = cfg["mu"]
    sigma = cfg.get("sigma", 0.1)
    lr   = cfg.get("lr", 0.1)

    # CMA-style log-rank weights
    w = np.array([np.log(mu + 0.5) - np.log(i) for i in range(1, mu + 1)])
    w /= w.sum()

    env = make_env(cfg["env_name"], robot=cfg["robot"])
    elite = Agent(cfg)
    elite.fitness = -np.inf
    theta = elite.genes.copy()
    d = len(theta)

    # Adaptive sigma (1/5 success rule variant)
    success_rate = 0.2
    c_adapt = 0.1

    fits_history = []
    steps_history = []
    t = tqdm(range(cfg["generations"]), desc=tag)

    for gen in t:
        if budget.exhausted:
            print(f"\n⚠ Budget exhausted at generation {gen}")
            break

        noise = np.random.randn(lam, d)
        pop_genes = theta + sigma * noise
        pop_fitness = []

        for i in range(lam):
            if budget.exhausted:
                pop_fitness.append(-np.inf)
                continue
            ind = Agent(cfg, genes=pop_genes[i])
            fit, steps = evaluate(ind, env, max_steps=cfg["max_steps"])
            budget.consume(steps)
            pop_fitness.append(fit)

        pop_fitness = np.array(pop_fitness)
        idx = np.argsort(-pop_fitness)  # highest first

        # Weighted recombination
        step = np.zeros(d)
        for i in range(mu):
            step += w[i] * noise[idx[i]]
        theta = theta + lr * sigma * step

        # 1/5 success rule for sigma adaptation
        successes = np.sum(pop_fitness[idx[:mu]] > elite.fitness) / mu
        sigma *= np.exp(c_adapt * (successes - success_rate))
        sigma = np.clip(sigma, 1e-5, 10.0)

        best_fit = pop_fitness[idx[0]]
        if best_fit > elite.fitness:
            elite.genes = pop_genes[idx[0]]
            elite.fitness = best_fit

        fits_history.append(elite.fitness)
        steps_history.append(budget.used)
        logger.log(gen, budget.used, elite.fitness,
                   float(np.mean(pop_fitness[pop_fitness > -np.inf])),
                   float(np.std(pop_fitness[pop_fitness > -np.inf])))
        t.set_description(f"{tag} | best={elite.fitness:.2f} | σ={sigma:.4f} | {budget}")

    env.close()
    save_solution(elite, cfg, f"{log_dir}/{tag}_solution.json")
    return elite, fits_history, steps_history


# ─────────────────────────────────────────────
# CMA-ES (via pycma library — best performance)
# ─────────────────────────────────────────────

def CMA_ES(config, run_id=0, log_dir="logs"):
    """
    CMA-ES using pycma library.
    Install: pip install cma
    Falls back to ES if cma not available.
    """
    try:
        import cma
    except ImportError:
        print("⚠ pycma not installed (pip install cma). Falling back to ES.")
        return ES(config, run_id=run_id, log_dir=log_dir)

    cfg = {**config, **get_cfg(config["env_name"], robot=config["robot"],
                                h_size=config.get("h_size", 32))}
    budget = BudgetTracker(config.get("budget", 5_000_000))
    tag = f"{config['env_name']}_CMAES_run{run_id}"
    logger = Logger(f"{log_dir}/{tag}.csv")

    env = make_env(cfg["env_name"], robot=cfg["robot"])

    # Initial solution
    init_agent = Agent(cfg)
    theta0 = init_agent.genes
    d = len(theta0)
    sigma0 = cfg.get("sigma", 0.5)

    # CMA-ES options
    opts = cma.CMAOptions()
    opts["maxfevals"] = config.get("budget", 5_000_000)
    opts["popsize"] = cfg.get("lambda", 20)
    opts["tolx"] = 1e-8
    opts["tolfun"] = 1e-8
    opts["verbose"] = -9  # silent
    opts["seed"] = cfg.get("seed", 42 + run_id)

    es_cma = cma.CMAEvolutionStrategy(theta0, sigma0, opts)

    elite_fitness = -np.inf
    elite_genes = theta0.copy()
    fits_history = []
    steps_history = []
    gen = 0

    bar = tqdm(total=config.get("generations", 500), desc=tag)

    while not es_cma.stop() and not budget.exhausted:
        solutions = es_cma.ask()
        fitnesses = []

        for genes in solutions:
            if budget.exhausted:
                fitnesses.append(np.inf)  # CMA minimizes
                continue
            ind = Agent(cfg, genes=genes)
            fit, steps = evaluate(ind, env, max_steps=cfg["max_steps"])
            budget.consume(steps)
            fitnesses.append(-fit)  # negate: CMA minimizes

        es_cma.tell(solutions, fitnesses)

        best_fit = -min(fitnesses)
        mean_fit = -np.mean(fitnesses)
        std_fit  = np.std(fitnesses)

        if best_fit > elite_fitness:
            elite_fitness = best_fit
            elite_genes = solutions[np.argmin(fitnesses)].copy()

        fits_history.append(elite_fitness)
        steps_history.append(budget.used)
        logger.log(gen, budget.used, elite_fitness, mean_fit, std_fit)
        bar.set_description(f"{tag} | best={elite_fitness:.2f} | σ={es_cma.sigma:.4f} | {budget}")
        bar.update(1)
        gen += 1

    bar.close()
    env.close()

    elite = Agent(cfg, genes=elite_genes)
    elite.fitness = elite_fitness
    save_solution(elite, cfg, f"{log_dir}/{tag}_solution.json")
    return elite, fits_history, steps_history


# ─────────────────────────────────────────────
# Save / Load
# ─────────────────────────────────────────────

def save_solution(agent, cfg, path="solution.json"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {
        "env_name": cfg["env_name"],
        "robot": cfg["robot"].tolist(),
        "n_in": cfg["n_in"],
        "h_size": cfg["h_size"],
        "n_out": cfg["n_out"],
        "genes": agent.genes.tolist(),
        "fitness": float(agent.fitness) if agent.fitness is not None else None,
        "timestamp": datetime.now().isoformat(),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✓ Saved solution to {path}  (fitness={data['fitness']:.3f})")


def load_solution(path):
    with open(path) as f:
        data = json.load(f)
    data["robot"] = np.array(data["robot"])
    data["genes"] = np.array(data["genes"])
    cfg = {k: data[k] for k in ["env_name", "robot", "n_in", "h_size", "n_out"]}
    agent = Agent(cfg, genes=data["genes"])
    agent.fitness = data["fitness"]
    return agent, cfg


# ─────────────────────────────────────────────
# Task configurations
# ─────────────────────────────────────────────

TASK_CONFIGS = {
    "Walker-v0": {
        "env_name": "Walker-v0",
        "robot": WALKER,
        "budget": 5_000_000,
        # ES / CMA-ES params
        "generations": 1000,
        "lambda": 50,
        "mu": 10,
        "sigma": 0.3,
        "lr": 0.2,
        "max_steps": 500,
        "h_size": 32,
        # 1+lambda params
        "1pl_sigma": 0.1,
    },
    "Thrower-v0": {
        "env_name": "Thrower-v0",
        "robot": WALKER,
        "budget": 5_000_000,
        "generations": 1000,
        "lambda": 50,
        "mu": 10,
        "sigma": 0.3,
        "lr": 0.15,
        "max_steps": 500,
        "h_size": 32,
        "1pl_sigma": 0.1,
    },
    "Climb-v2": {
        "env_name": "Climb-v2",
        "robot": WALKER,
        "budget": 5_000_000,
        "generations": 1000,
        "lambda": 50,
        "mu": 10,
        "sigma": 0.4,
        "lr": 0.15,
        "max_steps": 500,
        "h_size": 64,  # harder task → bigger network
        "1pl_sigma": 0.15,
    },
}


# ─────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────

ALGORITHMS = {
    "ES": ES,
    "CMA_ES": CMA_ES,
    "1plus_lambda": one_plus_lambda,
}

def run_task(task_name, algorithm="CMA_ES", n_runs=2, log_dir="logs"):
    """
    Run evolution for a given task, n_runs times.
    Returns list of (elite_agent, fits_history, steps_history) per run.
    """
    config = TASK_CONFIGS[task_name]
    algo_fn = ALGORITHMS[algorithm]
    results = []

    print(f"\n{'='*60}")
    print(f"  Task: {task_name} | Algorithm: {algorithm} | Runs: {n_runs}")
    print(f"{'='*60}")

    for run_id in range(n_runs):
        print(f"\n── Run {run_id + 1}/{n_runs} ──")
        np.random.seed(run_id * 100 + 42)
        torch.manual_seed(run_id * 100 + 42)

        elite, fits, steps = algo_fn(config, run_id=run_id, log_dir=log_dir)
        results.append((elite, fits, steps))
        print(f"  → Final fitness: {elite.fitness:.4f}")

    # Print summary
    final_fits = [r[0].fitness for r in results]
    print(f"\n{'─'*40}")
    print(f"  {task_name} summary over {n_runs} runs:")
    print(f"  Scores: {[f'{f:.3f}' for f in final_fits]}")
    print(f"  Mean ± std: {np.mean(final_fits):.3f} ± {np.std(final_fits):.3f}")
    print(f"{'─'*40}")

    return results


def run_all_tasks(algorithm="CMA_ES", n_runs=2, log_dir="logs"):
    """Run evolution for all three tasks."""
    all_results = {}
    for task in ["Walker-v0", "Thrower-v0", "Climb-v2"]:
        all_results[task] = run_task(task, algorithm=algorithm,
                                     n_runs=n_runs, log_dir=log_dir)

    print("\n" + "="*60)
    print("  FINAL SUMMARY")
    print("="*60)
    for task, results in all_results.items():
        fits = [r[0].fitness for r in results]
        print(f"  {task:15s}  avg={np.mean(fits):.3f}  std={np.std(fits):.3f}  "
              f"best={np.max(fits):.3f}")
    return all_results


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EvoGym Neuroevolution Project")
    parser.add_argument("--task",  default="all",
                        choices=["all", "Walker-v0", "Thrower-v0", "Climb-v2"])
    parser.add_argument("--algo",  default="CMA_ES",
                        choices=["CMA_ES", "ES", "1plus_lambda"])
    parser.add_argument("--runs",  type=int, default=2)
    parser.add_argument("--log_dir", default="logs")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    if args.task == "all":
        run_all_tasks(algorithm=args.algo, n_runs=args.runs, log_dir=args.log_dir)
    else:
        run_task(args.task, algorithm=args.algo, n_runs=args.runs,
                 log_dir=args.log_dir)
