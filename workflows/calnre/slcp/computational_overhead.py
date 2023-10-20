from collections import defaultdict
from copy import deepcopy
import pickle
import time
import random

import numpy
from tqdm import tqdm, trange
import torch
from hypothesis.util import load_module
from src.calibration.nre.train import GDStep
from src.calibration.nre.criterion import CalibratedCriterion, BaseCriterion
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader

import setting
import ratio_estimation
from ratio_estimation import RatioEstimator

simulation_budgets = [2**i for i in range(10, 18)]
n_sampless = [2, 4, 8, 16, 32, 64, 128]
epochs = 10
n = 5
gamma = 5
batch_size = 128
lr = 0.001
weight_decay = 0
Prior = setting.Prior()
Simulator = setting.Simulator
batched = setting.batched
seed = 2023


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def copy_dataset_device(ds, device):
    ds = deepcopy(ds)
    for key, val in ds._datasets.items():
        ds._datasets[key] = TensorDataset(*(t.to(device) for t in val.tensors))
    return ds


def nested_defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        out_d = dict()
        for key, val in d.items():
            out_d[key] = nested_defaultdict_to_dict(val)
        return out_d
    else:
        return d


def main():
    gpu = torch.device("cuda")

    # CPU and GPU warmup
    warmup(budget=2**14, gpu=gpu, rounds=5)

    # Experiment
    nre_times = defaultdict(lambda: defaultdict(list))
    calnre_times = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for budget in tqdm(simulation_budgets, desc="Budget"):
        dataset = load_module(f"ratio_estimation.DatasetJointTrain{budget}")()
        gpu_dataset = copy_dataset_device(dataset, gpu)
        for _ in trange(n, leave=False):
            estimator = RatioEstimator()
            # CPU
            ## Vanilla
            _estimator = deepcopy(estimator)
            optimizer = torch.optim.AdamW(
                _estimator.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
            step_optimizer = GDStep(optimizer, clip=1)
            criterion = BaseCriterion(_estimator, batch_size=batch_size)
            g = torch.Generator()
            g.manual_seed(seed)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                drop_last=True,
                num_workers=0,
                pin_memory=False,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=g,
            )
            _estimator.train()
            start_time = time.time()
            for _ in range(epochs):
                for sample_joint in dataloader:
                    loss = criterion(**sample_joint)
                    step_optimizer(loss)
            nre_times["cpu"][budget].append(time.time() - start_time)
            ## Calibrated
            for n_samples in n_sampless:
                _estimator = deepcopy(estimator)
                optimizer = torch.optim.AdamW(
                    _estimator.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                )
                step_optimizer = GDStep(optimizer, clip=1)
                criterion = CalibratedCriterion(
                    estimator=_estimator,
                    prior=Prior,
                    n_samples=n_samples,
                    gamma=gamma,
                    batched=batched,
                    batch_size=batch_size,
                    device=torch.device("cpu"),
                )
                g = torch.Generator()
                g.manual_seed(seed)
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    drop_last=True,
                    num_workers=0,
                    pin_memory=False,
                    shuffle=True,
                    worker_init_fn=seed_worker,
                    generator=g,
                )
                _estimator.train()
                start_time = time.time()
                for _ in range(epochs):
                    for sample_joint in dataloader:
                        loss = criterion(**sample_joint)
                        step_optimizer(loss)
                calnre_times["cpu"][budget][n_samples].append(
                    time.time() - start_time
                )

            # GPU
            ## Vanilla
            _estimator = deepcopy(estimator)
            _estimator.to(gpu)
            optimizer = torch.optim.AdamW(
                _estimator.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
            step_optimizer = GDStep(optimizer, clip=1)
            criterion = BaseCriterion(_estimator, batch_size=batch_size)
            criterion.to(gpu)
            g = torch.Generator()
            g.manual_seed(seed)
            dataloader = DataLoader(
                gpu_dataset,
                batch_size=batch_size,
                drop_last=True,
                num_workers=0,
                pin_memory=False,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=g,
            )
            _estimator.train()
            start_time = time.time()
            for _ in range(epochs):
                for sample_joint in dataloader:
                    loss = criterion(**sample_joint)
                    step_optimizer(loss)
            nre_times["gpu"][budget].append(time.time() - start_time)
            ## Calibrated
            for n_samples in n_sampless:
                _estimator = deepcopy(estimator)
                _estimator.to(gpu)
                optimizer = torch.optim.AdamW(
                    _estimator.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                )
                step_optimizer = GDStep(optimizer, clip=1)
                criterion = CalibratedCriterion(
                    estimator=_estimator,
                    prior=Prior,
                    n_samples=n_samples,
                    gamma=gamma,
                    batched=batched,
                    batch_size=batch_size,
                    device=gpu,
                )
                criterion.to(gpu)
                g = torch.Generator()
                g.manual_seed(seed)
                dataloader = DataLoader(
                    gpu_dataset,
                    batch_size=batch_size,
                    drop_last=True,
                    num_workers=0,
                    pin_memory=False,
                    shuffle=True,
                    worker_init_fn=seed_worker,
                    generator=g,
                )
                _estimator.train()
                start_time = time.time()
                for _ in range(epochs):
                    for sample_joint in dataloader:
                        loss = criterion(**sample_joint)
                        step_optimizer(loss)
                calnre_times["gpu"][budget][n_samples].append(
                    time.time() - start_time
                )
    with open("./slcp/computational_overhead.pkl", "wb") as f:
        pickle.dump(
            {
                "nre": nested_defaultdict_to_dict(nre_times),
                "calnre": nested_defaultdict_to_dict(calnre_times),
            },
            f,
        )


def warmup(budget, gpu, rounds=5):
    dataset = load_module(f"ratio_estimation.DatasetJointTrain{budget}")()
    gpu_dataset = copy_dataset_device(dataset, gpu)
    estimator = RatioEstimator()
    for _ in range(rounds):
        # CPU
        ## Vanilla
        _estimator = deepcopy(estimator)
        optimizer = torch.optim.AdamW(
            _estimator.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        step_optimizer = GDStep(optimizer, clip=1)
        criterion = BaseCriterion(_estimator, batch_size=batch_size)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=True,
            num_workers=0,
            pin_memory=False,
            shuffle=True,
        )
        _estimator.train()
        for sample_joint in dataloader:
            loss = criterion(**sample_joint)
            step_optimizer(loss)
        # GPU
        ## Vanilla
        _estimator = deepcopy(estimator)
        _estimator.to(gpu)
        optimizer = torch.optim.AdamW(
            _estimator.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        step_optimizer = GDStep(optimizer, clip=1)
        criterion = BaseCriterion(_estimator, batch_size=batch_size)
        criterion.to(gpu)
        dataloader = DataLoader(
            gpu_dataset,
            batch_size=batch_size,
            drop_last=True,
            num_workers=0,
            pin_memory=False,
            shuffle=True,
        )
        _estimator.train()
        for sample_joint in dataloader:
            loss = criterion(**sample_joint)
            step_optimizer(loss)


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA device not available!"
    main()
