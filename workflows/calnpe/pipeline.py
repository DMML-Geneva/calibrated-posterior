import argparse
from glob import glob
import inspect
import os
import sys
from copy import deepcopy
from matplotlib import pyplot as plt

from tqdm import tqdm
import awflow as aw
import hypothesis as h
import numpy as np
import torch
from awflow.contrib.simulate import generate
from hypothesis.util import load_module
from hypothesis.util.data.numpy import merge
from hypothesis.stat import highest_density_level
from sbi.utils.get_nn_models import posterior_nn
from torch.utils.data import TensorDataset, DataLoader

from workflows.calnpe.utils import build_embedding, load_checkpoint

# Increase recursion depth (large workflow)
sys.setrecursionlimit(100000000)

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument(
    "--estimators",
    type=int,
    default=5,
    help="Number of estimators to train (default: 5).",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=None,
    help="Regularizer strength (default: 100.0).",
)
parser.add_argument(
    "--simulation_budget",
    type=int,
    default=None,
    help="simulation budget to run (default: run all budgets).",
)
parser.add_argument(
    "--lr", type=float, default=0.001, help="Learning rate (default: 0.001)."
)
parser.add_argument(
    "--partition",
    type=str,
    default=None,
    help="Slurm partition to execute the pipeline on (default: none).",
)
parser.add_argument(
    "--problem",
    type=str,
    default="",
    help="Problem to execute (default: none).",
)
parser.add_argument(
    "--redo",
    action="store_true",
    help="Remove the generating files, pipeline will be re-executed (default: false).",
)
parser.add_argument(
    "--schedule_gamma",
    action="store_true",
    help="Schedule the parameter gamma (default: false).",
)
parser.add_argument(
    "--small_test_val",
    action="store_true",
    help="Use smaller test and validation datasets (default: false).",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Use smaller test and validation datasets (default: false).",
)
arguments, _ = parser.parse_known_args()

# Check if the pipeline needs to be re-executed.
if arguments.redo:
    os.system("rm -rf output")

# Check if a problem has been specified
if not os.path.exists(arguments.problem):
    raise ValueError("Unknown problem:", arguments.problem)
problem = arguments.problem

# Load the problem scope (I was previously using importlib)
if problem == "gw":
    from gw import setting
elif problem == "lotka_volterra":
    from lotka_volterra import setting
elif problem == "mg1":
    from mg1 import setting
elif problem == "slcp":
    from slcp import setting
elif problem == "spatialsir":
    from spatialsir import setting
elif problem == "weinberg":
    from weinberg import setting

# Setup the problem setting
Prior = setting.Prior
Simulator = setting.Simulator
memory = setting.memory
ngpus = setting.ngpus
if ngpus == 0:
    workers = 0
else:
    workers = 4

prior_module = inspect.getmodule(Prior).__name__
# Calibration regularizer hyperparams:
n_samples = 16
instances_subsample = 1.0
# Script parameters
batch_sizes = [128]
confidence_levels = np.linspace(0.05, 0.95, 19)
learning_rate = arguments.lr
num_estimators = arguments.estimators
schedule_gamma = arguments.schedule_gamma
small_test_val = arguments.small_test_val
debug = arguments.debug
if arguments.simulation_budget is None:
    if debug:
        simulation_budgets = [2**i for i in range(15, 16)]
    else:
        simulation_budgets = [2**i for i in range(10, 18)]

else:
    simulation_budgets = [arguments.simulation_budget]

# Utilities ####################################################################


def get_extent(problem):
    """
    Get the extent for different problems.

    Args:
        problem (str)
    """
    prior = Prior()

    if problem == "weinberg":
        extent = [prior.low.item(), prior.high.item()]
    elif problem == "mg1":
        extent = [
            prior.low[0].item(),
            prior.high[0].item(),
            prior.low[1].item(),
            prior.high[1].item(),
            prior.low[2].item(),
            prior.high[2].item(),
        ]
    elif (
        (problem == "slcp")
        or (problem == "spatialsir")
        or (problem == "gw")
        or (problem == "lotka_volterra")
    ):
        extent = [
            prior.low[0].item(),
            prior.high[0].item(),
            prior.low[1].item(),
            prior.high[1].item(),
        ]
    else:
        raise NotImplementedError

    return extent


@torch.no_grad()
def simulate(outputdir, budget):
    # Check if files have been generated.
    inputs_exists = os.path.exists(outputdir + "/inputs.npy")
    outputs_exists = os.path.exists(outputdir + "/outputs.npy")
    if not inputs_exists or not outputs_exists:
        simulator = Simulator()
        prior = Prior()
        inputs = prior.sample((budget,)).view(budget, -1)
        outputs = simulator(inputs)
        np.save(outputdir + "/inputs.npy", inputs.float().numpy())
        np.save(outputdir + "/outputs.npy", outputs.float().numpy())


def build_density_estimator(problem):
    """
    Build a density estimator.

    Args:
        problem (str)

    Returns:
        Function that builds density estimator for learning the posterior.
    """
    embedding = build_embedding(problem=problem)

    if problem == "weinberg":
        hidden_features = 64
        num_transforms = 1
    else:
        hidden_features = 64
        num_transforms = 3

    return posterior_nn(
        model="nsf",
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        embedding_net=embedding,
    )


def get_log_prob_for_context(
    r, inputs, context=None, embedded_context=None, ensemble=False
):
    if not ensemble:
        if embedded_context is None:
            embedded_context = r._embedding_net(context.unsqueeze(0))
        noise, logabsdet = r._transform(
            inputs,
            context=embedded_context.expand(
                len(inputs), *[-1] * (embedded_context.ndim - 1)
            ),
        )
        log_prob = r._distribution.log_prob(
            noise,
            context=embedded_context.expand(
                len(inputs), *[-1] * (embedded_context.ndim - 1)
            ),
        )
        return log_prob + logabsdet
    else:
        posteriors = [
            get_log_prob_for_context(
                flow,
                inputs,
                context,
                embedded_context[idx]
                if embedded_context is not None
                else None,
            )
            for idx, flow in enumerate(r.flows)
        ]
        log_sum_exp = torch.stack(posteriors, dim=0).logsumexp(dim=0)
        return log_sum_exp - torch.tensor(
            float(len(r.flows)), device=log_sum_exp.device
        )


@torch.no_grad()
def compute_log_posterior(
    r, observable, problem, resolution=100, return_grid=False, batch_size=None
):
    extent = get_extent(problem=problem)
    # Prepare grid
    epsilon = 0.00001
    # Account for half-open interval of uniform
    # prior, ``accelerator`` is just the device:
    p1 = torch.linspace(extent[0], extent[1] - epsilon, resolution).to(
        h.accelerator
    )
    if problem != "weinberg":
        p2 = torch.linspace(extent[2], extent[3] - epsilon, resolution).to(
            h.accelerator
        )
    else:
        inputs = deepcopy(p1.view(-1, 1))
    if problem == "mg1":
        p3 = torch.linspace(extent[4], extent[5] - epsilon, resolution).to(
            h.accelerator
        )
        g1, g2, g3 = torch.meshgrid(p1.view(-1), p2.view(-1), p3.view(-1))
        # Vectorize
        inputs = torch.cat(
            [g1.reshape(-1, 1), g2.reshape(-1, 1), g3.reshape(-1, 1)], dim=1
        )
    elif problem == "weinberg":
        pass
    else:
        g1, g2 = torch.meshgrid(p1.view(-1), p2.view(-1))
        # Vectorize
        inputs = torch.cat([g1.reshape(-1, 1), g2.reshape(-1, 1)], dim=1)

    observable = observable.to(h.accelerator)
    if isinstance(r, FlowEnsemble):
        embeded_observable = [
            flow._embedding_net(observable) for flow in r.flows
        ]
    else:
        embeded_observable = r._embedding_net(observable)

    if batch_size is None:
        log_posterior = get_log_prob_for_context(
            r,
            inputs=inputs,
            embedded_context=embeded_observable,
            ensemble=isinstance(r, FlowEnsemble),
        )
    else:
        log_posterior = torch.empty(inputs.shape[0])
        for b in range(0, inputs.shape[0], batch_size):
            cur_inputs = inputs[b : b + batch_size]
            log_posterior[b : b + batch_size] = get_log_prob_for_context(
                r,
                inputs=cur_inputs,
                embedded_context=embeded_observable,
                ensemble=isinstance(r, FlowEnsemble),
            )
    if problem == "gw":
        log_posterior = log_posterior.view(resolution, resolution).cpu()

    elif problem == "mg1":
        log_posterior = log_posterior.view(
            resolution, resolution, resolution
        ).cpu()

    else:
        if problem != "weinberg":
            log_posterior = log_posterior.view(resolution, resolution).cpu()
        else:
            log_posterior = log_posterior.view(resolution).cpu()

    if return_grid:
        if problem == "mg1":
            return log_posterior, p1.cpu(), p2.cpu(), p3.cpu()
        elif problem == "weinberg":
            return log_posterior, p1.cpu()
        else:
            return log_posterior, p1.cpu(), p2.cpu()
    else:
        return log_posterior


@torch.no_grad()
def estimate_coverage(r, inputs, outputs, outputdir, problem, alphas=[0.05]):
    n = len(inputs)
    covered = [0 for _ in alphas]
    sizes = [[] for _ in range(len(alphas))]
    bias = [0.0, 0.0]
    bias_square = [0.0, 0.0]
    variance = [0.0, 0.0]
    if problem == "mg1":
        bias.extend([0.0])
        bias_square.extend([0.0])
        variance.extend([0.0])
    elif problem == "weinberg":
        bias.pop(1)
        bias_square.pop(1)
        variance.pop(1)

    resolution = 90  # this is BNRE (and CalNRE) hyperparameter, kept for now
    return_grid = True

    extent = get_extent(problem=problem)
    length_1 = (extent[1] - extent[0]) / resolution
    if problem != "weinberg":
        length_2 = (extent[3] - extent[2]) / resolution
    if problem == "mg1":
        length_3 = (extent[5] - extent[4]) / resolution

    for index in tqdm(range(n), "Coverages evaluated"):
        # Prepare setup
        nominal = inputs[index]
        if problem != "weinberg":
            nominal = nominal.squeeze().unsqueeze(0)
        else:
            nominal = nominal.unsqueeze(dim=0)
        observable = outputs[index].squeeze().unsqueeze(0)
        nominal = nominal.to(h.accelerator)
        observable = observable.to(h.accelerator)
        with torch.no_grad():
            if problem == "mg1":
                pdf, p1, p2, p3 = compute_log_posterior(
                    r=r,
                    observable=observable,
                    problem=problem,
                    resolution=resolution,
                    return_grid=return_grid,
                )
            elif problem == "weinberg":
                pdf, p1 = compute_log_posterior(
                    r=r,
                    observable=observable,
                    problem=problem,
                    resolution=resolution,
                    return_grid=return_grid,
                )
                # p1 = p1[:, 0]  # this is only reshaping operation in this case
            elif problem == "gw":
                pdf, p1, p2 = compute_log_posterior(
                    r=r,
                    observable=observable,
                    problem=problem,
                    resolution=resolution,
                    return_grid=return_grid,
                )
            else:
                pdf, p1, p2 = compute_log_posterior(
                    r=r,
                    observable=observable,
                    problem=problem,
                    resolution=resolution,
                    return_grid=return_grid,
                )
            pdf = pdf.exp()
            nominal_pdf = (
                compute_log_pdf(r=r, inputs=nominal, outputs=observable)
                .exp()
                .cpu()
            )
        for i, alpha in enumerate(alphas):
            level, mask = highest_density_level(pdf, alpha, region=True)
            sizes[i].append(np.sum(mask) / np.prod(np.shape(mask)))
            if nominal_pdf >= level:
                covered[i] += 1

        if problem == "mg1":
            pdf = pdf / (length_1 * length_2 * length_3 * pdf.sum())

            margin_1 = pdf.sum(dim=2).sum(dim=1) * length_3 * length_2
            margin_2 = pdf.sum(dim=2).sum(dim=0) * length_3 * length_1
            margin_3 = pdf.sum(dim=1).sum(dim=0) * length_2 * length_1
            mean_1 = (margin_1 * length_1 * p1).sum()
            mean_2 = (margin_2 * length_2 * p2).sum()
            mean_3 = (margin_3 * length_3 * p3).sum()
            bias[0] += torch.abs((mean_1 - nominal[0, 0]).cpu().float())
            bias[1] += torch.abs((mean_2 - nominal[0, 1]).cpu().float())
            bias[2] += torch.abs((mean_3 - nominal[0, 2]).cpu().float())
            bias_square[0] += (mean_1 - nominal[0, 0]).cpu().float() ** 2
            bias_square[1] += (mean_2 - nominal[0, 1]).cpu().float() ** 2
            bias_square[2] += (mean_3 - nominal[0, 2]).cpu().float() ** 2
            variance[0] += (
                (margin_1 * length_1 * (p1 - mean_1) ** 2).sum().cpu().float()
            )
            variance[1] += (
                (margin_2 * length_2 * (p2 - mean_2) ** 2).sum().cpu().float()
            )
            variance[2] += (
                (margin_3 * length_3 * (p3 - mean_3) ** 2).sum().cpu().float()
            )
        elif problem == "weinberg":
            pdf = pdf / (length_1 * pdf.sum())
            margin_1 = pdf
            mean_1 = (margin_1 * length_1 * p1).sum()
            bias[0] += torch.abs((mean_1 - nominal[0]).cpu().float())
            bias_square[0] += (mean_1 - nominal[0]).cpu().float() ** 2
            variance[0] += (
                (margin_1 * length_1 * (p1 - mean_1) ** 2).sum().cpu().float()
            )
        else:
            pdf = pdf / (length_1 * length_2 * pdf.sum())
            margin_1 = pdf.sum(dim=1) * length_2
            margin_2 = pdf.sum(dim=0) * length_1
            mean_1 = (margin_1 * length_1 * p1).sum()
            mean_2 = (margin_2 * length_2 * p2).sum()
            bias[0] += torch.abs((mean_1 - nominal[0, 0]).cpu().float())
            bias[1] += torch.abs((mean_2 - nominal[0, 1]).cpu().float())
            bias_square[0] += (mean_1 - nominal[0, 0]).cpu().float() ** 2
            bias_square[1] += (mean_2 - nominal[0, 1]).cpu().float() ** 2
            variance[0] += (
                (margin_1 * length_1 * (p1 - mean_1) ** 2).sum().cpu().float()
            )
            variance[1] += (
                (margin_2 * length_2 * (p2 - mean_2) ** 2).sum().cpu().float()
            )

        if (index < 20) and (problem not in ["mg1", "weinberg"]):
            plot_posterior(
                p1=p1,
                p2=p2,
                pdf=pdf,
                nominal=nominal,
                mean_1=mean_1,
                mean_2=mean_2,
                index=index,
                outputdir=outputdir,
                extent=extent,
            )

    return (
        [x / n for x in covered],
        sizes,
        [x / n for x in bias],
        [x / n for x in variance],
        [x / n for x in bias_square],
    )


class FlowEnsemble:
    def __init__(self, flows):
        """
        Args:
            flows (list): List containing flows
        """
        self.flows = flows

    def log_prob(self, *args, **kwargs):
        posteriors = [flow.log_prob(*args, **kwargs) for flow in self.flows]
        return torch.stack(posteriors, dim=0).exp().mean(dim=0).log()


def plot_posterior(
    p1, p2, pdf, nominal, mean_1, mean_2, index, outputdir, extent
):
    p1 = p1.cpu()
    p2 = p2.cpu()
    pdf = pdf.cpu()
    nominal = nominal.cpu()
    mean_1 = mean_1.cpu()
    mean_2 = mean_2.cpu()
    g1, g2 = torch.meshgrid(p1.view(-1), p2.view(-1))
    plt.pcolormesh(
        g1, g2, pdf, antialiased=True, edgecolors="face", shading="auto"
    )
    plt.set_cmap("viridis_r")
    plt.colorbar()
    plt.plot(nominal[0, 0], nominal[0, 1], "*", color="k")
    plt.hlines(mean_2, extent[0], extent[1])
    plt.vlines(mean_1, extent[2], extent[3])

    plt.savefig(outputdir + "/posterior_{}.pdf".format(index))
    plt.close()


def optimize(outputdir, budget, batch_size, gamma, read_checkpoint):
    if small_test_val:
        val = "DatasetJointValidateSmall"
    else:
        val = "DatasetJointValidate"

    if debug:
        epochs = 2
    else:
        epochs = 50

    # ``problem`` is an argparsing argument
    assert problem in [
        "gw",
        "lotka_volterra",
        "mg1",
        "slcp",
        "spatialsir",
        "weinberg",
    ]
    # ``outputdir`` can also be split to give task index and the train stage
    # index
    if ngpus > 0:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"

    density_estimator = build_density_estimator(problem=problem)
    inference = load_module("src.calibration.npe.snpe.CalibratedSNPE")(
        prior=getattr(load_module(prior_module), "Prior")(),
        density_estimator=density_estimator,
        summary_writer=None,
        device=device,
        num_samples=n_samples,
        instances_subsample=instances_subsample,
        validation_fraction=0,
        show_progress_bars=True,
    )
    train_dataset = load_module(
        f"{problem}.ratio_estimation.DatasetJointTrain{budget}"
    )()
    val_dataset = load_module(f"{problem}.ratio_estimation.{val}")()
    inference.append_simulations(
        theta=train_dataset._datasets["inputs"],
        x=train_dataset._datasets["outputs"],
        data_device="cpu",
    ).train(
        learning_rate=learning_rate,
        max_num_epochs=epochs,
        training_batch_size=batch_size,
        validation_fraction=0,
        validation_inputs=val_dataset._datasets["inputs"],
        validation_outputs=val_dataset._datasets["outputs"],
        path_checkpoint=read_checkpoint,
        show_train_summary=True,
    )
    # Save validation losses (per stage):
    np.save(os.path.join(outputdir, "val_losses.npy"), inference.val_losses)
    # Save checkpoint corresponding to latest model:
    inference.save_checkpoint(path=outputdir)
    # Save model parameters of best model:
    torch.save(
        inference._best_state_dict,
        os.path.join(outputdir, "checkpoint_best.pt"),
    )


@torch.no_grad()
def compute_log_pdf(r, inputs, outputs):
    log_posterior = r.log_prob(inputs=inputs, context=outputs)
    return log_posterior


def generate_simulations():
    root = problem + "/data"
    dependencies = []

    # Simulation pipeline properties
    n = simulation_budgets[-1]
    props = {"cpus": 1, "memory": "8GB", "timelimit": "1-00:00:00"}
    # Training
    g_train = generate(simulate, n, root + "/train", blocks=128, **props)
    dependencies.extend(merge_blocks(root + "/train", g_train))
    # Testing
    g_test = generate(simulate, n, root + "/test", blocks=128, **props)
    dependencies.extend(merge_blocks(root + "/test", g_test))
    # Validation
    g_validate = generate(simulate, n, root + "/validate", blocks=128, **props)
    dependencies.extend(merge_blocks(root + "/validate", g_validate))
    # Coverage
    g_coverage = generate(
        simulate, 10000, root + "/coverage", blocks=10, **props
    )
    dependencies.extend(merge_blocks(root + "/coverage", g_coverage))

    return dependencies


def merge_blocks(directory, dependencies):
    # Merge input files
    @aw.cpus(2)
    @aw.dependency(dependencies)
    @aw.memory("16GB")
    @aw.postcondition(aw.exists(directory + "/inputs.npy"))
    @aw.partition(
        os.environ.get("CPU_PARTITIONS", "shared-cpu,public-cpu").split(",")
    )
    def merge_inputs():
        files = glob(directory + "/blocks/*/inputs.npy")
        files.sort()
        merge(
            input_files=files,
            output_file=directory + "/inputs.npy",
            tempfile=directory + "/temp_inputs",
            in_memory=False,
            axis=0,
        )

    # Merge output files
    @aw.cpus(2)
    @aw.dependency(dependencies)
    @aw.memory("16GB")
    @aw.postcondition(aw.exists(directory + "/outputs.npy"))
    @aw.partition(
        os.environ.get("CPU_PARTITIONS", "shared-cpu,public-cpu").split(",")
    )
    def merge_outputs():
        files = glob(directory + "/blocks/*/outputs.npy")
        files.sort()
        merge(
            input_files=files,
            output_file=directory + "/outputs.npy",
            tempfile=directory + "/temp_outputs",
            in_memory=False,
            axis=0,
        )

    return [merge_inputs, merge_outputs]


# Workflow definition ##########################################################


def train_and_evaluate(budget, batch_size, dependencies=None):
    if dependencies is None:
        dependencies = []

    root = (
        problem
        + "/output/estimator/"
        + str(budget)
        + "/"
        + str(batch_size)
        + "/"
        + str(learning_rate)
    )
    if arguments.gamma is None:
        gamma = 5.0
    else:
        gamma = arguments.gamma
        root += "/" + str(gamma)

    if schedule_gamma:
        root += "/schedule"

    os.makedirs(root, exist_ok=True)

    # Train the ratio estimator
    @aw.cpus_and_memory(2, memory)
    @aw.dependency(dependencies)
    @aw.gpus(
        ngpus, os.environ.get("VRAM_PER_GPU", None) if ngpus > 0 else None
    )
    @aw.postcondition(
        aw.num_files(
            root + "/*/0/checkpoint.pt",
            num_estimators,
        )
    )
    @aw.tasks(num_estimators)
    @aw.timelimit("12:00:00" if not debug else "24:00:00")
    @aw.constraint(
        os.environ.get("GPU_CONSTRAINT", None) if ngpus > 0 else None
    )
    @aw.partition(
        os.environ.get("GPU_PARTITIONS", "shared-gpu,public-gpu").split(",")
        if ngpus > 0
        else os.environ.get("CPU_PARTITIONS", "shared-cpu,public-cpu").split(
            ","
        )
    )
    def train_0(task_index=0):
        outputdir = root + ("/" + str(task_index) + "/0")
        os.makedirs(outputdir, exist_ok=True)
        if not os.path.exists(outputdir + "/checkpoint.pt"):
            optimize(
                outputdir,
                budget,
                batch_size,
                gamma,
                read_checkpoint=None,
            )

    # The training method is the dependency
    dependencies.append(train_0)

    @aw.cpus_and_memory(2, memory)
    @aw.dependency(train_0)
    @aw.gpus(
        ngpus, os.environ.get("VRAM_PER_GPU", None) if ngpus > 0 else None
    )
    @aw.postcondition(
        aw.num_files(
            root + "/*/1/checkpoint.pt",
            num_estimators,
        )
    )
    @aw.tasks(num_estimators)
    @aw.timelimit("12:00:00" if not debug else "24:00:00")
    @aw.constraint(
        os.environ.get("GPU_CONSTRAINT", None) if ngpus > 0 else None
    )
    @aw.partition(
        os.environ.get("GPU_PARTITIONS", "shared-gpu,public-gpu").split(",")
        if ngpus > 0
        else os.environ.get("CPU_PARTITIONS", "shared-cpu,public-cpu").split(
            ","
        )
    )
    def train_1(task_index=0):
        outputdir = root + ("/" + str(task_index) + "/1")
        os.makedirs(outputdir, exist_ok=True)
        if not os.path.exists(outputdir + "/checkpoint.pt"):
            optimize(
                outputdir,
                budget,
                batch_size,
                gamma,
                read_checkpoint=root + "/" + str(task_index) + "/0",
            )

    # The training method is the dependency
    dependencies.append(train_1)

    @aw.cpus_and_memory(2, memory)
    @aw.dependency(train_1)
    @aw.gpus(
        ngpus, os.environ.get("VRAM_PER_GPU", None) if ngpus > 0 else None
    )
    @aw.postcondition(
        aw.num_files(
            root + "/*/2/checkpoint.pt",
            num_estimators,
        )
    )
    @aw.tasks(num_estimators)
    @aw.timelimit("12:00:00" if not debug else "24:00:00")
    @aw.constraint(
        os.environ.get("GPU_CONSTRAINT", None) if ngpus > 0 else None
    )
    @aw.partition(
        os.environ.get("GPU_PARTITIONS", "shared-gpu,public-gpu").split(",")
        if ngpus > 0
        else os.environ.get("CPU_PARTITIONS", "shared-cpu,public-cpu").split(
            ","
        )
    )
    def train_2(task_index=0):
        outputdir = root + ("/" + str(task_index) + "/2")
        os.makedirs(outputdir, exist_ok=True)
        if not os.path.exists(outputdir + "/checkpoint.pt"):
            optimize(
                outputdir,
                budget,
                batch_size,
                gamma,
                read_checkpoint=root + "/" + str(task_index) + "/1",
            )

    # The training method is the dependency
    dependencies.append(train_2)

    @aw.cpus_and_memory(2, memory)
    @aw.dependency(train_2)
    @aw.gpus(
        ngpus, os.environ.get("VRAM_PER_GPU", None) if ngpus > 0 else None
    )
    @aw.postcondition(
        aw.num_files(
            root + "/*/3/checkpoint.pt",
            num_estimators,
        )
    )
    @aw.tasks(num_estimators)
    @aw.timelimit("12:00:00" if not debug else "24:00:00")
    @aw.constraint(
        os.environ.get("GPU_CONSTRAINT", None) if ngpus > 0 else None
    )
    @aw.partition(
        os.environ.get("GPU_PARTITIONS", "shared-gpu,public-gpu").split(",")
        if ngpus > 0
        else os.environ.get("CPU_PARTITIONS", "shared-cpu,public-cpu").split(
            ","
        )
    )
    def train_3(task_index=0):
        outputdir = root + ("/" + str(task_index) + "/3")
        os.makedirs(outputdir, exist_ok=True)
        if not os.path.exists(outputdir + "/checkpoint.pt"):
            optimize(
                outputdir,
                budget,
                batch_size,
                gamma,
                read_checkpoint=root + "/" + str(task_index) + "/2",
            )

    # The training method is the dependency
    dependencies.append(train_3)

    @aw.cpus_and_memory(2, memory)
    @aw.dependency(train_3)
    @aw.gpus(
        ngpus, os.environ.get("VRAM_PER_GPU", None) if ngpus > 0 else None
    )
    @aw.postcondition(
        aw.num_files(
            root + "/*/checkpoint.pt",
            num_estimators,
        )
    )
    @aw.tasks(num_estimators)
    @aw.timelimit("12:00:00" if not debug else "24:00:00")
    @aw.constraint(
        os.environ.get("GPU_CONSTRAINT", None) if ngpus > 0 else None
    )
    @aw.partition(
        os.environ.get("GPU_PARTITIONS", "shared-gpu,public-gpu").split(",")
        if ngpus > 0
        else os.environ.get("CPU_PARTITIONS", "shared-cpu,public-cpu").split(
            ","
        )
    )
    def train_4(task_index=0):
        outputdir = root + ("/" + str(task_index) + "/4")
        os.makedirs(outputdir, exist_ok=True)
        if not os.path.exists(outputdir + "/checkpoint.pt"):
            optimize(
                outputdir,
                budget,
                batch_size,
                gamma,
                read_checkpoint=root + "/" + str(task_index) + "/3",
            )

    # The training method is the dependency
    dependencies.append(train_4)

    @aw.cpus_and_memory(2, memory)
    @aw.dependency(train_4)
    @aw.gpus(
        ngpus, os.environ.get("VRAM_PER_GPU", None) if ngpus > 0 else None
    )
    @aw.postcondition(
        aw.num_files(
            root + "/*/checkpoint.pt",
            num_estimators,
        )
    )
    @aw.tasks(num_estimators)
    @aw.timelimit("12:00:00" if not debug else "24:00:00")
    @aw.constraint(
        os.environ.get("GPU_CONSTRAINT", None) if ngpus > 0 else None
    )
    @aw.partition(
        os.environ.get("GPU_PARTITIONS", "shared-gpu,public-gpu").split(",")
        if ngpus > 0
        else os.environ.get("CPU_PARTITIONS", "shared-cpu,public-cpu").split(
            ","
        )
    )
    def train_5(task_index=0):
        outputdir = root + ("/" + str(task_index) + "/5")
        os.makedirs(outputdir, exist_ok=True)
        if not os.path.exists(outputdir + "/checkpoint.pt"):
            optimize(
                outputdir,
                budget,
                batch_size,
                gamma,
                read_checkpoint=root + "/" + str(task_index) + "/4",
            )

    # The training method is the dependency
    dependencies.append(train_5)

    @aw.cpus_and_memory(2, memory)
    @aw.dependency(train_5)
    @aw.gpus(
        ngpus, os.environ.get("VRAM_PER_GPU", None) if ngpus > 0 else None
    )
    @aw.postcondition(
        aw.num_files(
            root + "/*/checkpoint.pt",
            num_estimators,
        )
    )
    @aw.tasks(num_estimators)
    @aw.timelimit("12:00:00" if not debug else "24:00:00")
    @aw.constraint(
        os.environ.get("GPU_CONSTRAINT", None) if ngpus > 0 else None
    )
    @aw.partition(
        os.environ.get("GPU_PARTITIONS", "shared-gpu,public-gpu").split(",")
        if ngpus > 0
        else os.environ.get("CPU_PARTITIONS", "shared-cpu,public-cpu").split(
            ","
        )
    )
    def train_6(task_index=0):
        outputdir = root + ("/" + str(task_index) + "/6")
        os.makedirs(outputdir, exist_ok=True)
        if not os.path.exists(outputdir + "/checkpoint.pt"):
            optimize(
                outputdir,
                budget,
                batch_size,
                gamma,
                read_checkpoint=root + "/" + str(task_index) + "/5",
            )

    # The training method is the dependency
    dependencies.append(train_6)

    @aw.cpus_and_memory(2, memory)
    @aw.dependency(train_6)
    @aw.gpus(
        ngpus, os.environ.get("VRAM_PER_GPU", None) if ngpus > 0 else None
    )
    @aw.postcondition(
        aw.num_files(
            root + "/*/checkpoint.pt",
            num_estimators,
        )
    )
    @aw.tasks(num_estimators)
    @aw.timelimit("12:00:00" if not debug else "24:00:00")
    @aw.constraint(
        os.environ.get("GPU_CONSTRAINT", None) if ngpus > 0 else None
    )
    @aw.partition(
        os.environ.get("GPU_PARTITIONS", "shared-gpu,public-gpu").split(",")
        if ngpus > 0
        else os.environ.get("CPU_PARTITIONS", "shared-cpu,public-cpu").split(
            ","
        )
    )
    def train_7(task_index=0):
        outputdir = root + ("/" + str(task_index) + "/7")
        os.makedirs(outputdir, exist_ok=True)
        if not os.path.exists(outputdir + "/checkpoint.pt"):
            optimize(
                outputdir,
                budget,
                batch_size,
                gamma,
                read_checkpoint=root + "/" + str(task_index) + "/6",
            )

    # The training method is the dependency
    dependencies.append(train_7)

    @aw.cpus_and_memory(2, memory)
    @aw.dependency(train_7)
    @aw.gpus(
        ngpus, os.environ.get("VRAM_PER_GPU", None) if ngpus > 0 else None
    )
    @aw.postcondition(
        aw.num_files(
            root + "/*/checkpoint.pt",
            num_estimators,
        )
    )
    @aw.tasks(num_estimators)
    @aw.timelimit("12:00:00" if not debug else "24:00:00")
    @aw.constraint(
        os.environ.get("GPU_CONSTRAINT", None) if ngpus > 0 else None
    )
    @aw.partition(
        os.environ.get("GPU_PARTITIONS", "shared-gpu,public-gpu").split(",")
        if ngpus > 0
        else os.environ.get("CPU_PARTITIONS", "shared-cpu,public-cpu").split(
            ","
        )
    )
    def train_8(task_index=0):
        outputdir = root + ("/" + str(task_index) + "/8")
        os.makedirs(outputdir, exist_ok=True)
        if not os.path.exists(outputdir + "/checkpoint.pt"):
            optimize(
                outputdir,
                budget,
                batch_size,
                gamma,
                read_checkpoint=root + "/" + str(task_index) + "/7",
            )

    # The training method is the dependency
    dependencies.append(train_8)

    @aw.cpus_and_memory(2, memory)
    @aw.dependency(train_8)
    @aw.gpus(
        ngpus, os.environ.get("VRAM_PER_GPU", None) if ngpus > 0 else None
    )
    @aw.postcondition(
        aw.num_files(
            root + "/*/checkpoint.pt",
            num_estimators,
        )
    )
    @aw.tasks(num_estimators)
    @aw.timelimit("12:00:00" if not debug else "24:00:00")
    @aw.constraint(
        os.environ.get("GPU_CONSTRAINT", None) if ngpus > 0 else None
    )
    @aw.partition(
        os.environ.get("GPU_PARTITIONS", "shared-gpu,public-gpu").split(",")
        if ngpus > 0
        else os.environ.get("CPU_PARTITIONS", "shared-cpu,public-cpu").split(
            ","
        )
    )
    def train_9(task_index=0):
        outputdir = root + ("/" + str(task_index))
        os.makedirs(outputdir, exist_ok=True)
        if not os.path.exists(outputdir + "/checkpoint.pt"):
            optimize(
                outputdir,
                budget,
                batch_size,
                gamma,
                read_checkpoint=root + "/" + str(task_index) + "/8",
            )

    # The training method is the dependency
    dependencies.append(train_9)

    # Evaluate its expected coverage
    @aw.cpus_and_memory(2, memory)
    @aw.dependency(train_9)
    @aw.gpus(
        ngpus, os.environ.get("VRAM_PER_GPU", None) if ngpus > 0 else None
    )
    @aw.postcondition(aw.num_files(root + "/*/coverage.npy", num_estimators))
    @aw.postcondition(
        aw.num_files(root + "/*/contour-sizes.npy", num_estimators)
    )
    @aw.postcondition(aw.num_files(root + "/*/bias.npy", num_estimators))
    @aw.postcondition(
        aw.num_files(root + "/*/bias_square.npy", num_estimators)
    )
    @aw.postcondition(aw.num_files(root + "/*/variance.npy", num_estimators))
    @aw.tasks(num_estimators)
    @aw.timelimit("24:00:00" if not debug else "24:00:00")
    @aw.constraint(
        os.environ.get("GPU_CONSTRAINT", None) if ngpus > 0 else None
    )
    @aw.partition(
        os.environ.get("GPU_PARTITIONS", "shared-gpu,public-gpu").split(",")
        if ngpus > 0
        else os.environ.get("CPU_PARTITIONS", "shared-cpu,public-cpu").split(
            ","
        )
    )
    def coverage(task_index=0):
        outputdir = root + "/" + str(task_index)
        if (
            not os.path.exists(outputdir + "/coverage.npy")
            or not os.path.exists(outputdir + "/contour-sizes.npy")
            or not os.path.exists(outputdir + "/bias.npy")
            or not os.path.exists(outputdir + "/variance.npy")
            or not os.path.exists(outputdir + "/bias_square.npy")
        ):
            inputs = torch.from_numpy(
                np.load(problem + "/data/coverage/inputs.npy")
            )
            outputs = torch.from_numpy(
                np.load(problem + "/data/coverage/outputs.npy")
            )
            if (problem != "gw") and (
                (inputs.ndim != 2) or (outputs.ndim != 2)
            ):
                inputs = inputs.reshape(inputs.shape[0], -1)
                outputs = outputs.reshape(outputs.shape[0], -1)

            if small_test_val:
                inputs = inputs[:1000]
                outputs = outputs[:1000]

            if debug:
                inputs = inputs[:50]
                outputs = outputs[:50]

            # ratio_estimator = load_estimator(outputdir + "/checkpoint.pt")
            # Construct flow
            density_estimator = build_density_estimator(problem=problem)
            if isinstance(density_estimator, str):
                build_neural_net = posterior_nn(model=density_estimator)
            else:
                build_neural_net = density_estimator

            flow_sbi = build_neural_net(
                inputs,
                outputs,
            )
            # Load weights for flow
            assert (
                outputdir is not None
            ), f"Invalid path for checkpoint: {outputdir}"
            load_checkpoint(
                model=flow_sbi,
                checkpoint=torch.load(
                    outputdir + "/checkpoint_best.pt",
                    map_location=torch.device("cpu"),
                ),
                load_cp_continue=False,
                optimizer=None,
            )
            flow_sbi = flow_sbi.to(h.accelerator)
            outputs = outputs.to(h.accelerator)
            inputs = inputs.to(h.accelerator)
            # Estimate coverage
            (
                coverages,
                contour_sizes,
                bias,
                variance,
                bias_square,
            ) = estimate_coverage(
                r=flow_sbi,
                inputs=inputs,
                outputs=outputs,
                outputdir=outputdir,
                problem=problem,
                alphas=confidence_levels,
            )
            np.save(outputdir + "/coverage.npy", coverages)
            np.save(outputdir + "/contour-sizes.npy", contour_sizes)
            np.save(outputdir + "/bias.npy", bias)
            np.save(outputdir + "/bias_square.npy", bias_square)
            np.save(outputdir + "/variance.npy", variance)

    # Evaluate expected coverage of ensemble
    @aw.cpus_and_memory(2, memory)
    @aw.dependency(train_9)
    @aw.gpus(
        ngpus, os.environ.get("VRAM_PER_GPU", None) if ngpus > 0 else None
    )
    @aw.postcondition(aw.exists(root + "/coverage.npy"))
    @aw.postcondition(aw.exists(root + "/contour-sizes.npy"))
    @aw.postcondition(aw.exists(root + "/bias.npy"))
    @aw.postcondition(aw.exists(root + "/bias_square.npy"))
    @aw.postcondition(aw.exists(root + "/variance.npy"))
    @aw.timelimit("12:00:00" if not debug else "24:00:00")
    @aw.constraint(
        os.environ.get("GPU_CONSTRAINT", None) if ngpus > 0 else None
    )
    @aw.partition(
        os.environ.get("GPU_PARTITIONS", "shared-gpu,public-gpu").split(",")
        if ngpus > 0
        else os.environ.get("CPU_PARTITIONS", "shared-cpu,public-cpu").split(
            ","
        )
    )
    def coverage_ensemble():
        outputdir = root
        if (
            not os.path.exists(outputdir + "/coverage.npy")
            or not os.path.exists(outputdir + "/contour-sizes.npy")
            or not os.path.exists(outputdir + "/bias.npy")
            or not os.path.exists(outputdir + "/variance.npy")
            or not os.path.exists(outputdir + "/bias_square.npy")
        ):
            inputs = torch.from_numpy(
                np.load(problem + "/data/coverage/inputs.npy")
            )
            outputs = torch.from_numpy(
                np.load(problem + "/data/coverage/outputs.npy")
            )
            if (problem != "gw") and (
                (inputs.ndim != 2) or (outputs.ndim != 2)
            ):
                inputs = inputs.reshape(inputs.shape[0], -1)
                outputs = outputs.reshape(outputs.shape[0], -1)

            if small_test_val:
                inputs = inputs[:1000]
                outputs = outputs[:1000]

            if debug:
                inputs = inputs[:50]
                outputs = outputs[:50]

            # Build ensemble
            path_density_estimators = glob(outputdir + "/*/checkpoint_best.pt")
            flows = []
            for idx, path in enumerate(path_density_estimators):
                density_estimator = build_density_estimator(problem=problem)
                flows.append(
                    density_estimator(
                        inputs,
                        outputs,
                    )
                )
                assert path is not None, f"Invalid path for checkpoint: {path}"
                load_checkpoint(
                    model=flows[idx],
                    checkpoint=torch.load(
                        path,
                        map_location=torch.device("cpu"),
                    ),
                    load_cp_continue=False,
                    optimizer=None,
                )
                flows[idx] = flows[idx].to(h.accelerator)
            outputs = outputs.to(h.accelerator)
            inputs = inputs.to(h.accelerator)
            # Estimate coverage
            (
                coverages,
                contour_sizes,
                bias,
                variance,
                bias_square,
            ) = estimate_coverage(
                r=FlowEnsemble(flows),
                inputs=inputs,
                outputs=outputs,
                outputdir=outputdir,
                problem=problem,
                alphas=confidence_levels,
            )
            np.save(outputdir + "/coverage.npy", coverages)
            np.save(outputdir + "/contour-sizes.npy", contour_sizes)
            np.save(outputdir + "/bias.npy", bias)
            np.save(outputdir + "/bias_square.npy", bias_square)
            np.save(outputdir + "/variance.npy", variance)

    # Evaluate the approximate log posterior of the nominal values
    @aw.cpus_and_memory(2, memory)
    @aw.dependency(train_9)
    @aw.gpus(
        ngpus, os.environ.get("VRAM_PER_GPU", None) if ngpus > 0 else None
    )
    @aw.postcondition(
        aw.num_files(root + "/*/log_posterior.npy", num_estimators)
    )
    @aw.tasks(num_estimators)
    @aw.timelimit("12:00:00" if not debug else "24:00:00")
    @aw.constraint(
        os.environ.get("GPU_CONSTRAINT", None) if ngpus > 0 else None
    )
    @aw.partition(
        os.environ.get("GPU_PARTITIONS", "shared-gpu,public-gpu").split(",")
        if ngpus > 0
        else os.environ.get("CPU_PARTITIONS", "shared-cpu,public-cpu").split(
            ","
        )
    )
    def log_posterior(task_index=0):
        outputdir = root + "/" + str(task_index)
        outputfile = outputdir + "/log_posterior.npy"
        if not os.path.exists(outputfile):
            with torch.no_grad():
                inputs = torch.from_numpy(
                    np.load(problem + "/data/coverage/inputs.npy")
                )
                outputs = torch.from_numpy(
                    np.load(problem + "/data/coverage/outputs.npy")
                )
                # Construct flow
                density_estimator = build_density_estimator(problem=problem)
                if isinstance(density_estimator, str):
                    build_neural_net = posterior_nn(model=density_estimator)
                else:
                    build_neural_net = density_estimator

                if problem in ("spatialsir", "lotka_volterra"):
                    num_examples = inputs.shape[0]
                    outputs = outputs.float().view(num_examples, -1)
                flow_sbi = build_neural_net(
                    inputs,
                    outputs,
                )
                dataloader = DataLoader(
                    TensorDataset(inputs, outputs), batch_size=1024
                )
                # Load weights for flow
                assert (
                    outputdir is not None
                ), f"Invalid path for checkpoint: {outputdir}"
                load_checkpoint(
                    model=flow_sbi,
                    checkpoint=torch.load(
                        outputdir + "/checkpoint_best.pt",
                        map_location=torch.device("cpu"),
                    ),
                    load_cp_continue=False,
                    optimizer=None,
                )
                flow_sbi = flow_sbi.to(h.accelerator)
                log_posterior = torch.empty(inputs.shape[0])
                # Compute the log posterior
                for b, (_input, _output) in enumerate(dataloader):
                    _input = _input.to(h.accelerator)
                    _output = _output.to(h.accelerator)
                    log_posterior[
                        b * 1024 : min(b * 1024 + 1024, len(log_posterior))
                    ] = compute_log_pdf(
                        r=flow_sbi,
                        inputs=_input,
                        outputs=_output,
                    ).cpu()
                np.save(outputfile, log_posterior.mean().numpy())


dependencies = None
for batch_size in batch_sizes:
    for budget in simulation_budgets:
        train_and_evaluate(budget, batch_size, dependencies=dependencies)


if __name__ == "__main__":
    aw.execute(partition=arguments.partition, name=problem)
