r"""CASBI: Conservative Amortized Simulation-Based Inference

Spatial SIR problem.

TODO:
  - Change `calibration` to `balancing`
  - Move folders to prevent regeneration
"""

import argparse
import glob
import hypothesis as h
import hypothesis.workflow as w
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import papermill as pm
import shutil

from hypothesis.workflow import shell
from ratio_estimation import load_estimator, train_flow_sbi
from tqdm import tqdm
from util import coverage_of_estimator, compute_sbc
from util import measure_diagnostic, expected_log_prob
from util import simulate

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument(
    "--redo",
    action="store_true",
    help="Executes the workflow from scratch by removing all postconditions (default: false).",
)
parser.add_argument(
    "--simulate-test-n",
    type=int,
    default=100000,
    help="Number of testing simulations (default: 100 000).",
)
parser.add_argument(
    "--simulate-train-n",
    type=int,
    default=1000000,
    help="Number of training simulations (default: 10 000 000).",
)
parser.add_argument(
    "--slurm",
    action="store_true",
    help="Execute the workflow on a Slurm-enabled HPC system (default: false).",
)
parser.add_argument(
    "--test",
    action="store_true",
    help="Execute the workflow with fast hyper parameters for testing( default: false).",
)
parser.add_argument(
    "--only_flows",
    action="store_true",
    help="Execute only the flow part of the workflow (default: false).",
)
arguments, _ = parser.parse_known_args()

### BEGIN Pre-workflow #########################################################

# Pipeline constants
root = os.path.dirname(os.path.abspath(__file__))
datadir = root + "/data"
outputdir = root + "/output"

# Hyperparameters
batch_size = 128
learning_rate = 0.001

if arguments.test:
    num_ensembles = 2
    epochs = 2
    simulations = [2**n for n in range(10, 11)]
    credible_interval_levels = [0.9, 0.95]
    simulation_block_size = 10
    arguments.simulate_train_n = 3000
    arguments.simulate_test_n = 20
    sbc_nb_rank_samples = 10
    sbc_nb_posterior_samples = 100
    diagnostic_n = 10
else:
    num_ensembles = 5
    epochs = 100
    simulations = [2**n for n in range(10, 18)]
    credible_interval_levels = [x / 20 for x in range(1, 20)]
    simulation_block_size = 10000
    sbc_nb_rank_samples = 10000
    sbc_nb_posterior_samples = 1000
    diagnostic_n = 100000

# Ensure that the training dataset can be simulated in blocks of 100000
assert arguments.simulate_train_n % simulation_block_size == 0
assert arguments.simulate_test_n % simulation_block_size == 0
num_train_blocks = int(arguments.simulate_train_n / simulation_block_size)
num_test_blocks = int(arguments.simulate_test_n / simulation_block_size)

# Check if everything needs to be cleaned.
if arguments.redo:
    shutil.rmtree(datadir, ignore_errors=True)
    shutil.rmtree(outputdir, ignore_errors=True)

# Simulation arguments
datadir_simulate_test = root + "/data/test"
datadir_simulate_train = root + "/data/train"

### END Pre-workflow ###########################################################

### BEGIN Workflow definition ##################################################


@w.root
def main():
    # Prepare simulation environment
    if not os.path.exists(datadir_simulate_train):
        logging.info("Creating training data directory.")
        os.makedirs(datadir_simulate_train)
    if not os.path.exists(datadir_simulate_test):
        logging.info("Creating testing data directory.")
        os.makedirs(datadir_simulate_test)
    # Prepare the output directory
    if not os.path.exists(outputdir):
        logging.info("Creating the output directory.")
        os.makedirs(outputdir)


@w.dependency(main)
@w.postcondition(
    w.num_files(
        datadir_simulate_train + "/block-*/inputs.npy", num_train_blocks
    )
)
@w.postcondition(
    w.num_files(
        datadir_simulate_train + "/block-*/outputs.npy", num_train_blocks
    )
)
@w.slurm.cpu_and_memory(1, "8g")
@w.slurm.timelimit("01:00:00")
@w.tasks(num_train_blocks)
def simulate_train(task_index):
    output_directory = (
        datadir_simulate_train + "/block-" + str(task_index).zfill(5)
    )
    # Check if the data has already been simulated
    dont_simulate = True
    dont_simulate &= os.path.exists(output_directory + "/inputs.npy")
    dont_simulate &= os.path.exists(output_directory + "/outputs.npy")
    if not dont_simulate:
        logging.info(
            "Simulating training data block ("
            + str(task_index + 1)
            + " / "
            + str(num_train_blocks)
            + ")"
        )
        simulate(n=simulation_block_size, directory=output_directory)


@w.dependency(main)
@w.postcondition(
    w.num_files(
        datadir_simulate_test + "/block-*/inputs.npy", num_train_blocks
    )
)
@w.postcondition(
    w.num_files(
        datadir_simulate_test + "/block-*/outputs.npy", num_train_blocks
    )
)
@w.slurm.cpu_and_memory(1, "8g")
@w.slurm.timelimit("01:00:00")
@w.tasks(num_test_blocks)
def simulate_test(task_index):
    output_directory = (
        datadir_simulate_test + "/block-" + str(task_index).zfill(5)
    )
    # Check if the data has already been simulated
    dont_simulate = True
    dont_simulate &= os.path.exists(output_directory + "/inputs.npy")
    dont_simulate &= os.path.exists(output_directory + "/outputs.npy")
    if not dont_simulate:
        logging.info(
            "Simulating testing data block ("
            + str(task_index + 1)
            + " / "
            + str(num_train_blocks)
            + ")"
        )
        simulate(n=simulation_block_size, directory=output_directory)


@w.dependency(simulate_train)
@w.postcondition(w.exists(datadir_simulate_train + "/inputs.npy"))
@w.postcondition(w.exists(datadir_simulate_train + "/outputs.npy"))
@w.slurm.cpu_and_memory(1, "16g")
@w.slurm.timelimit("01:00:00")
def merge_train():
    logging.info("Merging training data.")
    cwd = os.getcwd()
    os.chdir(root)
    shell(
        "hypothesis merge --extension numpy --dimension 0 --files 'data/train/block-*/inputs.npy' --sort --out data/train/inputs.npy"
    )
    shell(
        "hypothesis merge --extension numpy --dimension 0 --files 'data/train/block-*/outputs.npy' --sort --out data/train/outputs.npy"
    )
    shell("rm -rf data/train/block-*")
    os.chdir(cwd)


@w.dependency(simulate_test)
@w.postcondition(w.exists(datadir_simulate_test + "/inputs.npy"))
@w.postcondition(w.exists(datadir_simulate_test + "/outputs.npy"))
@w.slurm.cpu_and_memory(1, "16g")
@w.slurm.timelimit("01:00:00")
def merge_test():
    logging.info("Merging testing data.")
    cwd = os.getcwd()
    os.chdir(root)
    shell(
        "hypothesis merge --extension numpy --dimension 0 --files 'data/test/block-*/inputs.npy' --sort --out data/test/inputs.npy"
    )
    shell(
        "hypothesis merge --extension numpy --dimension 0 --files 'data/test/block-*/outputs.npy' --sort --out data/test/outputs.npy"
    )
    shell("rm -rf data/test/block-*")
    os.chdir(cwd)


dependencies = []
r"""
"""


def evaluate_point_classifier(simulation_budget, regularize, cl_list=[0.95]):
    r""""""
    storagedir = outputdir + "/" + str(simulation_budget)
    if regularize:
        storagedir += "/with-regularization"
    else:
        storagedir += "/without-regularization"

    @w.dependency(merge_test)
    @w.dependency(merge_train)
    @w.postcondition(
        w.num_files(storagedir + "/mlp-0*/weights.th", num_ensembles)
    )
    @w.slurm.cpu_and_memory(4, "30g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("24:00:00")
    @w.tasks(num_ensembles)
    def train_ratio_estimator(task_index):
        resultdir = storagedir + "/mlp-" + str(task_index).zfill(5)
        os.makedirs(resultdir, exist_ok=True)
        if not os.path.exists(resultdir + "/weights.th"):
            logging.info(
                "Training classifier ratio estimator ({index} / {n}) for the Spatial SIR problem.".format(
                    index=task_index + 1, n=num_ensembles
                )
            )
            logging.info("Using the following hyper parameters:")
            logging.info(" - batch-size     : " + str(batch_size))
            logging.info(" - regularize     : " + str(regularize))
            logging.info(" - epochs         : " + str(epochs))
            logging.info(" - learning-rate  : " + str(learning_rate))
            logging.info(" - simulations    : " + str(simulation_budget))
            command = r"""python -m hypothesis.bin.ratio_estimation.train --batch-size {batch_size} \
                              --data-test ratio_estimation.DatasetJointTest \
                              --data-train ratio_estimation.DatasetJointTrain{simulations} \
                              --epochs {epochs} \
                              --estimator ratio_estimation.ClassifierRatioEstimator \
                              --hooks hooks.add \
                              --lr {lr} \
                              --lrsched-on-plateau \
                              --out {out} \
                              --show""".format(
                batch_size=batch_size,
                epochs=epochs,
                simulations=simulation_budget,
                lr=learning_rate,
                out=resultdir,
            )
            if not regularize:
                command += (
                    " --criterion hypothesis.nn.ratio_estimation.BaseCriterion"
                )
            # Execute the training procedure
            shell(command)

    @w.dependency(train_ratio_estimator)
    @w.postcondition(
        w.num_files(storagedir + "/mlp-0*/coverage.npy", num_ensembles)
    )
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("24:00:00")
    @w.tasks(num_ensembles)
    def coverage_individual(task_index):
        resultdir = storagedir + "/mlp-" + str(task_index).zfill(5)
        if not os.path.exists(resultdir + "/coverage.npy"):
            query = resultdir + "/weights.th"
            coverage = coverage_of_estimator(query, cl_list=cl_list)
            np.save(resultdir + "/coverage.npy", coverage)

    @w.dependency(train_ratio_estimator)
    @w.postcondition(w.exists(storagedir + "/coverage-classifier.npy"))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("24:00:00")
    def coverage_ensemble():
        if not os.path.exists(storagedir + "/coverage-classifier.npy"):
            query = storagedir + "/mlp-0*/weights.th"
            coverage = coverage_of_estimator(
                query, cl_list=cl_list, max_samples=20000
            )
            np.save(storagedir + "/coverage-classifier.npy", coverage)

    @w.dependency(train_ratio_estimator)
    @w.postcondition(
        w.num_files(storagedir + "/mlp-0*/diagnostic.npy", num_ensembles)
    )
    @w.slurm.cpu_and_memory(4, "8g")
    @w.slurm.timelimit("00:15:00")
    @w.tasks(num_ensembles)
    def diagnostic_individual(task_index):
        resultdir = storagedir + "/mlp-" + str(task_index).zfill(5)
        outputfile = resultdir + "/diagnostic.npy"
        if not os.path.exists(outputfile):
            query = resultdir + "/weights.th"
            r = load_estimator(query)
            result = measure_diagnostic(r, n=diagnostic_n)
            np.save(outputfile, result)

    @w.dependency(train_ratio_estimator)
    @w.postcondition(w.exists(storagedir + "/diagnostic-mlp.npy"))
    @w.slurm.cpu_and_memory(4, "8g")
    @w.slurm.timelimit("00:15:00")
    def diagnostic_ensemble():
        resultdir = storagedir
        outputfile = resultdir + "/diagnostic-mlp.npy"
        if not os.path.exists(outputfile):
            query = resultdir + "/mlp-0*/weights.th"
            r = load_estimator(query)
            result = measure_diagnostic(r, n=diagnostic_n)
            np.save(outputfile, result)

    @w.dependency(train_ratio_estimator)
    @w.postcondition(
        w.num_files(
            storagedir + "/mlp-0*/expected-posterior.npy", num_ensembles
        )
    )
    @w.postcondition(
        w.num_files(
            storagedir + "/mlp-0*/expected-posterior-10.npy", num_ensembles
        )
    )
    @w.postcondition(
        w.num_files(
            storagedir + "/mlp-0*/expected-posterior-25.npy", num_ensembles
        )
    )
    @w.postcondition(
        w.num_files(
            storagedir + "/mlp-0*/expected-posterior-50.npy", num_ensembles
        )
    )
    @w.postcondition(
        w.num_files(
            storagedir + "/mlp-0*/expected-posterior-75.npy", num_ensembles
        )
    )
    @w.postcondition(
        w.num_files(
            storagedir + "/mlp-0*/expected-posterior-90.npy", num_ensembles
        )
    )
    @w.postcondition(
        w.num_files(
            storagedir + "/mlp-0*/expected-posterior-distrib.npy",
            num_ensembles,
        )
    )
    @w.slurm.cpu_and_memory(4, "8g")
    @w.slurm.timelimit("00:15:00")
    @w.tasks(num_ensembles)
    def expected_posterior_individual(task_index):
        resultdir = storagedir + "/mlp-" + str(task_index).zfill(5)
        outputfiles = [
            resultdir + "/expected-posterior.npy",
            resultdir + "/expected-posterior-10.npy",
            resultdir + "/expected-posterior-25.npy",
            resultdir + "/expected-posterior-50.npy",
            resultdir + "/expected-posterior-75.npy",
            resultdir + "/expected-posterior-90.npy",
            resultdir + "/expected-posterior-distrib.npy",
        ]
        if any([not os.path.exists(outputfile) for outputfile in outputfiles]):
            query = resultdir + "/weights.th"
            results = expected_log_prob(query, n=diagnostic_n)
            for outputfile, result in zip(outputfiles, results):
                np.save(outputfile, result)

    @w.dependency(train_ratio_estimator)
    @w.postcondition(w.exists(storagedir + "/expected-posterior.npy"))
    @w.postcondition(w.exists(storagedir + "/expected-posterior-10.npy"))
    @w.postcondition(w.exists(storagedir + "/expected-posterior-25.npy"))
    @w.postcondition(w.exists(storagedir + "/expected-posterior-50.npy"))
    @w.postcondition(w.exists(storagedir + "/expected-posterior-75.npy"))
    @w.postcondition(w.exists(storagedir + "/expected-posterior-90.npy"))
    @w.postcondition(w.exists(storagedir + "/expected-posterior-distrib.npy"))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("02:00:00")
    def expected_posterior_ensemble():
        resultdir = storagedir
        outputfiles = [
            resultdir + "/expected-posterior.npy",
            resultdir + "/expected-posterior-10.npy",
            resultdir + "/expected-posterior-25.npy",
            resultdir + "/expected-posterior-50.npy",
            resultdir + "/expected-posterior-75.npy",
            resultdir + "/expected-posterior-90.npy",
            resultdir + "/expected-posterior-distrib.npy",
        ]
        if any([not os.path.exists(outputfile) for outputfile in outputfiles]):
            query = resultdir + "/mlp-0*/weights.th"
            results = expected_log_prob(query, n=diagnostic_n)
            for outputfile, result in zip(outputfiles, results):
                np.save(outputfile, result)

    @w.dependency(train_ratio_estimator)
    @w.postcondition(
        w.num_files(storagedir + "/mlp-0*/sbc.npy", num_ensembles)
    )
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("48:00:00")
    @w.tasks(num_ensembles)
    def sbc_individual(task_index):
        resultdir = storagedir + "/mlp-" + str(task_index).zfill(5)
        if not os.path.exists(resultdir + "/sbc.npy"):
            query = resultdir + "/weights.th"
            compute_sbc(
                query,
                sbc_nb_rank_samples,
                sbc_nb_posterior_samples,
                resultdir + "/sbc.npy",
            )

    @w.dependency(train_ratio_estimator)
    @w.postcondition(w.exists(storagedir + "/sbc-classifier.npy"))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("48:00:00")
    def sbc_ensemble():
        if not os.path.exists(storagedir + "/sbc-classifier.npy"):
            query = storagedir + "/mlp-0*/weights.th"
            compute_sbc(
                query,
                sbc_nb_rank_samples,
                sbc_nb_posterior_samples,
                storagedir + "/sbc-classifier.npy",
            )

    # Add the dependencies for the summary notebook.
    dependencies.append(diagnostic_individual)
    dependencies.append(diagnostic_ensemble)
    dependencies.append(expected_posterior_individual)
    dependencies.append(expected_posterior_ensemble)
    dependencies.append(coverage_individual)
    dependencies.append(coverage_ensemble)
    dependencies.append(sbc_individual)
    dependencies.append(sbc_ensemble)

    @w.dependency(merge_test)
    @w.dependency(merge_train)
    @w.postcondition(
        w.num_files(storagedir + "/mlp-bagging-0*/weights.th", num_ensembles)
    )
    @w.slurm.cpu_and_memory(4, "30g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("24:00:00")
    @w.tasks(num_ensembles)
    def train_ratio_estimator_bagging(task_index):
        resultdir = storagedir + "/mlp-bagging-" + str(task_index).zfill(5)
        os.makedirs(resultdir, exist_ok=True)
        if not os.path.exists(resultdir + "/weights.th"):
            logging.info(
                "Training classifier ratio estimator ({index} / {n}) for the Spatial SIR problem.".format(
                    index=task_index + 1, n=num_ensembles
                )
            )
            logging.info("Using the following hyper parameters:")
            logging.info(" - batch-size     : " + str(batch_size))
            logging.info(" - regularize     : " + str(regularize))
            logging.info(" - epochs         : " + str(epochs))
            logging.info(" - learning-rate  : " + str(learning_rate))
            logging.info(" - simulations    : " + str(simulation_budget))
            command = r"""python -m hypothesis.bin.ratio_estimation.train --batch-size {batch_size} \
                              --data-test ratio_estimation.DatasetJointTest \
                              --data-train ratio_estimation.DatasetJointTrainBagging{simulations} \
                              --epochs {epochs} \
                              --estimator ratio_estimation.ClassifierRatioEstimator \
                              --hooks hooks.add \
                              --lr {lr} \
                              --lrsched-on-plateau \
                              --out {out} \
                              --show""".format(
                batch_size=batch_size,
                epochs=epochs,
                simulations=simulation_budget,
                lr=learning_rate,
                out=resultdir,
            )
            if not regularize:
                command += (
                    " --criterion hypothesis.nn.ratio_estimation.BaseCriterion"
                )
            # Execute the training procedure
            shell(command)

    @w.dependency(train_ratio_estimator_bagging)
    @w.postcondition(w.exists(storagedir + "/coverage-classifier-bagging.npy"))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("24:00:00")
    def coverage_ensemble_bagging():
        if not os.path.exists(storagedir + "/coverage-classifier-bagging.npy"):
            query = storagedir + "/mlp-bagging-0*/weights.th"
            coverage = coverage_of_estimator(
                query, cl_list=cl_list, max_samples=20000
            )
            np.save(storagedir + "/coverage-classifier-bagging.npy", coverage)

    @w.dependency(train_ratio_estimator_bagging)
    @w.postcondition(w.exists(storagedir + "/diagnostic-mlp-bagging.npy"))
    @w.slurm.cpu_and_memory(4, "8g")
    @w.slurm.timelimit("00:15:00")
    def diagnostic_ensemble_bagging():
        resultdir = storagedir
        outputfile = resultdir + "/diagnostic-mlp-bagging.npy"
        if not os.path.exists(outputfile):
            query = resultdir + "/mlp-bagging-0*/weights.th"
            r = load_estimator(query)
            result = measure_diagnostic(r, n=diagnostic_n)
            np.save(outputfile, result)

    @w.dependency(train_ratio_estimator_bagging)
    @w.postcondition(w.exists(storagedir + "/expected-posterior-bagging.npy"))
    @w.postcondition(
        w.exists(storagedir + "/expected-posterior-bagging-10.npy")
    )
    @w.postcondition(
        w.exists(storagedir + "/expected-posterior-bagging-25.npy")
    )
    @w.postcondition(
        w.exists(storagedir + "/expected-posterior-bagging-50.npy")
    )
    @w.postcondition(
        w.exists(storagedir + "/expected-posterior-bagging-75.npy")
    )
    @w.postcondition(
        w.exists(storagedir + "/expected-posterior-bagging-90.npy")
    )
    @w.postcondition(
        w.exists(storagedir + "/expected-posterior-bagging-distrib.npy")
    )
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("02:00:00")
    def expected_posterior_ensemble_bagging():
        resultdir = storagedir
        outputfiles = [
            resultdir + "/expected-posterior-bagging.npy",
            resultdir + "/expected-posterior-bagging-10.npy",
            resultdir + "/expected-posterior-bagging-25.npy",
            resultdir + "/expected-posterior-bagging-50.npy",
            resultdir + "/expected-posterior-bagging-75.npy",
            resultdir + "/expected-posterior-bagging-90.npy",
            resultdir + "/expected-posterior-bagging-distrib.npy",
        ]
        if any([not os.path.exists(outputfile) for outputfile in outputfiles]):
            query = resultdir + "/mlp-bagging-0*/weights.th"
            results = expected_log_prob(query, n=diagnostic_n)
            for outputfile, result in zip(outputfiles, results):
                np.save(outputfile, result)

    @w.dependency(train_ratio_estimator_bagging)
    @w.postcondition(w.exists(storagedir + "/sbc-classifier-bagging.npy"))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("48:00:00")
    def sbc_ensemble_bagging():
        if not os.path.exists(storagedir + "/sbc-classifier-bagging.npy"):
            query = storagedir + "/mlp-bagging-0*/weights.th"
            compute_sbc(
                query,
                sbc_nb_rank_samples,
                sbc_nb_posterior_samples,
                storagedir + "/sbc-classifier-bagging.npy",
            )

    # Add the dependencies for the summary notebook.
    dependencies.append(diagnostic_ensemble_bagging)
    dependencies.append(expected_posterior_ensemble_bagging)
    dependencies.append(coverage_ensemble_bagging)
    dependencies.append(sbc_ensemble_bagging)

    @w.dependency(merge_test)
    @w.dependency(merge_train)
    @w.postcondition(
        w.num_files(storagedir + "/mlp-static-0*/weights.th", num_ensembles)
    )
    @w.slurm.cpu_and_memory(4, "30g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("24:00:00")
    @w.tasks(num_ensembles)
    def train_ratio_estimator_static(task_index):
        resultdir = storagedir + "/mlp-static-" + str(task_index).zfill(5)
        os.makedirs(resultdir, exist_ok=True)
        if not os.path.exists(resultdir + "/weights.th"):
            logging.info(
                "Training classifier ratio estimator ({index} / {n}) for the Spatial SIR problem.".format(
                    index=task_index + 1, n=num_ensembles
                )
            )
            logging.info("Using the following hyper parameters:")
            logging.info(" - batch-size     : " + str(batch_size))
            logging.info(" - regularize     : " + str(regularize))
            logging.info(" - epochs         : " + str(epochs))
            logging.info(" - learning-rate  : " + str(learning_rate))
            logging.info(" - simulations    : " + str(simulation_budget))
            command = r"""python -m hypothesis.bin.ratio_estimation.train --batch-size {batch_size} \
                              --data-test ratio_estimation.DatasetJointTest \
                              --data-train ratio_estimation.DatasetJointTrainStatic{simulations} \
                              --epochs {epochs} \
                              --estimator ratio_estimation.ClassifierRatioEstimator \
                              --hooks hooks.add \
                              --lr {lr} \
                              --lrsched-on-plateau \
                              --out {out} \
                              --show""".format(
                batch_size=batch_size,
                epochs=epochs,
                simulations=simulation_budget,
                lr=learning_rate,
                out=resultdir,
            )
            if not regularize:
                command += (
                    " --criterion hypothesis.nn.ratio_estimation.BaseCriterion"
                )
            # Execute the training procedure
            shell(command)

    @w.dependency(train_ratio_estimator_static)
    @w.postcondition(w.exists(storagedir + "/coverage-classifier-static.npy"))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("24:00:00")
    def coverage_ensemble_static():
        if not os.path.exists(storagedir + "/coverage-classifier-static.npy"):
            query = storagedir + "/mlp-static-0*/weights.th"
            coverage = coverage_of_estimator(
                query, cl_list=cl_list, max_samples=20000
            )
            np.save(storagedir + "/coverage-classifier-static.npy", coverage)

    @w.dependency(train_ratio_estimator_static)
    @w.postcondition(w.exists(storagedir + "/diagnostic-mlp-static.npy"))
    @w.slurm.cpu_and_memory(4, "8g")
    @w.slurm.timelimit("00:15:00")
    def diagnostic_ensemble_static():
        resultdir = storagedir
        outputfile = resultdir + "/diagnostic-mlp-static.npy"
        if not os.path.exists(outputfile):
            query = resultdir + "/mlp-static-0*/weights.th"
            r = load_estimator(query)
            result = measure_diagnostic(r, n=diagnostic_n)
            np.save(outputfile, result)

    @w.dependency(train_ratio_estimator_static)
    @w.postcondition(w.exists(storagedir + "/expected-posterior-static.npy"))
    @w.postcondition(
        w.exists(storagedir + "/expected-posterior-static-10.npy")
    )
    @w.postcondition(
        w.exists(storagedir + "/expected-posterior-static-25.npy")
    )
    @w.postcondition(
        w.exists(storagedir + "/expected-posterior-static-50.npy")
    )
    @w.postcondition(
        w.exists(storagedir + "/expected-posterior-static-75.npy")
    )
    @w.postcondition(
        w.exists(storagedir + "/expected-posterior-static-90.npy")
    )
    @w.postcondition(
        w.exists(storagedir + "/expected-posterior-static-distrib.npy")
    )
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("02:00:00")
    def expected_posterior_ensemble_static():
        resultdir = storagedir
        outputfiles = [
            resultdir + "/expected-posterior-static.npy",
            resultdir + "/expected-posterior-static-10.npy",
            resultdir + "/expected-posterior-static-25.npy",
            resultdir + "/expected-posterior-static-50.npy",
            resultdir + "/expected-posterior-static-75.npy",
            resultdir + "/expected-posterior-static-90.npy",
            resultdir + "/expected-posterior-static-distrib.npy",
        ]
        if any([not os.path.exists(outputfile) for outputfile in outputfiles]):
            query = resultdir + "/mlp-static-0*/weights.th"
            results = expected_log_prob(query, n=diagnostic_n)
            for outputfile, result in zip(outputfiles, results):
                np.save(outputfile, result)

    @w.dependency(train_ratio_estimator_static)
    @w.postcondition(w.exists(storagedir + "/sbc-classifier-static.npy"))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("48:00:00")
    def sbc_ensemble_static():
        if not os.path.exists(storagedir + "/sbc-classifier-static.npy"):
            query = storagedir + "/mlp-static-0*/weights.th"
            compute_sbc(
                query,
                sbc_nb_rank_samples,
                sbc_nb_posterior_samples,
                storagedir + "/sbc-classifier-static.npy",
            )

    # Add the dependencies for the summary notebook.
    dependencies.append(diagnostic_ensemble_static)
    dependencies.append(expected_posterior_ensemble_static)
    dependencies.append(coverage_ensemble_static)
    dependencies.append(sbc_ensemble_static)


def evaluate_point_flow(simulation_budget, regularize, cl_list=[0.95]):
    r""""""
    storagedir = outputdir + "/" + str(simulation_budget)
    if regularize:
        storagedir += "/with-regularization"
    else:
        storagedir += "/without-regularization"

    @w.dependency(merge_test)
    @w.dependency(merge_train)
    @w.postcondition(
        w.num_files(storagedir + "/flow-0*/weights.th", num_ensembles)
    )
    @w.slurm.cpu_and_memory(4, "30g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("24:00:00")
    @w.tasks(num_ensembles)
    def train_ratio_estimator(task_index):
        resultdir = storagedir + "/flow-" + str(task_index).zfill(5)
        os.makedirs(resultdir, exist_ok=True)
        if not os.path.exists(resultdir + "/weights.th"):
            learning_rate = 0.001
            logging.info(
                "Training flow ratio estimator ({index} / {n}) for the Spatial SIR problem.".format(
                    index=task_index + 1, n=num_ensembles
                )
            )
            logging.info("Using the following hyper parameters:")
            logging.info(" - batch-size     : " + str(batch_size))
            logging.info(" - regularize     : " + str(regularize))
            logging.info(" - epochs         : " + str(epochs))
            logging.info(" - learning-rate  : " + str(learning_rate))
            logging.info(" - simulations    : " + str(simulation_budget))
            command = r"""python -m hypothesis.bin.ratio_estimation.train --batch-size {batch_size} \
                              --data-test ratio_estimation.DatasetJointTest \
                              --data-train ratio_estimation.DatasetJointTrain{simulations} \
                              --epochs {epochs} \
                              --estimator ratio_estimation.FlowRatioEstimator \
                              --hooks hooks.add \
                              --lr {lr} \
                              --gamma 250.0 \
                              --lrsched-on-plateau \
                              --out {out} \
                              --show""".format(
                batch_size=batch_size,
                epochs=epochs,
                simulations=simulation_budget,
                lr=learning_rate,
                out=resultdir,
            )
            if not regularize:
                command += (
                    " --criterion hypothesis.nn.ratio_estimation.BaseCriterion"
                )
            # Execute the training procedure
            shell(command)

    @w.dependency(train_ratio_estimator)
    @w.postcondition(
        w.num_files(storagedir + "/flow-0*/coverage.npy", num_ensembles)
    )
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("24:00:00")
    @w.tasks(num_ensembles)
    def coverage_individual(task_index):
        resultdir = storagedir + "/flow-" + str(task_index).zfill(5)
        if not os.path.exists(resultdir + "/coverage.npy"):
            query = resultdir + "/weights.th"
            coverage = coverage_of_estimator(query, cl_list=cl_list)
            np.save(resultdir + "/coverage.npy", coverage)

    @w.dependency(train_ratio_estimator)
    @w.postcondition(w.exists(storagedir + "/coverage-flow.npy"))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("24:00:00")
    def coverage_ensemble():
        if not os.path.exists(storagedir + "/coverage-flow.npy"):
            query = storagedir + "/flow-0*/weights.th"
            coverage = coverage_of_estimator(
                query, cl_list=cl_list, max_samples=20000
            )
            np.save(storagedir + "/coverage-flow.npy", coverage)

    @w.dependency(train_ratio_estimator)
    @w.postcondition(
        w.num_files(storagedir + "/flow-0*/diagnostic.npy", num_ensembles)
    )
    @w.slurm.cpu_and_memory(4, "8g")
    @w.slurm.timelimit("00:15:00")
    @w.tasks(num_ensembles)
    def diagnostic_individual(task_index):
        resultdir = storagedir + "/flow-" + str(task_index).zfill(5)
        outputfile = resultdir + "/diagnostic.npy"
        if not os.path.exists(outputfile):
            query = resultdir + "/weights.th"
            r = load_estimator(query)
            result = measure_diagnostic(r)
            np.save(outputfile, result)

    @w.dependency(train_ratio_estimator)
    @w.postcondition(w.exists(storagedir + "/diagnostic-flow.npy"))
    @w.slurm.cpu_and_memory(4, "8g")
    @w.slurm.timelimit("00:15:00")
    def diagnostic_ensemble():
        resultdir = storagedir
        outputfile = resultdir + "/diagnostic-flow.npy"
        if not os.path.exists(outputfile):
            query = resultdir + "/flow-0*/weights.th"
            r = load_estimator(query)
            result = measure_diagnostic(r)
            np.save(outputfile, result)

    # Add the dependencies for the summary notebook.
    dependencies.append(diagnostic_individual)
    dependencies.append(diagnostic_ensemble)
    dependencies.append(coverage_individual)
    dependencies.append(coverage_ensemble)


def evaluate_point_flow_sbi(
    simulation_budget, storagedir=None, cl_list=[0.95]
):
    if storagedir is None:
        storagedir = (
            outputdir
            + "/"
            + str(simulation_budget)
            + "/without-regularization"
        )

    @w.dependency(simulate_test)
    @w.dependency(merge_train)
    @w.postcondition(
        w.num_files(storagedir + "/flow-sbi-0*/posterior.pkl", num_ensembles)
    )
    @w.slurm.cpu_and_memory(4, "30g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("48:00:00")
    @w.tasks(num_ensembles)
    def train_flow(task_index):
        resultdir = storagedir + "/flow-sbi-" + str(task_index).zfill(5)
        os.makedirs(resultdir, exist_ok=True)
        train_flow_sbi(
            simulation_budget,
            epochs,
            learning_rate,
            batch_size,
            resultdir,
            task_index,
        )

    @w.dependency(train_flow)
    @w.postcondition(
        w.num_files(storagedir + "/flow-sbi-0*/coverage.npy", num_ensembles)
    )
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("48:00:00")
    @w.tasks(num_ensembles)
    def coverage_individual(task_index):
        resultdir = storagedir + "/flow-sbi-" + str(task_index).zfill(5)
        if not os.path.exists(resultdir + "/coverage.npy"):
            query = resultdir + "/posterior.pkl"
            coverage = coverage_of_estimator(
                query, cl_list=cl_list, flow_sbi=True
            )
            np.save(resultdir + "/coverage.npy", coverage)

    @w.dependency(train_flow)
    @w.postcondition(w.exists(storagedir + "/coverage-flow-sbi.npy"))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("96:00:00")
    def coverage_ensemble():
        if not os.path.exists(storagedir + "/coverage-flow-sbi.npy"):
            query = storagedir + "/flow-sbi-0*/posterior.pkl"
            coverage = coverage_of_estimator(
                query, cl_list=cl_list, flow_sbi=True, max_samples=20000
            )
            np.save(storagedir + "/coverage-flow-sbi.npy", coverage)

    @w.dependency(train_flow)
    @w.postcondition(
        w.num_files(
            storagedir + "/flow-sbi-0*/expected-posterior.npy", num_ensembles
        )
    )
    @w.postcondition(
        w.num_files(
            storagedir + "/flow-sbi-0*/expected-posterior-10.npy",
            num_ensembles,
        )
    )
    @w.postcondition(
        w.num_files(
            storagedir + "/flow-sbi-0*/expected-posterior-25.npy",
            num_ensembles,
        )
    )
    @w.postcondition(
        w.num_files(
            storagedir + "/flow-sbi-0*/expected-posterior-50.npy",
            num_ensembles,
        )
    )
    @w.postcondition(
        w.num_files(
            storagedir + "/flow-sbi-0*/expected-posterior-75.npy",
            num_ensembles,
        )
    )
    @w.postcondition(
        w.num_files(
            storagedir + "/flow-sbi-0*/expected-posterior-90.npy",
            num_ensembles,
        )
    )
    @w.postcondition(
        w.num_files(
            storagedir + "/flow-sbi-0*/expected-posterior-distrib.npy",
            num_ensembles,
        )
    )
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("02:00:00")
    @w.tasks(num_ensembles)
    def expected_posterior_individual(task_index):
        resultdir = storagedir + "/flow-sbi-" + str(task_index).zfill(5)
        outputfiles = [
            resultdir + "/expected-posterior.npy",
            resultdir + "/expected-posterior-10.npy",
            resultdir + "/expected-posterior-25.npy",
            resultdir + "/expected-posterior-50.npy",
            resultdir + "/expected-posterior-75.npy",
            resultdir + "/expected-posterior-90.npy",
            resultdir + "/expected-posterior-distrib.npy",
        ]
        if any([not os.path.exists(outputfile) for outputfile in outputfiles]):
            query = resultdir + "/posterior.pkl"
            results = expected_log_prob(query, n=diagnostic_n, flow_sbi=True)
            for outputfile, result in zip(outputfiles, results):
                np.save(outputfile, result)

    @w.dependency(train_flow)
    @w.postcondition(w.exists(storagedir + "/expected-posterior-flow-sbi.npy"))
    @w.postcondition(
        w.exists(storagedir + "/expected-posterior-flow-sbi-10.npy")
    )
    @w.postcondition(
        w.exists(storagedir + "/expected-posterior-flow-sbi-25.npy")
    )
    @w.postcondition(
        w.exists(storagedir + "/expected-posterior-flow-sbi-50.npy")
    )
    @w.postcondition(
        w.exists(storagedir + "/expected-posterior-flow-sbi-75.npy")
    )
    @w.postcondition(
        w.exists(storagedir + "/expected-posterior-flow-sbi-90.npy")
    )
    @w.postcondition(
        w.exists(storagedir + "/expected-posterior-flow-sbi-distrib.npy")
    )
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("02:00:00")
    def expected_posterior_ensemble():
        resultdir = storagedir
        outputfiles = [
            resultdir + "/expected-posterior-flow-sbi.npy",
            resultdir + "/expected-posterior-flow-sbi-10.npy",
            resultdir + "/expected-posterior-flow-sbi-25.npy",
            resultdir + "/expected-posterior-flow-sbi-50.npy",
            resultdir + "/expected-posterior-flow-sbi-75.npy",
            resultdir + "/expected-posterior-flow-sbi-90.npy",
            resultdir + "/expected-posterior-flow-sbi-distrib.npy",
        ]
        if any([not os.path.exists(outputfile) for outputfile in outputfiles]):
            query = resultdir + "/flow-sbi-0*/posterior.pkl"
            results = expected_log_prob(query, n=diagnostic_n, flow_sbi=True)
            for outputfile, result in zip(outputfiles, results):
                np.save(outputfile, result)

    @w.dependency(train_flow)
    @w.postcondition(
        w.num_files(storagedir + "/flow-sbi-0*/sbc.npy", num_ensembles)
    )
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("48:00:00")
    @w.tasks(num_ensembles)
    def sbc_individual(task_index):
        resultdir = storagedir + "/flow-sbi-" + str(task_index).zfill(5)
        if not os.path.exists(resultdir + "/sbc.npy"):
            query = resultdir + "/posterior.pkl"
            compute_sbc(
                query,
                sbc_nb_rank_samples,
                sbc_nb_posterior_samples,
                resultdir + "/sbc.npy",
                flow_sbi=True,
            )

    @w.dependency(train_flow)
    @w.postcondition(w.exists(storagedir + "/sbc-flow-sbi.npy"))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("48:00:00")
    def sbc_ensemble():
        if not os.path.exists(storagedir + "/sbc-flow-sbi.npy"):
            query = storagedir + "/flow-sbi-0*/posterior.pkl"
            compute_sbc(
                query,
                sbc_nb_rank_samples,
                sbc_nb_posterior_samples,
                storagedir + "/sbc-flow-sbi.npy",
                flow_sbi=True,
            )

    # Add the dependencies for the summary notebook.
    dependencies.append(expected_posterior_individual)
    dependencies.append(expected_posterior_ensemble)
    dependencies.append(coverage_individual)
    dependencies.append(coverage_ensemble)
    dependencies.append(sbc_individual)
    dependencies.append(sbc_ensemble)

    @w.dependency(simulate_test)
    @w.dependency(merge_train)
    @w.postcondition(
        w.num_files(
            storagedir + "/flow-sbi-bagging-0*/posterior.pkl", num_ensembles
        )
    )
    @w.slurm.cpu_and_memory(4, "30g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("48:00:00")
    @w.tasks(num_ensembles)
    def train_flow_bagging(task_index):
        resultdir = (
            storagedir + "/flow-sbi-bagging-" + str(task_index).zfill(5)
        )
        os.makedirs(resultdir, exist_ok=True)
        train_flow_sbi(
            simulation_budget,
            epochs,
            learning_rate,
            batch_size,
            resultdir,
            task_index,
            bagging=True,
        )

    @w.dependency(train_flow_bagging)
    @w.postcondition(w.exists(storagedir + "/coverage-flow-sbi-bagging.npy"))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("96:00:00")
    def coverage_ensemble_bagging():
        if not os.path.exists(storagedir + "/coverage-flow-sbi-bagging.npy"):
            query = storagedir + "/flow-sbi-bagging-0*/posterior.pkl"
            coverage = coverage_of_estimator(
                query, cl_list=cl_list, flow_sbi=True, max_samples=20000
            )
            np.save(storagedir + "/coverage-flow-sbi-bagging.npy", coverage)

    @w.dependency(train_flow_bagging)
    @w.postcondition(
        w.exists(storagedir + "/expected-posterior-flow-sbi-bagging.npy")
    )
    @w.postcondition(
        w.exists(storagedir + "/expected-posterior-flow-sbi-bagging-10.npy")
    )
    @w.postcondition(
        w.exists(storagedir + "/expected-posterior-flow-sbi-bagging-25.npy")
    )
    @w.postcondition(
        w.exists(storagedir + "/expected-posterior-flow-sbi-bagging-50.npy")
    )
    @w.postcondition(
        w.exists(storagedir + "/expected-posterior-flow-sbi-bagging-75.npy")
    )
    @w.postcondition(
        w.exists(storagedir + "/expected-posterior-flow-sbi-bagging-90.npy")
    )
    @w.postcondition(
        w.exists(
            storagedir + "/expected-posterior-flow-sbi-bagging-distrib.npy"
        )
    )
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("02:00:00")
    def expected_posterior_ensemble_bagging():
        resultdir = storagedir
        outputfiles = [
            resultdir + "/expected-posterior-flow-sbi-bagging.npy",
            resultdir + "/expected-posterior-flow-sbi-bagging-10.npy",
            resultdir + "/expected-posterior-flow-sbi-bagging-25.npy",
            resultdir + "/expected-posterior-flow-sbi-bagging-50.npy",
            resultdir + "/expected-posterior-flow-sbi-bagging-75.npy",
            resultdir + "/expected-posterior-flow-sbi-bagging-90.npy",
            resultdir + "/expected-posterior-flow-sbi-bagging-distrib.npy",
        ]
        if any([not os.path.exists(outputfile) for outputfile in outputfiles]):
            query = resultdir + "/flow-sbi-bagging-0*/posterior.pkl"
            results = expected_log_prob(query, n=diagnostic_n, flow_sbi=True)
            for outputfile, result in zip(outputfiles, results):
                np.save(outputfile, result)

    @w.dependency(train_flow_bagging)
    @w.postcondition(w.exists(storagedir + "/sbc-flow-sbi-bagging.npy"))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("48:00:00")
    def sbc_ensemble_bagging():
        if not os.path.exists(storagedir + "/sbc-flow-sbi-bagging.npy"):
            query = storagedir + "/flow-sbi-bagging-0*/posterior.pkl"
            compute_sbc(
                query,
                sbc_nb_rank_samples,
                sbc_nb_posterior_samples,
                storagedir + "/sbc-flow-sbi-bagging.npy",
                flow_sbi=True,
            )

    # Add the dependencies for the summary notebook.
    dependencies.append(coverage_ensemble_bagging)
    dependencies.append(expected_posterior_ensemble_bagging)
    dependencies.append(sbc_ensemble_bagging)

    @w.dependency(simulate_test)
    @w.dependency(merge_train)
    @w.postcondition(
        w.num_files(
            storagedir + "/flow-sbi-static-0*/posterior.pkl", num_ensembles
        )
    )
    @w.slurm.cpu_and_memory(4, "30g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("48:00:00")
    @w.tasks(num_ensembles)
    def train_flow_static(task_index):
        resultdir = storagedir + "/flow-sbi-static-" + str(task_index).zfill(5)
        os.makedirs(resultdir, exist_ok=True)
        train_flow_sbi(
            simulation_budget,
            epochs,
            learning_rate,
            batch_size,
            resultdir,
            task_index,
            static=True,
        )

    @w.dependency(train_flow_static)
    @w.postcondition(w.exists(storagedir + "/coverage-flow-sbi-static.npy"))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("96:00:00")
    def coverage_ensemble_static():
        if not os.path.exists(storagedir + "/coverage-flow-sbi-static.npy"):
            query = storagedir + "/flow-sbi-static-0*/posterior.pkl"
            coverage = coverage_of_estimator(
                query, cl_list=cl_list, flow_sbi=True, max_samples=20000
            )
            np.save(storagedir + "/coverage-flow-sbi-static.npy", coverage)

    @w.dependency(train_flow_static)
    @w.postcondition(
        w.exists(storagedir + "/expected-posterior-flow-sbi-static.npy")
    )
    @w.postcondition(
        w.exists(storagedir + "/expected-posterior-flow-sbi-static-10.npy")
    )
    @w.postcondition(
        w.exists(storagedir + "/expected-posterior-flow-sbi-static-25.npy")
    )
    @w.postcondition(
        w.exists(storagedir + "/expected-posterior-flow-sbi-static-50.npy")
    )
    @w.postcondition(
        w.exists(storagedir + "/expected-posterior-flow-sbi-static-75.npy")
    )
    @w.postcondition(
        w.exists(storagedir + "/expected-posterior-flow-sbi-static-90.npy")
    )
    @w.postcondition(
        w.exists(
            storagedir + "/expected-posterior-flow-sbi-static-distrib.npy"
        )
    )
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("02:00:00")
    def expected_posterior_ensemble_static():
        resultdir = storagedir
        outputfiles = [
            resultdir + "/expected-posterior-flow-sbi-static.npy",
            resultdir + "/expected-posterior-flow-sbi-static-10.npy",
            resultdir + "/expected-posterior-flow-sbi-static-25.npy",
            resultdir + "/expected-posterior-flow-sbi-static-50.npy",
            resultdir + "/expected-posterior-flow-sbi-static-75.npy",
            resultdir + "/expected-posterior-flow-sbi-static-90.npy",
            resultdir + "/expected-posterior-flow-sbi-static-distrib.npy",
        ]
        if any([not os.path.exists(outputfile) for outputfile in outputfiles]):
            query = resultdir + "/flow-sbi-static-0*/posterior.pkl"
            results = expected_log_prob(query, n=diagnostic_n, flow_sbi=True)
            for outputfile, result in zip(outputfiles, results):
                np.save(outputfile, result)

    @w.dependency(train_flow_static)
    @w.postcondition(w.exists(storagedir + "/sbc-flow-sbi-static.npy"))
    @w.slurm.cpu_and_memory(4, "16g")
    @w.slurm.gpu(1)
    @w.slurm.timelimit("48:00:00")
    def sbc_ensemble_static():
        if not os.path.exists(storagedir + "/sbc-flow-sbi-static.npy"):
            query = storagedir + "/flow-sbi-static-0*/posterior.pkl"
            compute_sbc(
                query,
                sbc_nb_rank_samples,
                sbc_nb_posterior_samples,
                storagedir + "/sbc-flow-sbi-static.npy",
                flow_sbi=True,
            )

    # Add the dependencies for the summary notebook.
    dependencies.append(coverage_ensemble_static)
    dependencies.append(expected_posterior_ensemble_static)
    dependencies.append(sbc_ensemble_static)


for simulation_budget in simulations:
    if arguments.only_flows:
        evaluate_point_flow_sbi(
            simulation_budget, cl_list=credible_interval_levels
        )
    else:
        # With regularization
        # evaluate_point_classifier(simulation_budget, regularize=True, cl_list=credible_interval_levels)
        # evaluate_point_flow(simulation_budget, regularize=True, cl_list=credible_interval_levels)
        # Without regularization
        evaluate_point_classifier(
            simulation_budget,
            regularize=False,
            cl_list=credible_interval_levels,
        )
        # evaluate_point_flow(simulation_budget, regularize=False, cl_list=credible_interval_levels)
        evaluate_point_flow_sbi(
            simulation_budget, cl_list=credible_interval_levels
        )


# @w.dependency(dependencies)
# @w.postcondition(w.exists(outputdir + "/summary.ipynb"))
# @w.slurm.cpu_and_memory(4, "16g")
# @w.slurm.timelimit("24:00:00")
# def summarize():
#     logging.info("Generating summary notebook.")
#     pm.execute_notebook(
#         root + "/summary.ipynb",
#         outputdir + "/summary.ipynb",
#         parameters={
#             "root": root,
#             "datadir": datadir,
#             "outputdir": outputdir})


### END Workflow definition ####################################################

# Execute the workflow
if __name__ == "__main__":
    if arguments.slurm:
        w.slurm.execute(directory=root)
    else:
        w.local.execute()
