parser = argparse.ArgumentParser()
parser.add_argument(
    "--test",
    action="store_true",
    help="Execute the workflow with fast hyper parameters for testing (default: false).",
)
parser.add_argument(
    "--only_flows",
    action="store_true",
    help="Execute only the flow part of the workflow (default: false).",
)
parser.add_argument(
    "--add_bias_variance",
    action="store_true",
    help="Compute the bias and variance of the posterior estimators (default: false).",
)
arguments, _ = parser.parse_known_args()

# Hyperparameters
batch_size = 128
learning_rate = 0.001
weight_decays = [0.01, 0.05, 0.1]

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


assert arguments.simulate_train_n % simulation_block_size == 0
num_train_blocks = int(arguments.simulate_train_n / simulation_block_size)


def conditional_decorator(dec, condition):
    def decorator(func):
        if not condition:
            return func
        return dec(func)

    return decorator


def evaluate_point_flow_sbi(
    simulation_budget, storagedir=None, cl_list=[0.95], add_bias_variance=False
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
    @w.slurm.cpu_and_memory(4, "8g")
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
    @conditional_decorator(
        w.postcondition(
            w.num_files(storagedir + "/flow-sbi-0*/bias.npy", num_ensembles)
        ),
        add_bias_variance,
    )
    @conditional_decorator(
        w.postcondition(
            w.num_files(
                storagedir + "/flow-sbi-0*/variance.npy", num_ensembles
            )
        ),
        add_bias_variance,
    )
    @conditional_decorator(
        w.postcondition(
            w.num_files(storagedir + "/flow-sbi-0*/shift.npy", num_ensembles)
        ),
        add_bias_variance,
    )
    @w.slurm.cpu_and_memory(4, "8g")
    @w.slurm.timelimit("48:00:00")
    @w.tasks(num_ensembles)
    def coverage_individual(task_index):
        resultdir = storagedir + "/flow-sbi-" + str(task_index).zfill(5)
        if not os.path.exists(resultdir + "/coverage.npy"):
            query = resultdir + "/posterior.pkl"

            if add_bias_variance:
                coverage, bias, variance, shift = coverage_of_estimator(
                    query,
                    cl_list=cl_list,
                    flow_sbi=True,
                    add_bias_variance=True,
                )
                np.save(resultdir + "/bias.npy", bias)
                np.save(resultdir + "/variance.npy", variance)
                np.save(resultdir + "/shift.npy", variance)
            else:
                coverage = coverage_of_estimator(
                    query, cl_list=cl_list, flow_sbi=True
                )

            np.save(resultdir + "/coverage.npy", coverage)

    @w.dependency(train_flow)
    @w.postcondition(w.exists(storagedir + "/coverage-flow-sbi.npy"))
    @conditional_decorator(
        w.postcondition(
            w.num_files(storagedir + "/bias-flow-sbi.npy", num_ensembles)
        ),
        add_bias_variance,
    )
    @conditional_decorator(
        w.postcondition(
            w.num_files(storagedir + "/variance-flow-sbi.npy", num_ensembles)
        ),
        add_bias_variance,
    )
    @conditional_decorator(
        w.postcondition(
            w.num_files(storagedir + "/shift-flow-sbi.npy", num_ensembles)
        ),
        add_bias_variance,
    )
    @w.slurm.cpu_and_memory(4, "8g")
    @w.slurm.timelimit("48:00:00")
    def coverage_ensemble():
        if not os.path.exists(storagedir + "/coverage-flow-sbi.npy"):
            query = storagedir + "/flow-sbi-0*/posterior.pkl"

            if add_bias_variance:
                coverage, bias, variance, shift = coverage_of_estimator(
                    query,
                    cl_list=cl_list,
                    flow_sbi=True,
                    max_samples=20000,
                    add_bias_variance=True,
                )
                np.save(storagedir + "/bias-flow-sbi.npy", bias)
                np.save(storagedir + "/variance-flow-sbi.npy", variance)
                np.save(storagedir + "/shift-flow-sbi.npy", variance)
            else:
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
    @w.slurm.cpu_and_memory(4, "8g")
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
    @w.slurm.cpu_and_memory(4, "8g")
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

    # Add the dependencies for the summary notebook.
    dependencies.append(expected_posterior_individual)
    dependencies.append(expected_posterior_ensemble)
    dependencies.append(coverage_individual)
    dependencies.append(coverage_ensemble)
    # dependencies.append(sbc_individual)
    # dependencies.append(sbc_ensemble)

    @w.dependency(train_flow_bagging)
    @w.postcondition(w.exists(storagedir + "/coverage-flow-sbi-bagging.npy"))
    @conditional_decorator(
        w.postcondition(
            w.num_files(
                storagedir + "/bias-flow-sbi-bagging.npy", num_ensembles
            )
        ),
        add_bias_variance,
    )
    @conditional_decorator(
        w.postcondition(
            w.num_files(
                storagedir + "/variance-flow-sbi-bagging.npy", num_ensembles
            )
        ),
        add_bias_variance,
    )
    @conditional_decorator(
        w.postcondition(
            w.num_files(
                storagedir + "/shift-flow-sbi-bagging.npy", num_ensembles
            )
        ),
        add_bias_variance,
    )
    @w.slurm.cpu_and_memory(4, "8g")
    @w.slurm.timelimit("48:00:00")
    def coverage_ensemble_bagging():
        if not os.path.exists(storagedir + "/coverage-flow-sbi-bagging.npy"):
            query = storagedir + "/flow-sbi-bagging-0*/posterior.pkl"

            if add_bias_variance:
                coverage, bias, variance, shift = coverage_of_estimator(
                    query,
                    cl_list=cl_list,
                    flow_sbi=True,
                    max_samples=20000,
                    add_bias_variance=True,
                )
                np.save(storagedir + "/bias-flow-sbi-bagging.npy", bias)
                np.save(
                    storagedir + "/variance-flow-sbi-bagging.npy", variance
                )
                np.save(storagedir + "/shift-flow-sbi-bagging.npy", variance)
            else:
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
    @w.slurm.cpu_and_memory(4, "8g")
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
    @w.slurm.cpu_and_memory(4, "8g")
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
    @w.slurm.cpu_and_memory(4, "8g")
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
    @conditional_decorator(
        w.postcondition(
            w.num_files(
                storagedir + "/bias-flow-sbi-static.npy", num_ensembles
            )
        ),
        add_bias_variance,
    )
    @conditional_decorator(
        w.postcondition(
            w.num_files(
                storagedir + "/variance-flow-sbi-static.npy", num_ensembles
            )
        ),
        add_bias_variance,
    )
    @conditional_decorator(
        w.postcondition(
            w.num_files(
                storagedir + "/shift-flow-sbi-static.npy", num_ensembles
            )
        ),
        add_bias_variance,
    )
    @w.slurm.cpu_and_memory(4, "8g")
    @w.slurm.timelimit("48:00:00")
    def coverage_ensemble_static():
        if not os.path.exists(storagedir + "/coverage-flow-sbi-static.npy"):
            query = storagedir + "/flow-sbi-static-0*/posterior.pkl"
            if add_bias_variance:
                coverage, bias, variance, shift = coverage_of_estimator(
                    query,
                    cl_list=cl_list,
                    flow_sbi=True,
                    max_samples=20000,
                    add_bias_variance=True,
                )
                np.save(storagedir + "/bias-flow-sbi-static.npy", bias)
                np.save(storagedir + "/variance-flow-sbi-static.npy", variance)
                np.save(storagedir + "/shift-flow-sbi-static.npy", variance)
            else:
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
    @w.slurm.cpu_and_memory(4, "8g")
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

    """
    @w.dependency(train_flow_static)
    @w.postcondition(w.exists(storagedir + "/sbc-flow-sbi-static.npy"))
    @w.slurm.cpu_and_memory(4, "8g")
    @w.slurm.timelimit("48:00:00")
    def sbc_ensemble_static():
        if not os.path.exists(storagedir + "/sbc-flow-sbi-static.npy"):
            query = storagedir + "/flow-sbi-static-0*/posterior.pkl"
            compute_sbc(query, sbc_nb_rank_samples, sbc_nb_posterior_samples, storagedir + "/sbc-flow-sbi-static.npy", flow_sbi=True)
    """

    dependencies.append(coverage_ensemble_static)
    dependencies.append(expected_posterior_ensemble_static)
    # dependencies.append(sbc_ensemble_static)


for simulation_budget in simulations:
    if arguments.only_flows:
        evaluate_point_flow_sbi(
            simulation_budget,
            cl_list=credible_interval_levels,
            add_bias_variance=arguments.add_bias_variance,
        )
    else:
        # With regularization
        evaluate_point_classifier(
            simulation_budget,
            regularization="balancing",
            cl_list=credible_interval_levels,
            add_bias_variance=arguments.add_bias_variance,
        )

        for weight_decay in weight_decays:
            evaluate_point_classifier(
                simulation_budget,
                regularization="l2",
                cl_list=credible_interval_levels,
                add_bias_variance=arguments.add_bias_variance,
                weight_decay=weight_decay,
            )
        # evaluate_point_flow(simulation_budget, regulariation="balancing", cl_list=credible_interval_levels, add_bias_variance=arguments.add_bias_variance)
        # Without regularization
        evaluate_point_classifier(
            simulation_budget,
            regularization="none",
            cl_list=credible_interval_levels,
            add_bias_variance=arguments.add_bias_variance,
        )
        # evaluate_point_flow(simulation_budget, regularization="none", cl_list=credible_interval_levels, add_bias_variance=arguments.add_bias_variance)
        evaluate_point_flow_sbi(
            simulation_budget,
            cl_list=credible_interval_levels,
            add_bias_variance=arguments.add_bias_variance,
        )
