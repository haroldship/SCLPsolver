from doe.doe import run_experiment_series
from subroutines.utils import relative_to_project
from doe.cplex_integration.run_cplex_experiments import run_cplex_experiments
from doe.doe_utils import path_utils
from doe.results_producer import combine_results, write_results_to_csv
import os

def test1(output_file, overwrite_output, exp_type, exp_num, K, I, T, settings, starting_seed = 1000, **kwargs):

    results, ftrials, files, raw_tau = run_experiment_series(exp_type, exp_num, K, I, T, settings, starting_seed, **kwargs)
    pu = path_utils(os.path.expanduser('~/Box/SCLP comparison/data'))
    DATADIRd = pu.get_CPLEX_data_path()
    # cplex_results = run_cplex_experiments(DATADIRd, relative_to_project('doe/cplex_integration/mod_files/main1.mod'), files)
    # results = combine_results(results, cplex_results, 1)
    # cplex_results = run_cplex_experiments(DATADIRd, relative_to_project('doe/cplex_integration/mod_files/dual1.mod'), files)
    # results = combine_results(results, cplex_results, 1, True)
    # cplex_results = run_cplex_experiments(DATADIRd, relative_to_project('doe/cplex_integration/mod_files/main2.mod'), files)
    # results = combine_results(results, cplex_results, 2)
    # cplex_results = run_cplex_experiments(DATADIRd, relative_to_project('doe/cplex_integration/mod_files/main3.mod'), files)
    # results = combine_results(results, cplex_results, 3)
    # cplex_results = run_cplex_experiments(DATADIRd, relative_to_project('doe/cplex_integration/mod_files/main4.mod'), files)
    # results = combine_results(results, cplex_results, 4)
    # cplex_results = run_cplex_experiments(DATADIRd, relative_to_project('doe/cplex_integration/mod_files/main5.mod'), files)
    # results = combine_results(results, cplex_results, 5)
    # cplex_results = run_cplex_experiments(DATADIRd, relative_to_project('doe/cplex_integration/mod_files/dual5.mod'), files)
    # results = combine_results(results, cplex_results, 5, True)
    cplex_results = run_cplex_experiments(DATADIRd, relative_to_project('doe/cplex_integration/mod_files/main10.mod'), files)
    results = combine_results(results, cplex_results, 10)
    # cplex_results = run_cplex_experiments(DATADIRd, relative_to_project('doe/cplex_integration/mod_files/dual10.mod'), files)
    # results = combine_results(results, cplex_results, 10, True)
    # cplex_results = run_cplex_experiments(DATADIRd, relative_to_project('doe/cplex_integration/mod_files/main100.mod'), files)
    # results = combine_results(results, cplex_results, 100)
    res_file = relative_to_project(output_file)
    write_results_to_csv(results, res_file, overwrite_output, raw_tau=None)
