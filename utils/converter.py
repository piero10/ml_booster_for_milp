'''
конвертирует mps файлы в lp, решает их, сохраняет:
 - релаксированную постановку,
 - lp файл
 - файл решения
'''

import os
import pandas as pd
from tabulate import tabulate
import pyscipopt
from timeit import default_timer as timer
import warnings

warnings.filterwarnings("ignore")

from tools.SCIP_keywords import STATUS_OPTIMAL_SOL

path_input_mps = os.path.join(os.getcwd(), 'input', 'sop_heavy', 'mps')
path_output_lp = os.path.join(os.getcwd(), 'input', 'sop_heavy', 'lp')
path_output_lp_relaxed = os.path.join(os.getcwd(), 'input', 'sop_heavy', 'lp_relaxed')
path_output_sol = os.path.join(os.getcwd(), 'input', 'sop_heavy', 'sol')

# обрабатываемые сценарии
scenarios = [
    #'tmpfvpqodxw.lp',
    #'tmpi4l6qjky.lp'

    #sop heavy
    '1664182445.0446522.lp',
    '1664182463.1948457.lp',
    '1664182471.9356806.lp',
    '1664182480.4326847.lp',
    '1664182491.7378433.lp',
    '1664182500.796735.lp',
    '1664182511.649937.lp',
    '1664182523.380519.lp',
    '1664182533.1587787.lp',
    '1664182546.82382.lp'

    #sop light
    #'1664186663_4513757.lp', # сценарии этой группы решаются за 200-400 секунд, в среднем 230 секунд
    #'1664186672_765825.lp',
    #'1664186685_8937309.lp',
    #'1664186722_313071.lp',
    #'1664186741_701503.lp',
    #'1664186792_9439187.lp',
    #'1664186865_1084983.lp',
    #'1664186875_4922888.lp',
    #'1664186889_6811075.lp',
    #'1664186899_8028727.lp'


    #'7fac4231_22.03.lp',
    #'337_22.03.lp',
    #'50197DF7_22.03.lp',
    #'A78CBEAD_22.03.lp',
    #'f398266b_25.05.lp'

    #'30_70_45_05_100.mps',     #340 sec
    #'30_70_45_095_100.mps', #167 sec
    #'comp08-2idx.mps', # 911
    #'gus-sch.mps', # 5
    #'qiu.mps', #149

    #'beasleyC3.mps',  # 1 sec
    #'mc7.mps',
    #'mc8.mps',
    #'nexp-50-20-1-1.mps',
    #'nexp-150-20-1-5.mps'
]


def load_model(file_name):
    path = os.path.join(path_input_mps, file_name)
    model = pyscipopt.Model()
    model.hideOutput(True)
    # model.setPresolve() #SCIP_PARAMSETTING.OFF)
    model.readProblem(path)
    return model


def save_relaxed_lp_file(file_name):
    '''

    '''
    model = pyscipopt.Model()
    model.readProblem(os.path.join(path_output_lp, file_name))
    for v in model.getVars():
        model.chgVarType(v, 'CONTINUOUS')
    filename_without_ext = file_name.split('.')[0]
    model.writeProblem(os.path.join(path_output_lp_relaxed, filename_without_ext + '_relaxed.lp'))

def solve_and_save(file_name):
    model = pyscipopt.Model()
    model.readProblem(os.path.join(path_output_lp, file_name))
    filename_without_ext = file_name.split('.')[0]
    start = timer()
    model.setParam('limits/time', 20 * 60)
    model.optimize()
    status = model.getStatus()
    end = timer()
    relax_sol: pyscipopt.scip.Solution = model.getBestSol()
    path_to_solution_file = os.path.join(path_output_sol, filename_without_ext + '.sol')
    model.writeSol(
        relax_sol,
        path_to_solution_file,
        write_zeros=True,
    )
    print(f'{file_name} solved. status: {status}')


def convert_mps_file(file_name,
                     save_lp_file=False,
                     save_sol_file=False,
                     save_relaxed_lp_file=False):
    """
    file_name - mps файл
    save_lp_file - сохрянять lp файл
    save_sol_file - сохрянять файл решения
    save_relaxed_lp_file - сохранять релаксированную постановку задачи
    """

    filename_without_ext = file_name.split('.')[0]
    model = load_model(file_name)

    if save_lp_file:
        model.writeProblem(os.path.join(path_output_lp, filename_without_ext + '.lp'))

    if save_sol_file:
        start = timer()
        model.setParam('limits/time', 20 * 60)
        model.optimize()
        status = model.getStatus()
        end = timer()
        relax_sol: pyscipopt.scip.Solution = model.getBestSol()
        path_to_solution_file = os.path.join(path_output_sol, filename_without_ext + '.sol')
        model.writeSol(
            relax_sol,
            path_to_solution_file,
            write_zeros=True,
        )

    if save_relaxed_lp_file:
        m = load_model(file_name)
        for v in m.getVars():
            m.chgVarType(v, 'CONTINUOUS')
        m.writeProblem(os.path.join(path_output_lp_relaxed, filename_without_ext + '.lp'))

    try:
        target_function_value = model.getObjVal()
    except:
        target_function_value = -1
    return model, status == STATUS_OPTIMAL_SOL, end - start, target_function_value


if __name__ == '__main__':
    tf = {}
    times = {}

    res = pd.DataFrame()
    for scenario in scenarios:
        print(f'scenario: {scenario:<30} processing... ===============================================================')
        save_relaxed_lp_file(scenario)
        solve_and_save(scenario)
        continue

        print(f'scenario: {scenario:<30} processing... ===============================================================')
        model, status, time, target_function_value = convert_mps_file(scenario,
                                                   save_lp_file=True,
                                                   save_sol_file=True,
                                                   save_relaxed_lp_file=True)

        result_dct = {'scenario': scenario,
                      'status': status,
                      'target_function_value': target_function_value,
                      'time': time}

        res = res.append(result_dct, ignore_index=True)

    print()
    print(tabulate(res, headers=res.columns))
