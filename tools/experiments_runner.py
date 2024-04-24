import os
import pandas as pd
import pyscipopt
from timeit import default_timer as timer
from multiprocessing import Pool

import warnings
warnings.filterwarnings('ignore')

class ExperimentsRunner():
    '''
    класс для проведения экспериментод для подбора оптимальных
    эвристик для группы сценариев
    '''

    def __init__(self, path_to_set_file, path_to_scenarios_folder, time_limit=30*60):
        '''
        path_to_set_file - путь к файлу базовых настроек
        path_to_scenarios_folder - директория со сценариями - lp файлами.
        '''
        self.path_to_set_file = path_to_set_file
        self.path_to_scenarios_folder = path_to_scenarios_folder
        self.time_limit = time_limit


    def run_model(self, scenario, changed_key, changed_val):
        '''
        проведение эксперимента - считается указанный сценарий с одним измененным параметром,
        параметр changed_key принимает значение changed_val

        возвращает:
        - сценарий
        - статус (оптимально/не оптимально)
        - значение целефой функции
        - имя измененного параметра
        - значение параметра
        - время вычисления
        '''
        print(f'case:  {scenario:<20}    {changed_key:<30} {changed_val:<6} in processing....')
        model = pyscipopt.Model()
        model.hideOutput(True)
        model.readParams(self.path_to_set_file)
        model.readProblem(os.path.join(self.path_to_scenarios_folder, scenario+'.lp'))
        start = timer()
        model.setParam('limits/time', self.time_limit)
        if changed_key != 'base':
            model.setParam(changed_key, changed_val)
        model.optimize()
        end = timer()
        time_calculated = int(end - start)
        status = model.getStatus()
        target_function_value = model.getObjVal()
        print(f'case:  {scenario:<20}    {changed_key:<30} {changed_val:<6} done')

        return scenario, status, int(target_function_value), changed_key, changed_val, int(time_calculated)


    def run_in_multithread(self,
                           scenarios: list,
                           iterate_params: list,
                           time_lim=5*60,
                           proc_num=4):
        '''
        запускаем множество указанных сценариев с заданными параметрами, результаты всех вычислений записываем
        в датафрейм и возвращаем
        '''
        collect = []
        for scenario in scenarios:
            for k, v in iterate_params:
                collect.append((scenario, k, v))

        with Pool(proc_num) as pool:
            res = pool.starmap(self.run_model, collect)

        res = pd.DataFrame(res, columns=['scenario', 'status', 'obj val', 'param', 'value', 'time'])
        return res


