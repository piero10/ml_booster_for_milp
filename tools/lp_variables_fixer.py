import os
from typing import TypedDict

import pyscipopt as pso
import mip


class VarVal(TypedDict):
    name: str
    value: int


class LPVariablesFixer:

    eps = 0.00001

    def __init__(self,
               lp_file: str = None,
               lp_file_postfix: str = '_fixed'):
        '''
        lp_file - путь к lp файлу
        '''
        self.lp_file = lp_file
        self.lp_file_postfix = lp_file_postfix



    def fix(self, model: mip.Model, #: pyo.Model, #pso.Model,
            variables_values: VarVal) -> pso.Model:
        '''
        variables_values -
        '''
        for var, val in variables_values.items():
            v = model.var_by_name(var)
            v.lb = val - self.eps
            v.ub = val + self.eps

        return model


    def load_fix_and_save(self, variables_values: VarVal,
                          lp_file: str = None,
                          lp_file_postfix: str = None):
        if lp_file is None:
            lp_file = self.lp_file

        if lp_file_postfix is None:
            lp_file_postfix = self.lp_file_postfix

        model = mip.Model('cbc')
        model.read(self.lp_file)

        model = self.fix(model, variables_values)
        newfilename = lp_file.split('.')[0] + lp_file_postfix + '.' + lp_file.split('.')[1]

        model.write(newfilename)
        return None