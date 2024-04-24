"""
В этом модуле описываются базовые эвристические приемы поиска решения
в задачах линейного программирования в частично-целочисленной постановке
"""
import sys
import typing as t

import numpy as np
import pandas as pd
import pathlib2
import pyscipopt

from tools.auxiliary_functions import (
    extract_from_relax_sol_zero_vars,
    extract_vars_set_type,
    get_and_write_best_sol,
    write_best_sol,
)
from tools.SCIP_keywords import (
    BIN_0,
    BIN_1,
    BINARY,
    INT_0,
    INTEGER,
    STATUS_GAPLIMIT,
    STATUS_INFEASIBLE,
    STATUS_TIMELIMIT,
    STATUS_USERINTERRUPT,
    ZERO,
)
from utils.make_logger import logger


def find_sol_1phase(
    problem_name: str,
    relax_sol: dict,
    path_to_lp_file: pathlib2.Path,
    path_to_set_file: pathlib2.Path,
    path_to_output_dir: pathlib2.Path,
    threshold_left: float,
    threshold_right: float,
    threshold_step: float,
) -> dict:
    """
    Первая фаза поиска решения.
    Подбирает порог бинаризации для бинарных переменных
    в релаксированном решении, готовит фиксацию на бинарных переменных
    с учетом найденного порога, а затем ищет допустимое целочисленное решение
    с подготовленной фиксацией
    """
    n_steps = int((threshold_right - threshold_left) / threshold_step) + 1

    for step in range(0, n_steps):
        logger.info(f"Step #{step + 1}")

        model = pyscipopt.Model()
        model.readProblem(path_to_lp_file)
        model.readParams(path_to_set_file)

        all_vars = model.getVars()
        bin_vars: t.List[pyscipopt.scip.Variable] = extract_vars_set_type(
            all_vars, BINARY
        )

        threshold = threshold_left + step * threshold_step
        logger.info(
            f"Problem has been launched for solution with threshold={threshold:.4g} ..."
        )
        vars_gt_threshold: t.List[pyscipopt.scip.Variable] = []
        vars_lt_threshold: t.List[pyscipopt.scip.Variable] = []

        for var in bin_vars:
            value = np.abs(relax_sol[var.name])
            if value > threshold:
                vars_gt_threshold.append(var)
                model.fixVar(var, BIN_1)
            else:
                vars_lt_threshold.append(var)
                model.fixVar(var, BIN_0)

        logger.info(
            f"Number of 0-bin: {len(vars_lt_threshold)}, number of 1-bin: {len(vars_gt_threshold)}"
        )
        model.optimize()
        status = model.getStatus()

        if status == STATUS_INFEASIBLE:
            continue
        elif (
            status == STATUS_GAPLIMIT
            or status == STATUS_TIMELIMIT
            or status == STATUS_USERINTERRUPT
        ):
            # если множество допустимых решений непустое, то записать и вернуть лучшее решение
            if model.getSols():
                best_sol: dict = get_and_write_best_sol(
                    problem_name=problem_name,
                    model=model,
                    vars_=all_vars,
                    path_to_output_dir=path_to_output_dir,
                )

                return best_sol
            else:
                logger.info("Process is interrupted ...")
                sys.exit(-1)


def find_sol_2phase(
    problem_name: str,
    warm_start: dict,
    path_to_lp_file: pathlib2.Path,
    path_to_set_file: pathlib2.Path,
    path_to_output_dir: pathlib2.Path,
) -> dict:
    """
    Вторая фаза поиска решения.
    Ищет допустимое целочисленное решение на базе решения первой фазы.
    Готовит фиксацию на единичных бинарных 1-bin и нулевых целочисленных 0-ints переменных
    """
    model = pyscipopt.Model()
    model.readProblem(path_to_lp_file)
    model.readParams(path_to_set_file)

    all_vars = model.getVars()
    bin_vars: t.List[pyscipopt.scip.Variable] = extract_vars_set_type(all_vars, BINARY)
    int_vars: t.List[pyscipopt.scip.Variable] = extract_vars_set_type(all_vars, INTEGER)

    for var in bin_vars:
        value = np.round(np.abs(warm_start[var.name]))
        if np.equal(value, BIN_1):
            model.fixVar(var, BIN_1)

    for var in int_vars:
        value = np.round(np.abs(warm_start[var.name]))
        if np.equal(value, INT_0):
            model.fixVar(var, INT_0)

    model.optimize()
    status = model.getStatus()

    if status == STATUS_INFEASIBLE:
        logger.info("Oops. Infeasible solution ...")
        sys.exit(-1)
    elif (
        status == STATUS_GAPLIMIT
        or status == STATUS_TIMELIMIT
        or status == STATUS_USERINTERRUPT
    ):
        # если множество допустимых решений непустое, то записать и вернуть лучшее решение
        if model.getSols():
            best_sol: dict = get_and_write_best_sol(
                problem_name=problem_name,
                model=model,
                vars_=all_vars,
                path_to_output_dir=path_to_output_dir,
            )

            return best_sol
        else:
            logger.info("Process is interrupted ...")
            sys.exit(-1)


def find_feas_sol(
    problem_name: str,
    relax_sol: dict,
    path_to_lp_file: pathlib2.Path,
    path_to_set_file: pathlib2.Path,
    path_to_output_dir: pathlib2.Path,
) -> t.NoReturn:
    """
    Ищет допустимое целочисленное решение
    с фиксацией нулевых бинарных и нулевых целочисленных переменных
    в релаксированном решении
    """
    relax_sol = pd.Series(relax_sol)

    model = pyscipopt.Model()
    model.readProblem(path_to_lp_file)
    model.readParams(path_to_set_file)

    all_vars: t.List[pyscipopt.scip.Variable] = model.getVars()
    bin_vars: t.List[pyscipopt.scip.Variable] = extract_vars_set_type(all_vars, BINARY)
    int_vars: t.List[pyscipopt.scip.Variable] = extract_vars_set_type(all_vars, INTEGER)

    all_zero_bin_vars: t.List[
        pyscipopt.scip.Variable
    ] = extract_from_relax_sol_zero_vars(
        relax_sol,
        sub_group_vars=bin_vars,
    )
    all_zero_int_vars: t.List[
        pyscipopt.scip.Variable
    ] = extract_from_relax_sol_zero_vars(
        relax_sol,
        sub_group_vars=int_vars,
    )

    for var in all_zero_bin_vars + all_zero_int_vars:
        model.fixVar(var, ZERO)

    model.optimize()
    status = model.getStatus()

    if status == STATUS_INFEASIBLE:
        logger.info("Oops. Infeasible solution ...")
        sys.exit(-1)
    elif (
        status == STATUS_GAPLIMIT
        or status == STATUS_TIMELIMIT
        or status == STATUS_USERINTERRUPT
    ):
        # если множество допустимых решений непустое, то записать лучшее решение
        if model.getSols():
            write_best_sol(
                problem_name=problem_name,
                model=model,
                path_to_output_dir=path_to_output_dir,
            )

        else:
            logger.info("Process is interrupted ...")
            sys.exit(-1)
