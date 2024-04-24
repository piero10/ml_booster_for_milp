import os
import sys
import typing as t

import marshmallow_dataclass
import numpy as np
import pandas as pd
import pathlib2
import pyscipopt
import yaml
from marshmallow import ValidationError
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from utils.artifacts import ZERO
from utils.make_logger import logger
from utils.schema import ProjectSchema
from utils.SCIP_keywords import *


class TSNEComponentsNumberError(Exception):
    """
    Ошибка числа компонент для процедуры t-SNE
    """


class SingleDetectorError(Exception):
    """
    Ошибка неоднозначности выбора детектора
    """


class NumberDetectorsError(Exception):
    """
    Ошибка числа детекторов. Должен быть выбран хотя бы один детектор
    """


def read_config_yaml_file(path_to_config_file: pathlib2.Path) -> ProjectSchema:
    """
    Читает конфигурационный yaml-файл с последующей валидацией схемы
    """
    try:
        with open(path_to_config_file, encoding="utf-8") as file:
            project_schema = marshmallow_dataclass.class_schema(ProjectSchema)()
            config: ProjectSchema = project_schema.load(yaml.safe_load(file))
    except FileNotFoundError as err:
        logger.error(f"{err}")
        sys.exit(-1)
    except ValidationError as err:
        for yaml_attr_name, error_message in err.messages.items():
            logger.error(f"{yaml_attr_name}, {error_message}")
        sys.exit(-1)
    else:
        logger.info(f"File `{path_to_config_file}` has been read successfully.")

    return config


def Xy_split(
    Xy_train: pd.DataFrame,
    Xy_test: pd.DataFrame,
    target_name: str,
) -> t.Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Разбивает Xy-матрицу на матрицу признаков X и целевой вектор y
    обучающего и тестового поднабора данных
    """
    X_train: pd.DataFrame = Xy_train.drop([target_name], axis=1)
    X_test: pd.DataFrame = Xy_test.drop([target_name], axis=1)

    y_train: pd.Series = Xy_train.loc[:, target_name]
    y_train = make_target_binary(target=y_train)

    y_test: pd.Series = Xy_test.loc[:, target_name]
    y_test = make_target_binary(target=y_test)

    return X_train, X_test, y_train, y_test


def run_SCIP(
    path_to_test_lp_file: pathlib2.Path,
    path_to_set_file: pathlib2.Path,
    path_to_output_dir: pathlib2.Path,
    y_pred_test: pd.Series,
    problem_name: str,
) -> t.NoReturn:
    """
    Запускает решатель SCIP
    """

    logger.info("Solver preparation ...")
    test_lp_file_name: str = path_to_test_lp_file.name
    model = pyscipopt.Model()

    model.readProblem(path_to_test_lp_file)
    model.readParams(path_to_set_file)

    all_vars: t.List[pyscipopt.scip.Variable] = model.getVars()
    all_var_names: t.List[str] = [var.name for var in all_vars]
    all_vars: pd.Series = pd.Series(all_vars, index=all_var_names)

    y_pred_test_zero_bins_ints: pd.Series = y_pred_test.loc[y_pred_test == 0]
    y_pred_test_zero_bin_int_var_names: t.List[str] = y_pred_test_zero_bins_ints.index.to_list()

    y_pred_test_zero_bin_int_vars: t.List[pyscipopt.scip.Variable] = all_vars.loc[
        y_pred_test_zero_bin_int_var_names
    ]

    for var in tqdm(y_pred_test_zero_bin_int_vars):
        model.fixVar(var, ZERO)

    logger.info(f"Starting solver SCIP for {test_lp_file_name} ...")
    model.optimize()
    status = model.getStatus()

    if status == STATUS_INFEASIBLE:
        logger.info("Oops. Infeasible solution ...")
        sys.exit(-1)
    elif status == STATUS_GAPLIMIT or status == STATUS_TIMELIMIT or status == STATUS_USERINTERRUPT:
        # Если множество допустимых решений непустое, то записать лучшее решение
        if model.getSols():
            write_best_sol(
                problem_name=problem_name,
                model=model,
                path_to_output_dir=pathlib2.Path(path_to_output_dir),
            )

        else:
            logger.info("Process is interrupted ...")
            sys.exit(-1)

    return model


def write_best_sol(
    problem_name: str,
    model: pyscipopt.scip.Model,
    path_to_output_dir: pathlib2.Path,
) -> t.NoReturn:
    """
    Записывает лучшее найденное решение
    """
    best_sol = model.getBestSol()
    primal_bound = model.getPrimalbound()
    gap = model.getGap()

    logger.info(
        f"Feasible solution found. SCIP objective: {model.getObjVal():.5g} (gap: {gap * 100:.4g}%)"
    )
    test_lp_file_name: str = model.getProbName().split(os.sep)[-1].split(".")[0]

    path_to_best_sol_file = path_to_output_dir.joinpath(
        pathlib2.Path(f"{problem_name}_{test_lp_file_name}_{primal_bound:.5g}.sol")
    )
    path_to_stat_file = path_to_output_dir.joinpath(
        pathlib2.Path(f"{problem_name}_{test_lp_file_name}.stat")
    )

    try:
        model.writeSol(best_sol, path_to_best_sol_file, write_zeros=True)
        model.writeStatistics(path_to_stat_file)
    except OSError as err:
        logger.error(f"{err}")
        sys.exit(-1)
    else:
        logger.info(
            f"Files `{path_to_best_sol_file}` "
            f"and '{path_to_stat_file}' was successfully written."
        )


def detector_fit_predict(
    detector,
    test_lp_file_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.Series:
    """
    Обучает детектор на обучающем поднаборе данных
    и строит прогноз на тестовом поднаборе
    """
    logger.info(f"Training anomaly detector started. X_train.shape: {X_train.shape} ...")
    detector.fit(X_train)
    logger.info(f"Detector making predict for '{test_lp_file_name}'...")
    y_pred_test = pd.Series(detector.predict(X_test), index=y_test.index)
    logger.info(f"y_pred_test.values_count(): {y_pred_test.value_counts().to_dict()}")

    if not y_test.isna().all():
        logger.info(
            f"""
        Metrics:
        - F1-score: {f1_score(y_test, y_pred_test):.5f},
        - Precision: {precision_score(y_test, y_pred_test):.5f},
        - Recall: {recall_score(y_test, y_pred_test):.5f}"""
        )

    return y_pred_test


def make_target_binary(target: pd.Series) -> pd.Series:
    """
    Приводит целевой вектор к бинарному формату <<нулевые - ненулевые переменные>>
    """
    if target.isna().all():
        return target
    else:
        return pd.Series((~(target == ZERO)).astype(np.int_), index=target.index)


def parse_sol_file(path_to_sol_file: pathlib2.Path) -> dict:
    """
    Парсит SCIP'ий sol-файл в словарь
    """
    parsed_sol_file: t.Dict[str, float] = {}

    with open(path_to_sol_file, encoding="utf-8") as sol_file:
        for line in sol_file:
            if line.startswith("sol") or line.startswith("obj") or line.startswith("#"):
                continue
            var_name, value = line.split()[:2]
            parsed_sol_file[var_name] = float(value)

    return parsed_sol_file


def SCIP_sol_to_dict(
    model: pyscipopt.scip.Model,
    sol: pyscipopt.scip.Solution,
    vars_: t.List[pyscipopt.scip.Variable],
) -> dict:
    """
    Преобразует SCIP'ий sol-объект в словарь
    """
    return {var.name: model.getSolVal(sol, var) for var in vars_}


def find_relax_sol(
    problem_name: str,
    path_to_lp_file: pathlib2.Path,
    path_to_output_dir: pathlib2.Path,
) -> dict:
    """
    Ищет решение LP-задачи в релаксированной постановке
    """
    model = pyscipopt.Model()
    model.readProblem(path_to_lp_file)

    all_vars: t.List[pyscipopt.scip.Variable] = model.getVars()
    for var in all_vars:
        model.chgVarType(var, CONTINUOUS)

    model.optimize()
    status = model.getStatus()
    if status == STATUS_OPTIMAL_SOL:
        logger.info(
            f"Optimal relax solution found. Objective: {model.getObjVal():.5g}, "
            f"gap: {model.getGap() * 100:.4g}%"
        )

        relax_sol: pyscipopt.scip.Solution = model.getBestSol()

        if not path_to_output_dir.exists():
            path_to_output_dir.mkdir()

        model.writeSol(
            relax_sol,
            path_to_output_dir.joinpath(pathlib2.Path(f"{problem_name}_relax.sol")),
            write_zeros=True,
        )
        relax_sol: dict = SCIP_sol_to_dict(model, relax_sol, all_vars)

        return relax_sol
    elif status == STATUS_INFEASIBLE:
        # Bug: SCIP 8.0.0 (PySCIPOpt 4.2.0, PySCIPOpt 4.0.0) считает,
        # что задача в релаксированной постановке приводит к недопустимому решению!
        # SCIP 7.0.3 (PySCIPOpt 3.4.0) как на ОС Windows 10, так и Unix-подобных операционных
        # системах работает корректно
        logger.info(
            "Oops. Infeasible solution. NB!: Check PySCIPOpt version. "
            "Probably this is a bug of PySCIPOpt 4.[0,2].0 ..."
        )
        sys.exit(-1)
    else:
        logger.info("Unknown status ...")
        sys.exit(-1)
