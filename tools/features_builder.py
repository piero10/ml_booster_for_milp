import sys
import typing as t

import numpy as np
import pandas as pd
import pathlib2
import pyscipopt

from tools.auxiliary_functions import SCIP_sol_to_dict, parse_sol_file
from utils.artifacts import COL_NAME_LP_FILE_NAME, COL_NAME_VAR_NAME, NDARRAY, NONE
from utils.make_logger import logger
from utils.schema import ProjectSchema
from utils.SCIP_keywords import (
    BINARY,
    CONTINUOUS,
    INTEGER,
    STATUS_INFEASIBLE,
    STATUS_OPTIMAL_SOL,
)


class FeatureMatrixBuilder:
    """
    Строит матрицу признакового описания объекта
    """

    def __init__(self, config: ProjectSchema, only_for_test_set: bool = False):
        self._decimals: int = config.tolerances.decimals
        self._target_name: str = config.main_params.target_name

        self._path_to_lp_files_dir = pathlib2.Path(config.paths.path_to_lp_files)
        self._path_to_test_lp_file = pathlib2.Path(config.paths.path_to_test_lp_file)
        _path_to_test_sol_file: str = config.paths.path_to_test_sol_file
        self._path_to_test_sol_file: t.Optional[pathlib2.Path] = (
            pathlib2.Path(_path_to_test_sol_file)
            if _path_to_test_sol_file.upper() != NONE
            else None
        )
        self._path_to_sol_files_dir = pathlib2.Path(config.paths.path_to_sol_files)
        self._path_to_output_dir = pathlib2.Path(config.paths.path_to_output_dir)

        self._relax_methods_flag: bool = config.features.relax_methods.use
        self._relax_methods: t.List[str] = config.features.relax_methods.method_names

        self._avg_bin_thresholds_flag: bool = config.features.avg_bin_thresholds.use
        self._min_thresholds_value: float = config.features.avg_bin_thresholds.min_threshold_value
        self._max_thresholds_value: float = config.features.avg_bin_thresholds.max_threshold_value
        self._n_thresholds: int = config.features.avg_bin_thresholds.n_thresholds

        self._obj_coeffs_flag: bool = config.features.obj_coeffs.use
        self._number_pos_and_neg_coeffs_flag: bool = config.features.number_pos_and_neg_coeffs.use

        if only_for_test_set:
            # Построить матрицу признакового описания объекта только для тестового поднабора данных
            self.Xy_test = self._make_Xy_test(
                path_to_test_lp_file=self._path_to_test_lp_file,
                path_to_test_sol_file=self._path_to_test_sol_file,
            )
        else:
            # Вычислить матрицу признакового описания объекта для обучающего и тествого поднаборов данных
            self.Xy_train = self._make_Xy_train(
                path_to_lp_files_dir=self._path_to_lp_files_dir,
                path_to_sol_files_dir=self._path_to_sol_files_dir,
            )
            self.Xy_test = self._make_Xy_test(
                path_to_test_lp_file=self._path_to_test_lp_file,
                path_to_test_sol_file=self._path_to_test_sol_file,
            )

    def _make_Xy_train(
        self,
        path_to_lp_files_dir: pathlib2.Path,
        path_to_sol_files_dir: pathlib2.Path,
    ) -> pd.DataFrame:
        """
        Строит матрицу признакового описания объекта
        на сценариях, которые войдут в обучающий поднабор данных
        """
        _Xy_train_parts: t.List[pd.DataFrame] = []
        self.train_lp_files: t.List[str] = []

        for lp_file in path_to_lp_files_dir.iterdir():
            if lp_file.name == self._path_to_test_lp_file.name:
                continue
            for sol_file in path_to_sol_files_dir.iterdir():
                lp_file_name: str = lp_file.name.split(".")[0]
                if lp_file_name in sol_file.name:
                    self.train_lp_files.append(lp_file.name)
                    bin_int_var_names, target = self._extract_bin_int_var_names_and_target(
                        path_to_lp_file=lp_file,
                        path_to_sol_file=sol_file,
                    )
                    _Xy_train: pd.DataFrame = self._make_feature_matrix(
                        path_to_lp_file=lp_file,
                        bin_and_int_var_names=bin_int_var_names,
                    )
                    _Xy_train[COL_NAME_LP_FILE_NAME] = lp_file_name
                    _Xy_train_parts.append(pd.concat([_Xy_train, target], axis=1))

        Xy_train: pd.DataFrame = pd.concat(_Xy_train_parts, axis=0)
        Xy_train.index.name = COL_NAME_VAR_NAME

        return Xy_train.drop([COL_NAME_LP_FILE_NAME], axis=1)

    def _make_Xy_test(
        self,
        path_to_test_lp_file: pathlib2.Path,
        path_to_test_sol_file: t.Optional[pathlib2.Path],
    ) -> pd.DataFrame:
        """
        Строит матрицу признакового описания объекта
        на сценарии, который войдет в тестовый поднабор данных
        """
        bin_int_var_names, target = self._extract_bin_int_var_names_and_target(
            path_to_lp_file=self._path_to_test_lp_file,
            path_to_sol_file=path_to_test_sol_file,
        )
        _Xy_test: pd.DataFrame = self._make_feature_matrix(
            path_to_lp_file=path_to_test_lp_file, bin_and_int_var_names=bin_int_var_names
        )
        _Xy_test[COL_NAME_LP_FILE_NAME] = path_to_test_lp_file.name.split(".")[0]
        if type(target).__name__ == NDARRAY:
            target: pd.Series = pd.Series(target, index=_Xy_test.index, name=self._target_name)

        Xy_test: pd.DataFrame = pd.concat([_Xy_test, target], axis=1)
        Xy_test.index.name = COL_NAME_VAR_NAME

        return Xy_test.drop([COL_NAME_LP_FILE_NAME], axis=1)

    def _extract_bin_int_var_names_and_target(
        self,
        path_to_lp_file: pathlib2.Path,
        path_to_sol_file: t.Optional[pathlib2.Path],
    ) -> t.Tuple[t.List[str], t.Union[pd.Series, np.array]]:
        """
        Извлекает имена бинарных и целочисленных переменных,
        а также целевой вектор
        """
        model = pyscipopt.Model()
        model.readProblem(path_to_lp_file)

        all_vars: t.List[pyscipopt.scip.Variable] = model.getVars()
        bin_int_vars: t.List[pyscipopt.scip.Variable] = [
            var for var in all_vars if (var.vtype() == BINARY) or (var.vtype() == INTEGER)
        ]
        number_bin_int_vars: int = len(bin_int_vars)
        bin_int_var_names: t.List[str] = [var.name for var in bin_int_vars]

        if path_to_sol_file is None:
            array_nans = np.empty(shape=number_bin_int_vars)
            array_nans.fill(np.nan)
            target: np.array = array_nans
        else:
            target: pd.Series = np.round(
                np.abs(
                    pd.Series(parse_sol_file(path_to_sol_file), name=self._target_name).loc[
                        bin_int_var_names
                    ]
                ),
                decimals=self._decimals,
            )

        return bin_int_var_names, target

    def _make_feature_matrix(
        self,
        path_to_lp_file: pathlib2.Path,
        bin_and_int_var_names: t.List[str],
    ) -> pd.DataFrame:
        """
        Строит матрицу признакового описания
        для заданной математической постановки
        """
        features: t.List[t.Union[pd.Series, pd.DataFrame]] = []
        lp_file_name = path_to_lp_file.name

        # Строит усредненное релаксированное решение
        if self._relax_methods_flag:
            logger.info(f"Building features 'relax_methods' for '{lp_file_name}' ...")
            _avg_relax_sol: pd.Series = self._find_avg_relax_sol(
                path_to_lp_file=path_to_lp_file,
                bin_and_int_var_names=bin_and_int_var_names,
            )
            features.append(_avg_relax_sol)

            # Строит усредненное бинарное представление на базе усредненного релаксированного решения
            if self._avg_bin_thresholds_flag:
                logger.info(f"Building features 'bin_threshold' for '{lp_file_name}' ...")
                avg_binary_representation: pd.DataFrame = self._make_avg_binary_representaion(
                    relax_sol=_avg_relax_sol,
                    bin_and_int_var_names=bin_and_int_var_names,
                    min_threshold=self._min_thresholds_value,
                    max_threshold=self._max_thresholds_value,
                    n_threshold=self._n_thresholds,
                )
                features.append(avg_binary_representation)

        # Извлекает значения коэффициентов бинарных и целочисленных переменных в целевой функции
        if self._obj_coeffs_flag:
            logger.info(f"Building features 'obj_coeffs' for '{lp_file_name}' ...")
            _obj_coeffs: pd.Series = self._get_obj_coeffs(
                path_to_lp_file=path_to_lp_file,
                bin_and_int_var_names=bin_and_int_var_names,
            )
            features.append(_obj_coeffs)

        # Для каждой переменной набора вычисляет количество положительных
        # и отрицательных коэффициентов, с которыми рассматриваемая переменная
        # входит в сопряженные ограничения
        if self._number_pos_and_neg_coeffs_flag:
            logger.info(f"Building features 'number_pos_and_neg_coeffs' for '{lp_file_name}' ...")
            number_of_conss_included_var: pd.DataFrame = self._get_number_pos_and_neg_coeffs(
                path_to_lp_file=path_to_lp_file,
                bin_and_int_var_names=bin_and_int_var_names,
            )
            features.append(number_of_conss_included_var)

        return pd.concat(features, axis=1)

    def _find_avg_relax_sol(
        self,
        path_to_lp_file: pathlib2.Path,
        bin_and_int_var_names: t.List[str],
    ) -> pd.Series:
        """
        Задача линейного программирования решается в релаксированной постановке с использованием
        различных методов (первичный симплекс-метод, двойственный симплекс-метод, метод внутренней точки и т.д.).
        Затем полученные решения усредняются
        """
        relax_sols: t.List[pd.Series] = []
        relax_method_names = {
            "p": "primal simplex",
            "d": "dual simplex",
            "b": "barrier",
            "c": "barrier with crossover",
        }

        for method in self._relax_methods:
            lp_file_name = path_to_lp_file.name
            method_name = relax_method_names[method]
            logger.info(
                f"Finding relax solution for '{lp_file_name}' ({method_name} method is used) ..."
            )
            model = pyscipopt.Model()
            model.readProblem(path_to_lp_file)
            model.setParam("lp/initalgorithm", f"{method}")

            all_vars: t.List[pyscipopt.scip.Variable] = model.getVars()
            bin_and_int_vars: t.List[pyscipopt.scip.Variable] = [
                var for var in all_vars if (var.vtype() == BINARY) or (var.vtype() == INTEGER)
            ]
            for var in bin_and_int_vars:
                model.chgVarType(var, CONTINUOUS)

            model.optimize()
            status = model.getStatus()
            if status == STATUS_OPTIMAL_SOL:
                logger.info(
                    f"Optimal relax solution found. Objective: {model.getObjVal():.5g}, "
                    f"gap: {model.getGap() * 100:.4g}%"
                )

                relax_sol: pyscipopt.scip.Solution = model.getBestSol()

                if not self._path_to_output_dir.exists():
                    self._path_to_output_dir.mkdir()

                method_name_with_underscores = method_name.replace(" ", "_")
                model.writeSol(
                    relax_sol,
                    self._path_to_output_dir.joinpath(
                        pathlib2.Path(
                            f"{lp_file_name.split('.')[0]}_{method_name_with_underscores}_relax.sol"
                        )
                    ),
                    write_zeros=True,
                )

                relax_sol: pd.Series = np.round(
                    pd.Series(SCIP_sol_to_dict(model, relax_sol, all_vars)),
                    decimals=self._decimals,
                )
                relax_sols.append(relax_sol)
            elif status == STATUS_INFEASIBLE:
                # Bug: SCIP 8.0.0 (PySCIPOpt 4.2.0, PySCIPOpt 4.0.0) считает,
                # что задача в релаксированной постановке приводит к недопустимому решению!
                # SCIP 7.0.3 (PySCIPOpt 3.4.0) как на ОС Windows 10, так и Unix-подобных операционных
                # системах работает корректно
                logger.info(
                    "Oops. Infeasible solution. NB!: Check PySCIPOpt version. "
                    "Probably this is a bug of PySCIPOpt 4.[0,2].0 ..."
                )
                continue
            else:
                logger.info("Unknown status ...")
                sys.exit(-1)

        avg_relax_sol = pd.concat(relax_sols, axis=1).mean(axis=1)
        avg_relax_sol.name = "relax_sol"

        return avg_relax_sol.loc[bin_and_int_var_names]

    def _make_avg_binary_representaion(
        self,
        relax_sol: pd.Series,
        bin_and_int_var_names: t.List[str],
        min_threshold: float = 0.0,
        max_threshold: float = 1.0,
        n_threshold: int = 10,
    ) -> pd.DataFrame:
        """
        Строит набор бинарных последовательностей на базе усредненного релаксированного решения.
        Переменные, значения которых не превышают заданный порог `threshold`,
        выставляются в ноль, в противном случае - в единицу.
        Затем набор бинарных представлений усредняется
        """
        avg_binary_repr: pd.Series = pd.concat(
            [
                pd.Series(
                    np.where(relax_sol < threshold, 0, 1),
                    index=relax_sol.index,
                    name=f"threshold_bin_{threshold:.3g}",
                )
                for threshold in np.linspace(min_threshold, max_threshold, n_threshold)
            ],
            axis=1,
        ).mean(axis=1)
        avg_binary_repr.name = "avg_binary_representation"

        return avg_binary_repr.loc[bin_and_int_var_names]

    def _get_obj_coeffs(
        self,
        path_to_lp_file: pathlib2.Path,
        bin_and_int_var_names: t.List[str],
        index_name: str = "var_name",
        feature_name: str = "obj_coeff",
    ) -> pd.Series:
        """
        Извлекает значения коэффициентов бинарных и целочисленных переменных
        в целевой функции заданной математической постановки
        """
        model = pyscipopt.Model()
        model.readProblem(path_to_lp_file)
        all_vars = model.getVars()

        return (
            pd.DataFrame.from_records(
                [(var.name, var.getObj()) for var in all_vars],
                columns=[index_name, feature_name],
            )
            .set_index(index_name)
            .squeeze()
            .loc[bin_and_int_var_names]
        )

    def _get_number_pos_and_neg_coeffs(
        self,
        path_to_lp_file: pathlib2.Path,
        bin_and_int_var_names: t.List[str],
        var_name: str = "var_name",
        value: str = "value",
        n_pos_coeffs_in_cons: str = "n_pos_coeffs_in_cons",
        n_neg_coeffs_in_cons: str = "n_neg_coeffs_in_cons",
    ) -> pd.DataFrame:
        """
        Для каждой переменной набора вычисляет количество положительных
        и отрицательных коэффициентов, с которыми рассматриваемая переменная
        входит в сопряженные ограничения
        NB!: Некоторые переменные могут входить только в целевую функцию и поэтому
        число переменных, ассоциированных с количеством сопряженных ограничений,
        может не совпадать с числом переменных в исходной задаче
        """
        model = pyscipopt.Model()
        model.readProblem(path_to_lp_file)

        all_conss: t.List[pyscipopt.scip.Constraint] = model.getConss()
        all_conss_dict_format: t.List[dict] = [model.getValsLinear(cons) for cons in all_conss]

        var_name_and_value_pairs: t.List[t.Tuple[str, float]] = []
        for cons in all_conss_dict_format:
            var_name_and_value_pairs.extend(list(cons.items()))

        var_names_vals = pd.DataFrame.from_records(
            var_name_and_value_pairs, columns=[var_name, value]
        )
        var_names_vals[n_pos_coeffs_in_cons]: pd.Series = (var_names_vals[value] > 0).astype(
            np.int_
        )
        var_names_vals[n_neg_coeffs_in_cons]: pd.Series = (var_names_vals[value] < 0).astype(
            np.int_
        )
        number_pos_and_neg_coeffs: pd.DataFrame = (
            var_names_vals.groupby(var_name)
            .agg(np.sum)
            .loc[:, [n_pos_coeffs_in_cons, n_neg_coeffs_in_cons]]
        )
        return number_pos_and_neg_coeffs.loc[bin_and_int_var_names]
