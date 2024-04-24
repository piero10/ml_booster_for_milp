# Для того, чтобы библиотека marshmallow-dataclass могла поддерживать тип Union,
# библиотеку требуется устанавливать с необзятальеным полем 'union'
# $> pip install -U "marshmallow-dataclass[union]"
import sys
import typing as t
from dataclasses import dataclass, field

import marshmallow.validate
import pathlib2

from utils.artifacts import NONE
from utils.make_logger import logger


@dataclass
class MainParams:
    problem_name: str = field(
        metadata={"validate": marshmallow.validate.Length(min=1)}, default="MILP"
    )
    strategy: str = field(
        metadata={
            "validate": marshmallow.validate.OneOf(["anomaly_detection", "binary_classification"])
        },
        default="anomaly_detection",
    )
    target_name: str = field(
        metadata={"validate": marshmallow.validate.Length(min=1)}, default="target"
    )
    Xy_train_file_name: str = field(
        metadata={"validate": marshmallow.validate.Regexp(r".*\.csv")},
        default="features.csv",
    )


@dataclass
class Tolerances:
    decimals: int = field(
        metadata={"validate": marshmallow.validate.Range(min=8, max=12)}, default=10
    )


@dataclass
class Paths:
    path_to_lp_files: str
    path_to_sol_files: str
    path_to_set_file: str
    path_to_output_dir: str
    path_to_test_lp_file: str
    path_to_test_sol_file: str

    def __post_init__(self):
        self.check_exists_file_or_dir(self.path_to_lp_files)
        self.check_exists_file_or_dir(self.path_to_sol_files)
        self.check_exists_file_or_dir(self.path_to_set_file)
        self.check_exists_file_or_dir(self.path_to_output_dir)
        self.check_exists_file_or_dir(self.path_to_test_lp_file)
        self.check_exists_file_or_dir(self.path_to_test_sol_file)

    def check_exists_file_or_dir(self, path: str):
        """
        Проверяет переданный путь до файла или директории на сущестование
        """
        if path.upper() == NONE:
            # Переменная конфигурационного файла 'path_to_test_sol_file'
            # получила строковое значение 'None'
            logger.warn(
                "Path to test sol-file was not passed! "
                "Config's key path_to_test_sol_file=None ..."
            )
        else:
            try:
                if not pathlib2.Path(path).exists():
                    raise FileNotFoundError(f"Error! File {path} not found ...")
            except FileNotFoundError as err:
                logger.error(f"{err}")
                sys.exit(-1)


@dataclass
class RelaxMethods:
    use: bool = field(default=True)
    method_names: t.List[str] = field(default_factory=["p"])


@dataclass
class AvgBinThresholds:
    use: bool = field(default=True)
    min_threshold_value: float = field(
        metadata={"validate": marshmallow.validate.Range(min=0.01, max=0.25)},
        default=0.05,
    )
    max_threshold_value: float = field(
        metadata={"validate": marshmallow.validate.Range(min=0.5, max=1.0)}, default=0.95
    )
    n_thresholds: int = field(
        metadata={"validate": marshmallow.validate.Range(min=5, max=20)}, default=8
    )


@dataclass
class ObjCoeffs:
    use: bool = field(default=True)


@dataclass
class NumberPosAndNegCoeffs:
    use: bool = field(default=True)


@dataclass
class TSNE:
    use: bool = field(default=True)
    n_components: int = field(
        metadata={"validate": marshmallow.validate.Range(min=2, max=3)}, default=3
    )
    init: str = field(
        metadata={"validate": marshmallow.validate.OneOf(["random", "pca"])},
        default="random",
    )
    perplexity: float = field(
        metadata={"validate": marshmallow.validate.Range(min=5, max=100)}, default=15
    )
    learning_rate: t.Union[str, float] = field(default="auto")
    n_iter: int = field(
        metadata={"validate": marshmallow.validate.Range(min=250, max=1000)}, default=250
    )
    test_size: float = field(
        metadata={"validate": marshmallow.validate.Range(min=0.01, max=0.05)},
        default=0.025,
    )
    title: str = field(default="t-SNE представление")
    figsize: t.Tuple[int, int] = field(default=(8, 8))
    dpi: int = field(default=350)
    path_to_figure_file: str = field(default="output/tsne.pdf")


@dataclass
class Manifold:
    tsne: TSNE


@dataclass
class Features:
    relax_methods: RelaxMethods
    avg_bin_thresholds: AvgBinThresholds
    obj_coeffs: ObjCoeffs
    number_pos_and_neg_coeffs: NumberPosAndNegCoeffs
    action: str = field(
        metadata={
            "validate": marshmallow.validate.OneOf(
                ["compute_train_test_features", "read_train_and_compute_test_features"]
            )
        },
        default="compute_train_test_features",
    )


@dataclass
class SUOD:
    use: bool = field(default=True)
    combination: str = field(
        metadata={"validate": marshmallow.validate.OneOf(["average", "maximization"])},
        default="average",
    )
    contamination: float = field(
        metadata={"validate": marshmallow.validate.Range(min=0.015, max=0.5)},
        default=0.1,
    )
    n_jobs: t.Optional[int] = field(default=-1)
    verbose: bool = field(default=True)


@dataclass
class COPOD:
    use: bool = field(default=True)
    contamination: float = field(
        metadata={"validate": marshmallow.validate.Range(min=0.015, max=0.5)},
        default=0.10,
    )
    n_jobs: t.Optional[int] = field(default=-1)


@dataclass
class ECOD(COPOD):
    """
    Атрибуты класса ECOD такие же как и у класса COPOD
    """


@dataclass
class IForest:
    use: bool = field(default=True)
    n_estimators: int = field(
        metadata={"validate": marshmallow.validate.Range(min=50, max=1000)}, default=250
    )
    contamination: float = field(
        metadata={"validate": marshmallow.validate.Range(min=0.015, max=0.5)},
        default=0.10,
    )
    n_jobs: t.Optional[int] = field(default=-1)


@dataclass
class HBOS:
    use: bool = field(default=True)
    n_bins: int = field(
        metadata={"validate": marshmallow.validate.Range(min=5, max=25)},
        default=15,
    )
    alpha: float = field(
        metadata={"validate": marshmallow.validate.Range(min=0.0, max=1.0)}, default=0.05
    )
    contamination: float = field(
        metadata={"validate": marshmallow.validate.Range(min=0.015, max=0.5)},
        default=0.10,
    )


@dataclass
class DetectorConfig:
    SUOD: SUOD
    COPOD: COPOD
    ECOD: ECOD
    IForest: IForest
    HBOS: HBOS


@dataclass
class ProjectSchema:
    """
    Главный класс-валидатор схемы конфигурационного файла
    """

    main_params: MainParams
    tolerances: Tolerances
    paths: Paths
    features: Features
    manifold: Manifold
    detector_config: DetectorConfig
