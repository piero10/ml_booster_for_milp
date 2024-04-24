# Для того чтобы библиотека PyOD могла поддерживать обертку SUOD,
# требуется установить дополнительную библиотеку suod
# $> pip install suod
import sys
import typing as t

from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.suod import SUOD

from tools.auxiliary_functions import NumberDetectorsError, SingleDetectorError
from utils.make_logger import logger
from utils.schema import DetectorConfig


class DetectorBuilder:
    """
    Строит ансамбль детекторов аномалий
    или одиночный детектор аномалий
    """

    def __init__(self, detector_config: DetectorConfig):
        detector_name_to_detector_schema: dict = detector_config.__dict__
        detector_name_to_obj_pyod_detector: dict = {
            "SUOD": SUOD,
            "COPOD": COPOD,
            "ECOD": ECOD,
            "HBOS": HBOS,
            "IForest": IForest,
        }
        # Общее число детекторов, объявленных в конфигурационном файле
        number_detectors = len(detector_name_to_detector_schema.keys())
        # Число отключенных детекторов
        number_unused_detectors: int = 0
        for detector_schema in detector_name_to_detector_schema.values():
            if not detector_schema.use:
                number_unused_detectors += 1

        try:
            if number_unused_detectors == number_detectors:
                raise NumberDetectorsError("Error! At least one detector must be selected ...")
        except NumberDetectorsError as err:
            logger.error(f"{err}")
            sys.exit(-1)

        ensemble_detectors: t.List = []

        SUOD_name: str = detector_config.SUOD.__class__.__name__
        SUOD_args = detector_name_to_detector_schema[SUOD_name].__dict__

        if detector_config.SUOD.use:
            # Построить ансамбль детекторов
            _ = SUOD_args.pop("use")
            _ = detector_name_to_detector_schema.pop(SUOD_name)

            for (
                detector_name,
                detector_schema,
            ) in detector_name_to_detector_schema.items():
                detector_args = detector_schema.__dict__
                if detector_args["use"]:
                    _ = detector_args.pop("use")
                    ensemble_detectors.append(
                        detector_name_to_obj_pyod_detector[detector_name](**detector_args)
                    )

            logger.info("SUOD detector is built ...")
            self.detector = detector_name_to_obj_pyod_detector[SUOD_name](
                base_estimators=ensemble_detectors, **SUOD_args
            )
        else:
            # Построить одиночный детектор
            number_used_detectors: int = 0
            _ = detector_name_to_detector_schema.pop(SUOD_name)

            for (
                detector_name,
                detector_schema,
            ) in detector_name_to_detector_schema.items():
                detector_args = detector_schema.__dict__
                if detector_args["use"]:
                    _ = detector_args.pop("use")
                    number_used_detectors += 1
                    try:
                        if number_used_detectors > 1:
                            raise SingleDetectorError(
                                f"Error! Too many detectors for a single detector ..."
                            )
                    except SingleDetectorError as err:
                        logger.error(f"{err}")
                        sys.exit(-1)
                    else:
                        logger.info(f"{detector_name} detector is built ...")
                        self.detector = detector_name_to_obj_pyod_detector[detector_name](
                            **detector_args
                        )
