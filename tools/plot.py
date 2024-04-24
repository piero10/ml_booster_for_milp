import sys
import typing as t

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib2
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

from tools.auxiliary_functions import TSNEComponentsNumberError
from utils.make_logger import logger


def TSNE_plot(
    X: pd.DataFrame,
    y: pd.Series,
    n_components: int,
    init: str,
    perplexity: float,
    learning_rate: t.Union[str, float],
    n_iter: int,
    test_size: float,
    title: str,
    figsize: t.Tuple[int, int],
    dpi: int,
    path_to_figure_file: pathlib2.Path,
    face_color: str = "#F6F6F6",
    xlabel: str = "Первая компонента",
    ylabel: str = "Вторая компонента",
    legend_zeros_vars: str = "Нулевые переменные",
    legend_non_zeros_vars: str = "Ненулевые переменные",
) -> t.NoReturn:
    """
    Строит t-SNE представление набора данных
    """
    try:
        if n_components != 2:
            raise TSNEComponentsNumberError(
                "Error! T-SNE parameter `n_components` must be equal 2 ..."
            )
    except TSNEComponentsNumberError as err:
        logger.error(f"{err}")
        sys.exit(-1)

    y_zeros_and_nonzeros = (y != 0.0).astype(np.int_)
    _, X_exploration, _, y_exploration = train_test_split(
        X,
        y_zeros_and_nonzeros,
        stratify=y_zeros_and_nonzeros,
        test_size=test_size,
    )

    tsne = TSNE(
        n_components=n_components,
        init=init,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
    )
    X_transform = tsne.fit_transform(X_exploration)
    zero_bin_and_ints: pd.Series[np.bool_] = y_exploration == 0
    non_zero_bin_and_ints: pd.Series[np.bool_] = y_exploration == 1

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        X_transform[zero_bin_and_ints, 0],
        X_transform[zero_bin_and_ints, 1],
        label=legend_zeros_vars,
    )
    ax.scatter(
        X_transform[non_zero_bin_and_ints, 0],
        X_transform[non_zero_bin_and_ints, 1],
        label=legend_non_zeros_vars,
    )

    ax.set_facecolor(face_color)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.tight_layout()
    try:
        fig.savefig(path_to_figure_file, dpi=dpi)
    except FileNotFoundError as err:
        logger.error(f"{err}")
    else:
        logger.info(f"File '{path_to_figure_file}' was saved successfully ...")
