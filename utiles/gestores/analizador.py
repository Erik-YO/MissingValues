
import pandas as pd
from typing import Tuple, Callable

import impyute
import numpy as np


class GestorAnalizador:
    __name__ = 'Analisis'

    def __init__(self, atributo: str) -> None:
        self.atributo = atributo

        self.tecnicas: Tuple[Callable[[pd.DataFrame], pd.DataFrame]] = (
            self.generativo_maximizacion_esperanza,
        )

    def generativo_maximizacion_esperanza(
            self, dataset: pd.DataFrame) -> pd.DataFrame:
        '''
        [2]: https://machinelearningmastery.com/expectation-maximization-em-algorithm/
        [3]: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-022-04740-9
        '''

        generado: np.ndarray = impyute.em(dataset.to_numpy())

        variable = [c for c in dataset.columns].index(self.atributo)
        dataset[self.atributo] = generado[:, variable]

        return dataset


"""
    def generativo_maxima_verosimilitud(
            self, dataset: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def discriminatorio(
            self, dataset: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()
"""
