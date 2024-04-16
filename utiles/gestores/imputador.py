
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from typing import Tuple, Callable
from .interpolador import GestorInterpolador


class GestorImputador:
    __name__ = 'Imputacion'

    def __init__(
            self, atributo: str,
            cp_rescale: bool = False,
            grado_polinomio: int = 2,
            grado_spline: int = 3) -> None:

        self.atributo = atributo
        self.interpolador = GestorInterpolador(
            atributo,
            cp_rescale,
            grado_polinomio,
            grado_spline)

        self.tecnicas: Tuple[Callable[[pd.DataFrame], pd.DataFrame]] = (
            self.constante_valor_0,
            self.constante_central_media,
            self.constante_central_mediana,
            self.constante_central_moda,
            *self.interpolador.tecnicas
        )

    def constante_valor_0(self, dataset: pd.DataFrame) -> pd.DataFrame:
        nans = dataset[self.atributo].isna()
        dataset[self.atributo] = dataset[self.atributo].mask(nans, 0)
        return dataset

    def constante_central_media(self, dataset: pd.DataFrame) -> pd.DataFrame:
        media = dataset[self.atributo].mean()
        nans = dataset[self.atributo].isna()
        dataset[self.atributo] = dataset[self.atributo].mask(nans, media)
        return dataset

    def constante_central_mediana(self, dataset: pd.DataFrame) -> pd.DataFrame:
        mediana = dataset[self.atributo].median()
        nans = dataset[self.atributo].isna()
        dataset[self.atributo] = dataset[self.atributo].mask(nans, mediana)
        return dataset

    def constante_central_moda(self, dataset: pd.DataFrame) -> pd.DataFrame:

        ''' # Para variables discretas
        moda, *_ = dataset[self.atributo].mode()
        '''

        # Para variables continuas
        atr = dataset[self.atributo]
        nans = atr.isna()
        xs = atr[~nans]

        # Calculo de la función de densidad
        density = gaussian_kde(xs)
        moda = xs.iloc[np.argmax(density(xs))]

        dataset[self.atributo] = atr.mask(nans, moda)
        return dataset
