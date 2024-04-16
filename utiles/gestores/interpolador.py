import pandas as pd
from typing import Tuple, Callable

from scipy import interpolate
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer


class GestorInterpolador:

    def __init__(
            self, atributo: str,
            cp_rescale: bool = False,
            grado_polinomio: int = 2,
            grado_spline: int = 3) -> None:
        '''
        - atributo es el nombre del atributo ausente que interpolar
        - cp_rescale es para constante por partes
        - grado se refiere al grado del polinomio
        ```interpolacion_polinomios``` y al grado del Spline
        ```interpolacion_spline```
        '''

        self.atributo = atributo
        self.cp_rescale = cp_rescale
        self.grado_polinomio = grado_polinomio
        self.grado_spline = grado_spline

        self.tecnicas: Tuple[Callable[[pd.DataFrame], pd.DataFrame]] = (
            self.interpolacion_constante_partes,
            self.interpolacion_lineal,
            self.interpolacion_polinomios,
            self.interpolacion_spline,
        )

    def interpolacion_constante_partes(
            self, dataset: pd.DataFrame) -> pd.DataFrame:
        '''
        Implementado obteniendo el vecino mas proximo
        ```Nearest Neighbour``` N Dimensiones (con k=1)
        '''

        presente = ~dataset[self.atributo].isna()
        x = dataset[
            [c for c in dataset.columns if c != self.atributo]]
        y = dataset[self.atributo]

        xini, yini = x[presente].to_numpy(), y[presente].to_numpy()

        interpolator = interpolate.NearestNDInterpolator(
            xini, yini, rescale=self.cp_rescale)

        data = interpolator(*x.to_numpy().T)

        dataset[self.atributo] = dataset[self.atributo].where(presente, data)

        return dataset

    def interpolacion_lineal(self, dataset: pd.DataFrame) -> pd.DataFrame:
        '''
        1 Dimension: obtenemos la columna con mas correlacion respecto de la
        que rellenar y la usamos para obtener la interpolacion lineal.
        No llena todos necesariamente, especialmente no los que están
        en los extremos (```outside of the convex hull of the input points```)
        '''

        presente = ~dataset[self.atributo].isna()
        x = dataset[
            [c for c in dataset.columns if c != self.atributo]]
        y = dataset[self.atributo]

        xini, yini = x[presente].to_numpy(), y[presente].to_numpy()

        interpolator = LinearRegression()

        interpolator.fit(xini, yini)
        data = interpolator.predict(x.to_numpy())
        dataset[self.atributo] = dataset[self.atributo].where(presente, data)

        return dataset

    def interpolacion_polinomios(self, dataset: pd.DataFrame) -> pd.DataFrame:

        if self.grado_polinomio < 2:
            raise ValueError('El grado del polinomio debe ser al menos 2')

        presente = ~dataset[self.atributo].isna()
        completo = dataset[presente]

        modelo = make_pipeline(
            PolynomialFeatures(self.grado_polinomio),
            Ridge(alpha=1e-3))
        modelo.fit(
            completo[[c for c in dataset.columns if c != self.atributo]],
            completo[self.atributo])

        data = modelo.predict(dataset[[
            c for c in dataset.columns if c != self.atributo]])

        dataset[self.atributo] = dataset[self.atributo].where(presente, data)

        return dataset

    def interpolacion_spline(self, dataset: pd.DataFrame) -> pd.DataFrame:

        if self.grado_spline < 2:
            raise ValueError('El grado del Spline debe ser al menos 2')

        presente = ~dataset[self.atributo].isna()
        completo = dataset[presente]

        modelo = make_pipeline(
            SplineTransformer(degree=self.grado_spline),
            Ridge(alpha=1e-3))
        modelo.fit(
            completo[[c for c in dataset.columns if c != self.atributo]],
            completo[self.atributo])

        data = modelo.predict(dataset[[
            c for c in dataset.columns if c != self.atributo]])

        dataset[self.atributo] = dataset[self.atributo].where(presente, data)

        return dataset


'''
        import matplotlib.pyplot as plt
        plt.scatter(dataset['V2'], data, dataset['otro'], c='red')
        plt.scatter(
            dataset['V2'], dataset[self.atributo],
            dataset['otro'], c='blue', marker='.')
        plt.show()
'''

'''
def _atributo_mas_correlacionado(self, dataset: pd.DataFrame) -> str:

    maxCol, maxCorr = None, -float('inf')
    presente = ~dataset[self.atributo].isna()
    for col in dataset.columns:
        if col == self.atributo:
            continue
        corr = dataset[self.atributo][presente].corr(
            dataset[col][presente])
        corr = abs(corr)
        if corr > maxCorr:
            maxCol, maxCorr = col, corr

    if maxCol is None:
        raise ValueError(
            'Para interpolacion lineal se requiere al menos un atributo '
            'mas que aquel a interpolar')
    return maxCol
'''
