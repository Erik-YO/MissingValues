
import pandas as pd
from typing import Tuple, Callable

from .gestores.eliminador import GestorEliminador
from .gestores.imputador import GestorImputador
from .gestores.analizador import GestorAnalizador


class Gestor:

    def __init__(
            self, atributo: str,
            cp_rescale: bool = False,
            grado_polinomio: int = 2,
            grado_spline: int = 3) -> None:

        self.atributo = atributo

        self.eliminador = GestorEliminador(atributo)
        self.imputador = GestorImputador(
            atributo,
            cp_rescale,
            grado_polinomio,
            grado_spline)
        self.analizador = GestorAnalizador(atributo)

        tecnicas = (
            (g.__name__, t)
            for g in (self.eliminador, self.imputador, self.analizador)
            for t in g.tecnicas
        )
        for g, t in tecnicas:
            t.__func__.__name__ = f'{g}.{t.__func__.__name__}'

        self.tecnicas: Tuple[Callable[[pd.DataFrame], pd.DataFrame]] = tuple(
            self.atributeCheck(t) for t in (
                *self.eliminador.tecnicas,
                *self.imputador.tecnicas,
                *self.analizador.tecnicas,
            ))

    def atributeCheck(self, func: Callable[[pd.DataFrame], pd.DataFrame]):

        def wrapper(dataset: pd.DataFrame, *args, **kwargs):
            if self.atributo not in dataset.columns:
                raise ValueError(
                    f'Validacion: atributo {self.atributo} no '
                    'existe en el dataset')

            return func(dataset, *args, **kwargs)

        wrapper.__name__ = func.__name__
        return wrapper
