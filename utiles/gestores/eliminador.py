
import pandas as pd
from typing import Tuple, Callable


class GestorEliminador:
    __name__ = 'Eliminacion'

    def __init__(self, atributo: str) -> None:
        self.atributo = atributo

        self.tecnicas: Tuple[Callable[[pd.DataFrame], pd.DataFrame]] = (
            self.listwise,
            self.pairwise
        )

    def listwise(self, dataset: pd.DataFrame) -> pd.DataFrame:
        '''Elimina las filas en las que haya algun valor ausente'''
        dataset.dropna(inplace=True, axis=0)
        return dataset

    def pairwise(self, dataset: pd.DataFrame) -> pd.DataFrame:
        '''
        Elimina el atributo en el que haya algun valor ausente.
        No es una buena opcion para clasificación, pero puede serlo
        para otro tipo de problemas.
        '''
        dataset.drop(self.atributo, axis=1, inplace=True)
        return dataset
