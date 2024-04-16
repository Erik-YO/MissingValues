
from math import floor
import pandas as pd
from typing import Any, Callable, Union
from random import random, seed
import numpy as np

from time import time

seed(time())


def rand_between(left: float, right: float) -> float:
    if left > right:
        left, right = right, left
    return random() * (right - left) + left


class EliminadorColeccion:

    def __init__(
            self, atributo: str, atributoCondicionanteMar: str,
            marInverso: bool = False, marProgresivo: float = 0) -> None:
        '''
        Elimina inplace cuando es posible.

        Los metodos mcar, mar y mnar devuelven el
        mismo dataset pasado si es posible.

        Para ver los usos de marInverso y marProgresivo ver el metodo mar,
        mnar usa la misma configuracion.

        Se prefieren valores enteros para marProgresivo.
        '''

        if marProgresivo < 0:
            raise ValueError(
                f'{self.__class__.__name__}() marProgresivo debe ser '
                'mayor o igual a 0.')

        self.atributo = atributo
        self.atributoCondicionante = atributoCondicionanteMar

        self.marInverso = bool(marInverso)
        self.marProgresivo = marProgresivo

    # Comprobaciones de argumentos

    def checkArgumentos(
            self, func: Callable, dataset: pd.DataFrame, proporcion: float):

        if proporcion < 0 or proporcion > 1:
            raise ValueError(
                f'{self.__class__.__name__}.{func.__name__}() proporcion'
                f' a eliminar debe estar en [0, 1], no {proporcion}')

        if self.atributo not in dataset:
            raise ValueError(
                f'{self.__class__.__name__}.{func.__name__}() el atributo'
                f'"{self.atributo}" no existe en el dataset proporcionado')

    def checkProporcion(
            self, func: Callable, dataset: pd.DataFrame, proporcion: float
            ) -> Union[int, pd.DataFrame]:

        if proporcion < 0 or proporcion > 1:
            raise ValueError(
                f'{self.__class__.__name__}.{func.__name__}() proporcion'
                f'debe estar en [0, 1], no {proporcion}')

        if proporcion == 0:
            return dataset
        if proporcion == 1:
            dataset[self.atributo] = dataset[self.atributo].mask(
                [True for _ in range(dataset.shape[0])], pd.NA)
            return dataset

        rows, *_ = dataset.shape
        n = round(rows * proporcion)

        return n

    # Metodos de eliminacion principales

    def mcar(self, dataset: pd.DataFrame, proporcion: float) -> pd.DataFrame:
        '''
        Eliminacion completamente aleatoria
        '''
        self.checkArgumentos(self.mcar, dataset, proporcion)
        n = self.checkProporcion(self.mcar, dataset, proporcion)
        if isinstance(n, pd.DataFrame):
            return n

        # Obtener filas a eliminar
        eliminables = pd.Series(dataset.index < n).sample(frac=1).reset_index(drop=True)

        # Poner Nan en esas filas
        dataset[self.atributo] = dataset[self.atributo].mask(
            eliminables, pd.NA)

        return dataset

    def mar(self, dataset: pd.DataFrame, proporcion: float) -> pd.DataFrame:
        '''
        Con atributo igual a atributoCondicionante, es equivalente a MNAR.

        Con marInverso en True, las entradas de atributoCondicionante con
        valores menores tienen mas probabilidades de generar ausencias.

        Con marProgresivo igual a 0 se genera una division abrupta entre las
        filas con valores por encima (o debajo dependiendo de marInverso)
        cuyo atributo queda en Nan.
        Con marProgresivo mayor que 0, la division no es abrupta, sino que
        genera un patron de desvanecimiento progresivo. Cuanto mayor sea el
        valor de marProgresivo, más suave es el desvanecimiento y hay más
        probabilidades de que filas más alejadas generen ausencias y filas
        más próximas mantengan su valor.
        '''
        self.checkArgumentos(self.mar, dataset, proporcion)
        n = self.checkProporcion(self.mar, dataset, proporcion)
        if isinstance(n, pd.DataFrame):
            return n

        if self.atributoCondicionante not in dataset.columns:
            raise ValueError(
                f'{self.__class__.__name__}.{self.mar.__name__}() el atributo'
                f' condicionante "{self.atributo}" no existe en el dataset '
                'proporcionado')

        if self.marProgresivo > 0:
            return self.mar_progresivo(dataset, proporcion)

        return self.mar_abrupto(dataset, proporcion)

    def mnar(self, dataset: pd.DataFrame, proporcion: float) -> pd.DataFrame:
        '''
        Los registros eliminados dependen del valor de los propios registros.
        Usa la misma configuracion que mar (marInverso y marProgresivo)
        '''
        self.checkArgumentos(self.mnar, dataset, proporcion)
        n = self.checkProporcion(self.mnar, dataset, proporcion)
        if isinstance(n, pd.DataFrame):
            return n

        oldCond = self.atributoCondicionante
        self.atributoCondicionante = self.atributo
        try:
            dataset = self.mar(dataset, proporcion)
        finally:
            self.atributoCondicionante = oldCond

        return dataset

    # Submetodos de eliminacion

    def mar_abrupto(
            self, dataset: pd.DataFrame, proporcion: float) -> pd.DataFrame:
        n = self.checkProporcion(self.mar_abrupto, dataset, proporcion)
        if isinstance(n, pd.DataFrame):
            return n

        condicionante = dataset[self.atributoCondicionante]
        valor_limite, *_ = condicionante.sort_values(
            ascending=self.marInverso)[n: n+1]

        eliminables = np.zeros(len(condicionante), dtype=bool)
        eliminados = 0
        if self.marInverso:
            for i, c in enumerate(condicionante):
                if eliminados >= n:
                    break
                if c <= valor_limite:
                    eliminados += 1
                    eliminables[i] = True
                # eliminables[i] = condicionante < valor_limite
        else:
            for i, c in enumerate(condicionante):
                if eliminados >= n:
                    break
                if c >= valor_limite:
                    eliminados += 1
                    eliminables[i] = True
            # eliminables = condicionante > valor_limite

        # Poner Nan en esas filas
        dataset[self.atributo] = dataset[self.atributo].mask(
            eliminables, pd.NA)

        return dataset

    def mar_progresivo(
            self, dataset: pd.DataFrame, proporcion: float) -> pd.DataFrame:
        n = self.checkProporcion(self.mar_progresivo, dataset, proporcion)
        if isinstance(n, pd.DataFrame):
            return n

        condicionante = dataset[self.atributoCondicionante].copy()
        minimo: float = condicionante.min()
        if minimo <= 0:  # cond > 0
            condicionante -= minimo - 1
            minimo -= minimo - 1

        # Normalizar (todos en [0, 1])
        condicionante /= condicionante.max()
        # Reestablecer la proporcionalidad
        condicionante /= condicionante.min()

        if self.marInverso:  # Revertido antes de hacer exponenciacion
            minimo, maximo = condicionante.min(), condicionante.max()
            condicionante = minimo + maximo - condicionante
        # Nivel de progresivo: aumentar diferencia entre menor y mayor
        condicionante = pd.Series(condicionante ** self.marProgresivo)

        incrementos = condicionante.copy()
        for i in range(1, incrementos.shape[0]):
            incrementos[i] += incrementos[i - 1]

        data = dataset[self.atributo]

        for _ in range(n):
            # Seleccionar siguiente posicion a eliminar

            # El mayor no nan
            newmax = next(x for x in incrementos[::-1] if not pd.isna(x))
            rv = rand_between(incrementos[0], newmax)
            # ridx = np.argmax(incrementos > rv)
            ridx = (incrementos > rv).argmax()

            data[ridx] = pd.NA

            # Siguiente iteracion
            val = condicionante[ridx]
            incrementos[ridx] = pd.NA
            incrementos[ridx:] = incrementos[ridx:] - val

        dataset[self.atributo] = data

        return dataset


# V2

class Eliminador:

    def __init__(self, atributo: str) -> None:
        '''
        Elimina inplace cuando es posible.
        
        Devuelven el mismo dataset pasado si es posible.
        '''
        self.atributo = atributo

    def checkArgumentos(
            self, dataset: pd.DataFrame, proporcion: float):

        if proporcion < 0 or proporcion > 1:
            raise ValueError(
                f'{self}() proporcion'
                f' a eliminar debe estar en [0, 1], no {proporcion}')

        if self.atributo not in dataset:
            raise ValueError(
                f'{self}() el atributo'
                f'"{self.atributo}" no existe en el dataset proporcionado')

    def checkProporcion(
            self, dataset: pd.DataFrame, proporcion: float
            ) -> Union[int, pd.DataFrame]:

        if proporcion < 0 or proporcion > 1:
            raise ValueError(
                f'{self}() proporcion'
                f'debe estar en [0, 1], no {proporcion}')

        if proporcion == 0:
            return dataset
        if proporcion == 1:
            dataset[self.atributo] = dataset[self.atributo].mask(
                [True for _ in range(dataset.shape[0])], pd.NA)
            return dataset

        rows, *_ = dataset.shape
        n = round(rows * proporcion)

        return n

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError('Metodo abstracto')

    def __str__(self) -> str:
        return f'{self.__class__.__name__} {self.atributo}'

    def __repr__(self) -> str:
        return str(self)


class EliminadorMCAR(Eliminador):

    def __call__(
            self, dataset: pd.DataFrame, proporcion: float) -> pd.DataFrame:
        '''
        Eliminacion completamente aleatoria
        '''
        self.checkArgumentos(dataset, proporcion)
        n = self.checkProporcion(dataset, proporcion)
        if isinstance(n, pd.DataFrame):
            return n

        # Obtener filas a eliminar
        eliminables = pd.Series(dataset.index < n).sample(frac=1).reset_index(drop=True)

        # Poner Nan en esas filas
        dataset[self.atributo] = dataset[self.atributo].mask(
            eliminables, pd.NA)

        return dataset

    def __str__(self) -> str:
        return f'MCAR {self.atributo}'


class EliminadorMAR(Eliminador):
    def __init__(
            self, atributo: str, atributoCondicionanteMar: str,
            marInverso: bool = False, marProgresivo: float = 0) -> None:
        '''
        Para ver los usos de marInverso y marProgresivo ver el metodo __call__.

        Se prefieren valores enteros para marProgresivo.
        '''
        super().__init__(atributo)

        if marProgresivo < 0:
            raise ValueError(
                f'{self}() marProgresivo debe ser '
                'mayor o igual a 0.')

        self.atributoCondicionante = atributoCondicionanteMar

        self.marInverso = bool(marInverso)
        self.marProgresivo = marProgresivo

    def __str__(self) -> str:
        inv = ' inverso' if self.marInverso else ''
        prog = ' progresivo' if self.marProgresivo else ''
        return 'MAR {}{}{} {}'.format(
            self.atributo, inv, prog, self.atributoCondicionante)

    def __call__(
            self, dataset: pd.DataFrame, proporcion: float) -> pd.DataFrame:
        '''
        Con atributo igual a atributoCondicionante, es equivalente a MNAR.

        Con marInverso en True, las entradas de atributoCondicionante con
        valores menores tienen mas probabilidades de generar ausencias.

        Con marProgresivo igual a 0 se genera una division abrupta entre las
        filas con valores por encima (o debajo dependiendo de marInverso)
        cuyo atributo queda en Nan.
        Con marProgresivo mayor que 0, la division no es abrupta, sino que
        genera un patron de desvanecimiento progresivo. Cuanto mayor sea el
        valor de marProgresivo, más suave es el desvanecimiento y hay más
        probabilidades de que filas más alejadas generen ausencias y filas
        más próximas mantengan su valor.
        '''
        self.checkArgumentos(dataset, proporcion)
        n = self.checkProporcion(dataset, proporcion)
        if isinstance(n, pd.DataFrame):
            return n

        if self.atributoCondicionante not in dataset.columns:
            raise ValueError(
                f'{self}() el atributo'
                f' condicionante "{self.atributo}" no existe en el dataset '
                'proporcionado')

        if self.marProgresivo > 0:
            return self.mar_progresivo(dataset, proporcion)

        return self.mar_abrupto(dataset, proporcion)

    def mar_abrupto(
            self, dataset: pd.DataFrame, proporcion: float) -> pd.DataFrame:
        n = self.checkProporcion(dataset, proporcion)
        if isinstance(n, pd.DataFrame):
            return n

        idx_ordenados = dataset[self.atributoCondicionante].sort_values(
            ascending=self.marInverso).index

        atr = dataset[self.atributo].copy()
        atr[idx_ordenados[:n]] = pd.NA
        dataset[self.atributo] = atr

        return dataset

    def mar_progresivo(
            self, dataset: pd.DataFrame, proporcion: float) -> pd.DataFrame:
        n = self.checkProporcion(dataset, proporcion)
        if isinstance(n, pd.DataFrame):
            return n

        # potencia=1 es mcar
        potencia = 1 + self.marProgresivo

        sidx = list(dataset[self.atributoCondicionante].sort_values(
            ascending=self.marInverso).index)
        for _ in range(n):
            i = floor(len(sidx) * (random() ** potencia))
            idx = sidx.pop(i)
            dataset[self.atributo][idx] = pd.NA

        return dataset


class EliminadorMNAR(EliminadorMAR):

    def __init__(
            self, atributo: str, inverso: bool = False,
            progresivo: float = 0) -> None:
        super().__init__(atributo, atributo, inverso, progresivo)

    def __str__(self) -> str:
        inv = ' inverso' if self.marInverso else ''
        prog = ' progresivo' if self.marProgresivo else ''
        return 'MNAR {}{}{}'.format(self.atributo, inv, prog)


if __name__ == '__main__':

    from matplotlib import pyplot as plt

    n = 30
    atr, cond = 'V1', 'V2'
    inv = False
    progres = 0

    #

    e = EliminadorMAR(atr, cond, inv, progres)

    arr = np.array([
        (i // n + random() / 2, i % n + random() / 2)
        for i in range(n * n)
    ])
    np.random.shuffle(arr)

    df = pd.DataFrame(arr, columns=(atr, cond))

    new = e(df.copy(), 0.5)
    print(pd.isna(new[atr]).value_counts())

    new[atr + '_prev'] = df[atr]
    new[cond + '_prev'] = df[cond]
    plt.scatter([0, 0, n, n], [0, n, 0, n], c='red')
    plt.scatter(new[atr], new[cond], marker='.')
    plt.show()

"""
MAR Abrupto

        if self.marInverso:
            def compara(valor): valor <= valor_limite
        else:
            def compara(valor): valor >= valor_limite

        for i, valor_condicionante in enumerate(condicionante):
            if eliminados >= n:
                break
            if compara(valor_condicionante):
                eliminados += 1
                eliminables[i] = True


"""

"""
Old MAR Abrupto



        condicionante = dataset[self.atributoCondicionante]
        valor_limite, *_ = condicionante.sort_values(
            ascending=self.marInverso)[n: n+1]

        eliminables = np.zeros(len(condicionante), dtype=bool)
        eliminados = 0
        if self.marInverso:
            for i, c in enumerate(condicionante):
                if eliminados >= n:
                    break
                if c <= valor_limite:
                    eliminados += 1
                    eliminables[i] = True
                # eliminables[i] = condicionante < valor_limite
        else:
            for i, c in enumerate(condicionante):
                if eliminados >= n:
                    break
                if c >= valor_limite:
                    eliminados += 1
                    eliminables[i] = True
            # eliminables = condicionante > valor_limite

        # Poner Nan en esas filas
        dataset[self.atributo] = dataset[self.atributo].mask(
            eliminables, pd.NA)

"""

"""
Old MAR Progresivo



        condicionante = dataset[self.atributoCondicionante].copy()
        minimo: float = condicionante.min()
        if minimo <= 0:  # cond > 0
            condicionante -= minimo - 1
            minimo -= minimo - 1

        # Normalizar (todos en [0, 1])
        condicionante /= condicionante.max()
        # Reestablecer la proporcionalidad
        condicionante /= condicionante.min()

        if self.marInverso:  # Revertido antes de hacer exponenciacion
            minimo, maximo = condicionante.min(), condicionante.max()
            condicionante = minimo + maximo - condicionante
        # Nivel de progresivo: aumentar diferencia entre menor y mayor
        condicionante = pd.Series(condicionante ** self.marProgresivo)

        incrementos = condicionante.copy()
        for i in range(1, incrementos.shape[0]):
            incrementos[i] += incrementos[i - 1]

        data = dataset[self.atributo]

        for _ in range(n):
            # Seleccionar siguiente posicion a eliminar

            # El mayor no nan
            newmax = next(x for x in incrementos[::-1] if not pd.isna(x))
            rv = rand_between(incrementos[0], newmax)
            # ridx = np.argmax(incrementos > rv)
            ridx = (incrementos > rv).argmax()

            data[ridx] = pd.NA

            # Siguiente iteracion
            val = condicionante[ridx]
            incrementos[ridx] = pd.NA
            incrementos[ridx:] = incrementos[ridx:] - val

        dataset[self.atributo] = data

"""
