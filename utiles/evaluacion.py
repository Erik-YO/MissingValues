
from typing import Union
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np


class Evaluador:

    def __init__(
            self, atributoGenerado: str, proporcionTest: float = 0.25,
            atributoClase: str = 'Class') -> None:

        self.atributoGenerado = atributoGenerado
        self.proporcionTest = proporcionTest
        self.atributoClase = atributoClase

        self.ultimo_evaluado: Union[None, pd.DataFrame] = None

        self.ultima_exactitud: Union[None, float] = None
        self.ultima_sensibilidad: Union[None, float] = None
        self.ultima_precision: Union[None, float] = None

        self.ultimo_modelo = None

    def similitud(
            self, original: pd.DataFrame,
            generado: pd.DataFrame) -> Union[None, float]:
        '''
        Por 1 - distancia euclidea de los atributos.
        Normalizados por min-max:
            norm = (valores - min) / (max - min)
        '''

        if original.shape != generado.shape:
            if len(original.columns) == len(generado.columns):
                return generado.shape[0] / original.shape[0]
            return 0

        if self.atributoGenerado not in original.columns:
            raise ValueError(
                f'{self.__class__.__name__}.{self.similitud.__name__}() '
                f'atributo {self.atributoGenerado} no existe en el dataset '
                'original')
        if self.atributoGenerado not in generado.columns:
            raise ValueError(
                f'{self.__class__.__name__}.{self.similitud.__name__}() '
                f'atributo {self.atributoGenerado} no existe en el dataset '
                'generado')

        generado_data = generado[self.atributoGenerado]
        original_data = original[self.atributoGenerado]

        generado_data_np = generado_data.to_numpy()
        generado_magnitud = np.sqrt(generado_data_np.dot(generado_data_np))
        original_data = original_data.to_numpy()
        original_magnitud = np.sqrt(original_data.dot(original_data))

        gen_norm: pd.Series = generado_data / generado_magnitud
        orig_norm: pd.Series = original_data / original_magnitud

        # Distancia euclidea
        distancia = ((orig_norm - gen_norm) ** 2).sum() ** 0.5
        return 1 - (distancia / 2)

    def evaluacion_por_modelo(
            self, dataset_original: pd.DataFrame,
            dataset_generado: pd.DataFrame):
        '''
        Entrenamos con el dataset generado y probamos con el original

        Exactitud, Sensibilidad, Precision

        - accuracy: (elementos clasificados correctamente) =
            (TP + TN) / total
        - recall: (positivos encontrados entre positivos reales) =
            TP / (TP + FN)
        - precision: (positivos bien predichos entre todos los positivos
            predichos) = TP / (TP + FP)
        '''
        x_gen = dataset_generado[[
            col for col in dataset_generado.columns
            if col != self.atributoClase]]
        y_gen = dataset_generado[self.atributoClase]

        x_ori = dataset_original[[
            col for col in dataset_generado.columns
            if col != self.atributoClase]]
        y_ori = dataset_original[self.atributoClase]

        x_train, _, y_train, _ = train_test_split(
            x_gen, y_gen, stratify=y_gen,
            test_size=self.proporcionTest)

        _, x_test, _, y_test = train_test_split(
            x_ori, y_ori, stratify=y_ori,
            test_size=self.proporcionTest)

        # Randomforest
        clasificador = RandomForestClassifier(n_estimators=3)
        clasificador.fit(x_train, y_train)
        self.ultimo_modelo = clasificador
        prediccion = clasificador.predict(x_test)

        self.ultimo_evaluado = dataset_generado
        self.ultima_exactitud = accuracy_score(y_test, prediccion)
        self.ultima_sensibilidad = recall_score(y_test, prediccion, zero_division=0)
        self.ultima_precision = precision_score(y_test, prediccion, zero_division=0)

        # F1 = (2 * precision * recall) / (precision + recall).
        if not self.ultima_precision and not self.ultima_sensibilidad:
            self.ultima_f1score = 0
        else:
            self.ultima_f1score = (
                (2 * self.ultima_precision * self.ultima_sensibilidad)
                / (self.ultima_precision + self.ultima_sensibilidad))

        return

    def exactitud(
            self, dataset_original: pd.DataFrame,
            dataset_generado: pd.DataFrame) -> float:
        if dataset_generado is not self.ultimo_evaluado:
            self.evaluacion_por_modelo(dataset_original, dataset_generado)

        return self.ultima_exactitud

    def sensibilidad(
            self, dataset_original: pd.DataFrame,
            dataset_generado: pd.DataFrame) -> float:
        if dataset_generado is not self.ultimo_evaluado:
            self.evaluacion_por_modelo(dataset_original, dataset_generado)

        return self.ultima_sensibilidad

    def precision(
            self, dataset_original: pd.DataFrame,
            dataset_generado: pd.DataFrame) -> float:
        if dataset_generado is not self.ultimo_evaluado:
            self.evaluacion_por_modelo(dataset_original, dataset_generado)

        return self.ultima_precision

    def f1_score(
            self, dataset_original: pd.DataFrame,
            dataset_generado: pd.DataFrame) -> float:
        if dataset_generado is not self.ultimo_evaluado:
            self.evaluacion_por_modelo(dataset_original, dataset_generado)

        return self.ultima_f1score

    def reset(self):
        self.ultimo_evaluado = None

        self.ultima_exactitud = None
        self.ultima_sensibilidad = None
        self.ultima_precision = None
        self.ultima_f1score = None

        self.ultimo_modelo = None

    def mostrar_modelo(
            self, dataset_original: pd.DataFrame,
            dataset_generado: pd.DataFrame, nombre: str) -> float:
        import matplotlib.pyplot as plt
        if dataset_generado is not self.ultimo_evaluado:
            self.evaluacion_por_modelo(dataset_original, dataset_generado)

        nombres = list(self.ultimo_modelo.feature_names_in_)
        importancias = list(self.ultimo_modelo.feature_importances_)
        idxv6 = nombres.index('V6') if 'V6' in nombres else -1
        print(
            (f'Importancia {nombres[idxv6]}: {importancias[idxv6]} con media '
            f'de {sum(importancias)/len(importancias)} es el '
            f'{importancias[idxv6]/max(importancias)} del maximo')
            if idxv6 >= 0
            else 'V6 no presente')
        print('_________________________________')
        print()

        plt.bar(self.ultimo_modelo.feature_names_in_, self.ultimo_modelo.feature_importances_)
        plt.title('Importancias ' + nombre)
        #plt.savefig('./resultados/importancias_modelo', format='svg', dpi=400, bbox_inches='tight')
        plt.show()
