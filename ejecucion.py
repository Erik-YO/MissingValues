

#
# Ejecucion
#

#
# Imports
#
import pandas as pd
from typing import Tuple
from time import perf_counter
from utiles.gestion import Gestor
from utiles.evaluacion import Evaluador
from utiles.eliminacion import EliminadorMCAR, EliminadorMAR, EliminadorMNAR

#
# Config
#

DEBUG = True

fichero_dataset = './eeg-eye-state.csv'

variable_ausente = 'V6'

variables_condicionantes = ('V3', 'Class')
variables_excluidas = ('V1', 'V2', 'V7', 'V8', 'V9', 'V11', 'V13', 'V14', )


numero_iteraciones = 5

proporcionTest: float = 0.3
proporciones_ausencia: Tuple[float] = tuple(
    i / 100
    for i in range(5, 100, 5))

interpolacion_cte_partes_rescale = False
interpolacion_grado_polinomio = 3
interpolacion_grado_spline = 4

#
# Calculos
#

dataset_original = pd.read_csv(fichero_dataset)

resultados = pd.DataFrame(columns=[
    'ausencia', 'reducido', 'proporcion', 'tecnica', 'similitud',
    'exactitud', 'sensibilidad', 'precision', 'f1 score', 'tiempo',])


datasets = (
    # Completo
    dataset_original,
    # Reducido
    dataset_original[[
        label for label in dataset_original.columns
        if label not in variables_excluidas]],
)

ausencias = (
    # MCAR
    EliminadorMCAR(variable_ausente),
    # MAR condicionada
    *(EliminadorMAR(variable_ausente, atr) for atr in variables_condicionantes),
    # EliminadorMAR(variable_ausente, 'Class'),
    # MNAR
    EliminadorMNAR(variable_ausente),
)

evaluador = Evaluador(variable_ausente, proporcionTest)
tecnicas_gestion = Gestor(
    variable_ausente, interpolacion_cte_partes_rescale,
    interpolacion_grado_polinomio, interpolacion_grado_spline).tecnicas

maxRows = (
    (len(ausencias) * len(proporciones_ausencia) * len(tecnicas_gestion) + 1) *
    len(datasets) * numero_iteraciones)

rowIdx = 0
iteracionIdx = 0


def log(*args, t: list = []):
    if not t:
        t.append(perf_counter())
        t.append(0)

    s = ' '.join((
        f'[{iteracionIdx + 1}/{numero_iteraciones}] '
        f'[{round(100 * rowIdx / maxRows, 2)}%] '
        f'[{round(perf_counter() - t[0], 1)}s]',
        *(f'{a}' for a in args))).ljust(t[1])

    print(s, end='\n' if DEBUG else '\r')
    t[1] = max(t[1], len(s))


def iteracion(dataset: pd.DataFrame):
    global rowIdx, resultados
    reducido = len(dataset.columns) != len(dataset_original.columns)

    for ausencia in ausencias:

        for proporcion in proporciones_ausencia:
            log(str(ausencia), proporcion, '[eliminando]')

            data_ausente = ausencia(dataset.copy(), proporcion)

            for tecnica in tecnicas_gestion:
                nombre_tecnica = tecnica.__name__.replace('_', ' ').title().replace(' ', '')
                log(str(ausencia), proporcion, tecnica.__name__, '[generando]')

                tiempo = perf_counter()
                try:
                    data_generada = tecnica(data_ausente.copy())
                except NotImplementedError:
                    continue
                tiempo = perf_counter() - tiempo

                log(str(ausencia), proporcion, nombre_tecnica,
                    '[evaluando...similitud]')

                similitud = evaluador.similitud(dataset, data_generada)

                log(str(ausencia), proporcion, tecnica.__name__,
                    '[evaluando...exactitud, sensibilidad, '
                    'precision, f1_score]')

                exactitud = evaluador.exactitud(dataset, data_generada)
                sensibilidad = evaluador.sensibilidad(dataset, data_generada)
                precision = evaluador.precision(dataset, data_generada)
                f1_score = evaluador.f1_score(dataset, data_generada)

                resultados.loc[rowIdx] = [
                    str(ausencia), reducido, proporcion, nombre_tecnica,
                    similitud, exactitud, sensibilidad, precision, f1_score,
                    tiempo]
                rowIdx += 1

                if DEBUG and ('MAR' in str(ausencia)) and proporcion == 0.6 or proporcion == 0.8:
                    print()
                    print(resultados.iloc[-1:])
                    evaluador.mostrar_modelo(
                        dataset, data_generada,
                        f'{str(ausencia)} {proporcion} {"reducido" if reducido else "completo"} {nombre_tecnica}')

                evaluador.reset()

                del data_generada

            del data_ausente

    # Datos originales (referencia)
    exactitud, sensibilidad, precision, f1_score = (
        evaluador.exactitud(dataset, dataset),
        evaluador.sensibilidad(dataset, dataset),
        evaluador.precision(dataset, dataset),
        evaluador.f1_score(dataset, dataset))
    resultados.loc[rowIdx] = [
        'original', reducido,
        0, 'Ninguna', 1, exactitud, sensibilidad, precision, f1_score, pd.NA
    ]
    rowIdx += 1
    evaluador.reset()
    log('Iteracion finalizada')


#
# Ejecucion
#

try:
    for iteracionIdx in range(numero_iteraciones):
        for dataset in datasets:
            iteracion(dataset)
except KeyboardInterrupt:
    print()
    print('Detenido')

print()

resultados.to_csv('./resultados/resultados_MAR.csv', index=False)
