
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter
from typing import Tuple
from os.path import basename
from math import sqrt, ceil


#
# Config
#

filename = './resultados/resultados.csv'

modo_agregacion = 'mean'  # mean min max median
variables_agrupables = ['ausencia', 'reducido', 'proporcion', 'tecnica']

DEBUG = True
img_extension = 'svg'  # png, svg

COLORES = [
    'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
    'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
    'm', 'b', 'yellow']

PREC_SENS = False

GRAFICAS = ({
    # variable: limites verticales, Y-log
    'f1 score': ((0.49, 1.01), False),
    'exactitud': ((0.49, 1.01), False),
    'similitud': (None, False),
    'tiempo': (None, True)
    } if not PREC_SENS else {
    'sensibilidad': ((0.49, 1.01), False),
    'precision': ((0.49, 1.01), False),
})

#
# Calculos
#


def log(*args, _t: list = [], **kwargs):
    if not _t:
        _t.append(perf_counter())
        _t.append(0)

    s = (
        f'[{round(perf_counter() - _t[0], 3)}s] ' +
        ' '.join((f'{a}' for a in args))).ljust(_t[1])

    if 'end' in kwargs:
        print(s, **kwargs)
    else:
        print(s, **kwargs, end=('\n' if DEBUG else '\r'))

    _t[1] = max(_t[1], len(s))


def plot(resultados: pd.DataFrame, original: pd.Series, titulo: str):

    variables = tuple(c for c in GRAFICAS if c in resultados.columns)

    c = ceil(sqrt(len(variables)))
    plots = (c, c)  # (1, len(variables))
    figsize = (6 * plots[0], 4 * plots[1])

    if PREC_SENS:
        plots = (1, len(variables))
        figsize = (16, 6)

    img, figuras = plt.subplots(
        *plots, figsize=figsize)

    img.supxlabel(titulo, fontweight='bold', fontsize=16)

    for i, variable in enumerate(variables, 1):

        plt.subplot(*plots, i)

        plt.title(variable.capitalize(), fontweight='bold', fontsize=15)
        plt.xlabel('Proporción ausente')

        proporciones = resultados['proporcion'].unique()
        br1 = np.arange(len(proporciones))

        tecnicas = resultados['tecnica'].unique()

        for tecnica, color in zip(tecnicas, COLORES):
            data = resultados[resultados['tecnica'] == tecnica]

            plt.plot(br1, data[variable], color=color, label=tecnica)

        plt.plot(
            br1, [original[variable]] * len(br1),
            color='black', label='Sin ausencia')

        limitY, ylog = GRAFICAS.get(variable)

        if limitY is not None:
            plt.ylim(limitY)
        plt.xticks(br1[::2], proporciones[::2], fontsize=8)
        if ylog:
            plt.yscale('log')

    plt.legend(bbox_to_anchor=(1, 1))
    plt.tight_layout()


#
# Lectura
#

path = filename[:filename.rfind('.')]
if PREC_SENS:
    path += '_sensibilidad_precision'

log(f'Leyendo datos de {filename}')


resultados_sin_agrupar = pd.read_csv(filename)

log('Agrupando datos de iteraciones')

resultados = resultados_sin_agrupar.groupby(
    variables_agrupables).aggregate(modo_agregacion).reset_index()


opciones_reducido: Tuple[bool] = tuple(resultados['reducido'].unique())
log(f'{opciones_reducido=}')
opciones_ausencia: Tuple[str] = tuple(resultados['ausencia'].unique())
log(f'{opciones_ausencia=}')


for reducido in opciones_reducido:

    log(f'Extrayendo resultados reduccion {reducido}')
    resultados_reduccion = resultados[
        resultados['reducido'] == reducido][[
            atr for atr in resultados.columns
            if atr != 'reducido']]

    log(f'Extrayendo resultado reduccion {reducido} original')
    original = resultados_reduccion[
        resultados_reduccion['ausencia'] == 'original'].iloc[0]

    for ausencia in opciones_ausencia:
        if ausencia == 'original':
            continue

        nombre = ausencia.split()
        nombre = '{}_{}_{}.{}'.format(
            path, 'reducido' if reducido else 'completo',
            '_'.join((nombre[0], *nombre[2:])).lower(), img_extension)
        log(f'{nombre=}')

        log(f'Extrayendo resultados ausencia {ausencia}')

        resultados_ausencia = resultados_reduccion[
            resultados_reduccion['ausencia'] == ausencia][[
                atr for atr in resultados_reduccion.columns
                if atr != 'ausencia']]
        # log(*tuple(resultados_ausencia.columns))

        titulo = basename(
            nombre[:nombre.rfind('.')]).replace('_', ' ').upper()

        plot(resultados_ausencia, original, titulo)

        plt.savefig(
            nombre, format=img_extension, dpi=400, bbox_inches='tight')
        # plt.show()

exit()
