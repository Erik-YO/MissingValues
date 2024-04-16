

# Estudio comparativo de técnicas de tratamiento de valores ausentes
Erik Yuste Ortiz. Abril 2024.  

Este repositorio incluye el código utilizado en la elaboración del TFG: Estudio comparativo de técnicas de tratamiento de valores ausentes.


## Dependencias

Se utilizó para su desarrollo Python 3.10 con las siguientes bibliotecas:
-	scikit-learn (v1.3.0) tanto para la implementación como para la obtención de las métricas.
-	pandas (v2.0.3) para la gestión de los datos en DataFrames.
-	scipy (v1.11.1) para la implementación de parte de las técnicas de interpolación.
-	numpy (v1.24.2) para operaciones generales con arrays de datos.
-	matplotlib (v3.7.1) y seaborn (v0.13.0) para la representación gráfica tanto de los resultados del análisis previo como de los resultados finales.
-	impyute (v0.0.8) para la implementación de la técnica de maximización de la esperanza.

**Nota:** para poder utilizar el módulo impyute (v0.0.8) con una versión de numpy igual o superior a la 1.20 se tuvo que sustituir manualmente ```== np.float``` por ```in (np.float16, np.float32, np.float64)``` en el fichero ```impyute/util/checks.py```.


## Conjunto de datos

Para obtener el conjunto de datos:
```
wget -O eeg-eye-state.arff https://www.openml.org/data/download/1587924/phplE7q6h
```
o accediendo a la dirección:
```https://archive.ics.uci.edu/dataset/264/eeg+eye+state```

El conjunto se encuentra transformado a formato csv en el respositorio.
