# Formalización Semántica de Conceptos Clínicos mediante Grafos de Conocimiento: Aplicación a SNOMED CT
Proyecto para realizar la formalización semántica de conceptos clínicos mediante grafos de conocimiento, aplicado como caso de uso a la terminología SNOMED CT. Máster Universitario en Inteligencia Artificial UPM. Curso 2020-2021.

Los modelos finales de FastText para la versión internacional de SNOMED CT con todas las relaciones y solo las relaciones es_un[a], así como el modelo para la versión en español y los corpora utilizados para entrenar dichos modelos están disponibles en el siguiente enlace: [ENLACE]

## Estructura de ficheros
La estructura predeterminada de ficheros que utilizamos es la que aparece a continuación, aunque puede ser modificada cambiando la función *get_jerarquia()* en el script *postcoordinate_functions.py*:\
./\
|-- input/\
|-- logs/\
|-- models/\
|-- dicts/\
|-- corpus/\
|-- concepts/

En *input/* se pondrían los ficheros de entrada, siendo estos los siguientes ficheros de SNOMED CT: ficheros de relaciones, conceptos y descripciones internacional y fichero de descripciones español.\
En *logs/* encontramos los logs obtenidos al realizar la evaluación.\
En *models/* pondríamos los modelos entrenados de Word2Vec o FastText.\
En *dicts/* guardamos los diccionarios de vectores.\
En *corpus/* estarían los corpora que utilizamos para entrenar los modelos.\
En *concepts/* estarían los diccionarios con la información de los conceptos extraídos de los ficheros de SNOMED CT.
## Scripts
### Generar copurs
Para generar el corpus simplemente hay que ejecutar el script *generar_corpus.py*.
```
python3 generar_corpus.py 
```
