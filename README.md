# Formalización Semántica de Conceptos Clínicos mediante Grafos de Conocimiento: Aplicación a SNOMED CT
Proyecto para realizar la formalización semántica de conceptos clínicos mediante grafos de conocimiento, aplicado como caso de uso a la terminología SNOMED CT. Máster Universitario en Inteligencia Artificial UPM. Curso 2020-2021.

Los modelos finales de FastText para la versión internacional de SNOMED CT con todas las relaciones y solo las relaciones es_un[a], así como el modelo para la versión en español y los corpora utilizados para entrenar dichos modelos están disponibles en el siguiente enlace:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5061247.svg)](https://doi.org/10.5281/zenodo.5061247)

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
### Generar diccionario de conceptos de SNOMED CT
Para obtener un diccionario con los conceptos de SNOMED CT en los que utiliza como clave sus IDs y tiene para cada entrada lo siguiente:
- FSN: Fully Specified Name.
- description: Las descripciones, incluida la que es el FSN.
- relations: Tuplas que indican con qué elemento tiene una relación. El primer elemento de la tupla es el ConceptID del destino y el segundo elemento es qué tipo de relación es.
- relationsAux: De cara a formar el corpus, este no tiene validez ya que solo lo utilizamos para que la relación es_un sea simétricas y podamos desplazarnos por el grafo más fácilmente para analizar ciertos aspectos o realizar algunas tareas como eliminar los metadatos.
- definition: Una pequeña definición en español del concepto. Muy pocos conceptos la tienen.
- semantic_tag: Indica a qué categoría semántica pertenece el concepto.
- vecinos: Indica qué nodos son vecinos mediante qué relación, por lo que tiene forma de lista de [typeID, destID]. NO se está utilizando la falsa simetría que utilizan en Owl2Vec*.

Al script *read_SNOMED.py* hay que pasarle el fichero de conceptos internacionales, el fichero de descripciones, el fichero de definiciones y el fichero de relaciones internacionales. Si se quiere utilizar una versión nacional de SNOMED hay que utilizar un fichero de descripciones y definiciones nacional, pero conceptos y relaciones siempre tiene que ser el internacional.
```
python3 read_SNOMED.py conceptos_internacional_path descriptions_path definitions_path relations_internacional_path
```
### Generar copurs
Para generar el corpus simplemente hay que ejecutar el script *generar_corpus.py* llamándolo con la ruta relativa al fichero de conceptos y al fichero de metadatos dentro de la carpeta *concepts/* y con la profundidad de los caminos de palabras e IDs. Un camino de 1 equivale a la siguiente secuencia: (concepto1, relación12, concepto2), uno de camino 2 equivale a: (concepto1, relación12, concepto2, relation23, concepto3) y así sucesivamente.
```
python3 generar_corpus.py conceptos_path metadatos_path id_depth word_depth
```
### Entrenar modelo
Para entrenar el modelo hay que ejecutar el script *train_model.py* llamándolo con 'w2v' o 'ft' en función de si queremos entrenar Word2Vec o FastText, la ruta relativa al corpus dentro de la carpeta *corpus/*, los parámetros para los caminos tal y como hemos mencionado en el script anterior y los hiperparámetros del modelo respecto al tamaño de embedding y ventana. También hay que pasarle el idioma del corpus.
```
python3 train_model.py model_type corpus_path id_depth word_depth embedding_size window_size language
```
### Obtener el diccionario de vectores para los conceptos de entrenamiento
Para obtener el diccionario de vectores de los conceptos de entrenamiento es necesario pasarle como entrada al script *train_dic.py* si estamos utilizando Word2Vec ('w2v') o FastText ('ft'), la ruta al modelo dentro de la carpeta *models/*, la ruta al fichero de conceptos de entranamiento dentro de *concepts/* y el idioma del corpus (english si inglés, spanish si español, etc.).
```
python3 train_dic.py model_type model_path concepts_path language
```
### Evaluar el modelo
Para obtener un log con información del rendimiento del modelo, es necesario pasarle al script *evaluate_model.py* el tipo de modelo de lenguaje que se está usando, la ruta hasta el modelo y el idioma del corpus (english si inglés, spanish si español, etc.). El concepto de entrenamiento y el concepto de evaluación hay que modificarlos dentro del código para permitir la evaluación de varios conjuntos.
```
python3 evaluate_model.py model_type model_path language
```
### Lectura de logs
Para obtener la lectura del log de la evaluación hay que ejecutar el script *read_logs.py* pasándole como parámetros la ruta al log, los conceptos y los metadatos y el número total de conceptos de evaluación.
```
python3 read_logs.py log_path concepts_path metadatos_path total_concepts
```
