# Automatic building of SNOMED CT postcoordinated expressions through Knowledge Graph Embeddings
This is the GitHub repository that contains the scripts used for the article *Automatic building of SNOMED CT postcoordinated expressions through Knowledge Graph Embeddings*. Our aim is to develop a tool that can be used to create a suggestion of a postcoordinate expression for SNOMED CT given a clinical term. This tool uses word embedding similarity and analogies to generate such postcoordination.

We have made available a website to test the tool: [WIP]

The article where we explain our methodology and show our results can be seen in: [WIP]

## Requirements
The library requirements are the following:
- re == 2.2.1
- gensim == 3.6.0
- json == 2.0.9
- nltk == 3.2.5
- numpy == 1.19.5
- pd == 1.1.5
- sklearn == 0.22.2

## File structure
The default file structure we used is the following, which can be changed by modifying the function *get_jerarquia()* in the script *postcoordinate_functions.py*:
./\
|-- input/\
|-- logs/\
|-- models/\
|-- dicts/\
|-- corpus/\
|-- concepts/

*input/* is where you can place the input files, which are the following SNOMED CT files: relationships, international concepts and descriptions, and Spanish descriptions file.\
*logs/* is where we can find the evaluation logs.\
*models/* is where we can find the trained embedding models.\
*dicts/* is where we save the dictionaries that link concepts and embeddings.\
*corpus/* is where we find the different corpora used to train the models.\
*concepts/* is where we find the json with SNOMED CT concepts.
## Scripts
### Generate SNOMED CT concept dictionary
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
### Generate corpus
Para generar el corpus simplemente hay que ejecutar el script *generar_corpus.py* llamándolo con la ruta relativa al fichero de conceptos y al fichero de metadatos dentro de la carpeta *concepts/* y con la profundidad de los caminos de palabras e IDs. Un camino de 1 equivale a la siguiente secuencia: (concepto1, relación12, concepto2), uno de camino 2 equivale a: (concepto1, relación12, concepto2, relation23, concepto3) y así sucesivamente.
```
python3 generar_corpus.py conceptos_path metadatos_path id_depth word_depth
```
### Train the model
Para entrenar el modelo hay que ejecutar el script *train_model.py* llamándolo con 'w2v' o 'ft' en función de si queremos entrenar Word2Vec o FastText, la ruta relativa al corpus dentro de la carpeta *corpus/*, los parámetros para los caminos tal y como hemos mencionado en el script anterior y los hiperparámetros del modelo respecto al tamaño de embedding y ventana. También hay que pasarle el idioma del corpus.
```
python3 train_model.py model_type corpus_path id_depth word_depth embedding_size window_size language
```
### Obtain word embedding dictionary for training concepts
Para obtener el diccionario de vectores de los conceptos de entrenamiento es necesario pasarle como entrada al script *train_dic.py* si estamos utilizando Word2Vec ('w2v') o FastText ('ft'), la ruta al modelo dentro de la carpeta *models/*, la ruta al fichero de conceptos de entranamiento dentro de *concepts/* y el idioma del corpus (english si inglés, spanish si español, etc.).
```
python3 train_dic.py model_type model_path concepts_path language
```
### Evaluate the model
To evaluate the model and obtain information about its performance, you need to run the script *evaluate_model.py* with which language model is being used, the path to said model and the corpus language ('english' for English, 'spanish' for Spanish, etc.). The training and evaluation concepts need to be modified from code to evaluate several subsets at once.
```
python3 evaluate_model.py model_type model_path language
```
### Logs reading
To better understand the information of the evalutation log, you need to run the script *read_logs.py* with the following parameters: path to said log, path to concepts and metadata, and total number of evaluation concepts.
```
python3 read_logs.py log_path concepts_path metadatos_path total_concepts
```
