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
To obtain the dictionary with the SNOMED CT concepts, where the key is the concept's ID and that is has the following structure:
- FSN: Fully Specified Name.
- description: The descriptions of the concept, which are the synonyms and FSN.
- relations: Tuples that indicate with which concept has a certain relationship. The first element of the tuple is the SCT ID of the tail concept and the second element is the ID of the relationship.
- relationsAux: This is to create the corpus and it is not valid otherwise. This is used to create symmetry for is_a relationships and to be able to easily travel the graph to analyze certain aspects.
- definition: A small definition of the concept in Spanish. Few concepts have it and it is not used.
- semantic_tag: The semantic tag of the concept.
- vecinos: Indicates which nodes are neighbours through a relationship and is a list of tuples [typeID, destID]. It is not using the false symmetry applied in Owl2Vec*.

As parameters for the script *read_SNOMED.py*, you need to pass the international files, the file with the descriptions, the file with the definitions, and the file with the international relationships. If you want to use a national version of SNOMED CT (such as the Spanish one), the description and definitions files need to be of the national version, while the concept and relationships ones need to be from the international version.
```
python3 read_SNOMED.py conceptos_internacional_path descriptions_path definitions_path relations_internacional_path
```
### Generate corpus
To generate the corpus, we need to run the script *generar_corpus.py* with the relative path to the concepts file and to the metadata file inside the *concepts/* folder. You also need to specify the depth of the random walks for both word sentences and ID sentence. A walk of depth 1 equals the following sentence: (concept_1, relationship_1_2, concept_2); uno of depth 2 equals: (concept_1, relationship_1_2, concept_2, relationship_2_3, concept_3) and so on.
```
python3 generar_corpus.py conceptos_path metadatos_path id_depth word_depth
```
### Train the model
To train the model, you need to run the script *train_model.py*, indicating the model type ('w2v' if Word2Vec or 'ft' if FastText), the relative path to the corpus inside the *corpus/* folder, the depths for random walks, and the hyperparameters of the model, such as the embedding and window sizes. You also need to specify the language of the corpus.
```
python3 train_model.py model_type corpus_path id_depth word_depth embedding_size window_size language
```
### Obtain word embedding dictionary for training concepts
To obtain the word embedding dictionary for the training concept, you need to run the script *train_dic.py* indicating if we are using Word2Vec ('w2v') or FastText ('ft'), the path to the model inside the *models/* folder, the path to the training concepts file inside the *concepts/* folder, and the language of the corpus ('english' if English, 'spanish' if Spanish, etc.).
```
python3 train_dic.py model_type model_path concepts_path language
```
### Evaluate the model
To evaluate the model and obtain information about its performance, you need to run the script *evaluate_model.py* with which language model is being used ('w2v' for Word2Vec, 'ft' for FastText, 'bert' for BERT or 'sbert' for SBERT), the path to said model and the corpus language ('english' for English, 'spanish' for Spanish, etc.). The training and evaluation concepts need to be modified from code to evaluate several subsets at once. 

If using SBERT, you can specify any word for the model_path. The tokenizer and the SBERT model needs to be change from code.
```
python3 evaluate_model.py model_type model_path language
```
### Logs reading
To better understand the information of the evalutation log, you need to run the script *read_logs.py* with the following parameters: path to said log, path to concepts and metadata, and total number of evaluation concepts.
```
python3 read_logs.py log_path concepts_path metadatos_path total_concepts
```
