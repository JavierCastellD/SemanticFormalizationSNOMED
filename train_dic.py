import sys
import gensim
import json
import nltk
import numpy  as np
from postcoordinate_functions import string_keys_to_int, preprocesar_frase, get_jerarquia

PATH, INPUT, LOGS, MODELS, DICT, CORPUS, CONCEPTS = get_jerarquia()

model_name = sys.argv[1]
model_path = sys.argv[2]
concepts_path = sys.argv[3]
language = sys.argv[4]

for i in range(1):
    print('Paso:', i)
    ##############################
    # ACTUALIZACIÓN DE VARIABLES #
    ##############################

    # Identificador de prueba
    id_prueba = '_test' + str(i)
    
    # Diccionarios de conceptos
    CONCEPTS_FILE = concepts_path

    # Cargamos los conceptos de entrenamiento
    training_concepts_file = open(PATH + CONCEPTS + CONCEPTS_FILE)
    concepts_training = string_keys_to_int(json.load(training_concepts_file))
    training_concepts_file.close()

    ##########################
    # PARÁMETROS IMPORTANTES #
    ##########################
    if model_name == 'w2v':
        print('Modelo: Word2Vec')
        model = gensim.models.Word2Vec.load(PATH + MODELS + model_path)
        dict_name = 'concepts_dictionary_w2v'+id_prueba+'.json'
    else:
        print('Modelo: FastText')
        model = gensim.models.FastText.load(PATH + MODELS + model_path)
        dict_name = 'concepts_dictionary_ft'+id_prueba+'.json'

    #########################################
    # OBTENEMOS EL DICCIONARIO DE CONCEPTOS #
    #########################################
    print('Vamos a obtener el diccionario de conceptos')
    dict_concepts = {}
    n = 0
    for conceptID, concept in concepts_training.items():
        # Usaremos el FSN como clave
        name = preprocesar_frase(concept['FSN'])
        palabras_name = nltk.word_tokenize(name, language=language)
        
        # Obtenemos el WE
        vectors = []
        for palabra in palabras_name:
            vectors.append(model.wv.get_vector(palabra))

        try:
            vector_URI = model.wv.get_vector(str(conceptID))
        except:
            vector_URI = np.array([])
            n += 1

        v = sum(vectors)
        if isinstance(v, np.ndarray):
            dict_concepts[conceptID] = {'vector': v.tolist(), 'vectorURI': vector_URI.tolist(), 'FSN': name}
        else:
            n += 1

    concepts_dict_file = open(PATH + DICT + dict_name, 'w')
    json.dump(dict_concepts, concepts_dict_file, indent=4)
    concepts_dict_file.close()  
   
    
    



