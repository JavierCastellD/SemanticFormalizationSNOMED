import sys
import time
import gensim
import json
import nltk
import numpy  as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from postcoordinate_functions import string_keys_to_int, preprocesar_frase, get_jerarquia, mismo_semantic_tag, mismo_semantic_tag_rels, mismas_relaciones, getIDdestino, concepto_destino_correcto

PATH, INPUT, LOGS, MODELS, DICT, CORPUS, CONCEPTS = get_jerarquia()
ES_UN_ID = 116680003
TOP_RELATIONS = [762705008, 246061005, 410663007, 408739003]
METADATOS_FILE = 'metadatos.json'

model_name = sys.argv[1]
model_path = sys.argv[2]
language = sys.argv[3]

# Cargamos los metadatos
metadatos_file = open(PATH + CONCEPTS + METADATOS_FILE)
metadatos = string_keys_to_int(json.load(metadatos_file))
metadatos_file.close()

for i in range(1):
    t_step_0 = time.time()
    print('Paso:', i)
    ##############################
    # ACTUALIZACIÓN DE VARIABLES #
    ##############################

    # Identificador de prueba
    id_prueba = '_test' + str(i)
    
    # Diccionarios de conceptos
    CONCEPTS_TRAINING = 'active_concepts_testing.json'
    CONCEPTS_EVALUATION = 'concepts_test.json'

    # Logs
    log_name = 'log_' + id_prueba + '.tsv'

    # Cargamos los conceptos de entrenamiento
    training_concepts_file = open(PATH + CONCEPTS + CONCEPTS_TRAINING)
    concepts_training = string_keys_to_int(json.load(training_concepts_file))
    training_concepts_file.close()

    # Cargamos los conceptos de evaluación
    evaluation_concepts_file = open(PATH + CONCEPTS + CONCEPTS_EVALUATION)
    concepts_evaluation = string_keys_to_int(json.load(evaluation_concepts_file))
    evaluation_concepts_file.close()

    ##########################
    # PARÁMETROS IMPORTANTES #
    ##########################
    if model_name == 'w2v':
        print('Modelo: Word2Vec')
        model = gensim.models.Word2Vec.load(PATH + MODELS + model_path)
    else:
        print('Modelo: FastText')
        model = gensim.models.FastText.load(PATH + MODELS + model_path)

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

    ####################
    # OBTENEMOS EL LOG #
    ####################
    print('Empezamos a obtener el log')

    embs = []
    embs_URI = []
    fsns = []
    ids = []
    for id, ele in dict_concepts.items():
        embs_URI.append(np.array(ele['vectorURI']))
        fsns.append(ele['FSN'])
        embs.append(np.array(ele['vector']))
        ids.append(id)

    # Estructura intermedia para luego hacer el log
    stats = {}
    
    # Abrimos el fichero de log
    log = open(PATH + LOGS + log_name, 'w')

    # Escribimos la cabecera
    log.write('Concepto\tID\tSemantic tag\tTop\tTop Rel\tSimilares\tPalabras no contenida\t% palabras contenidas\t'
              'Relaciones igual 1\t% igual 1\tNombres igual 1\tID 1\t'
              'Relaciones igual match\t% igual match\tNombres igual match\tID match\n')
    
    concepts_vectors = []
    concepts_ids = []
    
    # Por cada concepto de evaluación
    for conceptID, concept in list(concepts_evaluation.items()):
      stats[conceptID] = {'sem' : '', 'FSN' : '', 'pal_no_c' : 0, '%_pal' : 0, 'Vector' : True}
      # Obtenemos su FSN
      FSN = concept['FSN']
    
      # Obtenemos su semantic tag
      semantic_tag = concept['semantic_tag']
      
      stats[conceptID]['sem'] = semantic_tag
      stats[conceptID]['FSN'] = FSN
    
      # Preprocesamos el nombre y separamos las palabras
      name = preprocesar_frase(FSN)
      palabras_name = nltk.word_tokenize(name, language=language)
    
      # Inicializamos esta variable auxiliar para controlar cuántas palabras no
      # están en el diccionario
      n_palabras_no_diccionario = 0
    
      # Obtenemos el vector de cada palabra, teniendo en cuenta que hay palabras
      # que quizá no estén en el diccionario
      vectors = []
      for palabra in palabras_name:
        try:
          vectors.append(model.wv.get_vector(palabra))
        except:
          #print('En el diccionario no está:', palabra)
          n_palabras_no_diccionario += 1
    
      # Almacenamos de cada concepto de evaluación información útil para el log
      stats[conceptID]['pal_no_c'] = n_palabras_no_diccionario
      if len(palabras_name) != 0: 
        stats[conceptID]['%_pal'] = 1 - round(n_palabras_no_diccionario / len(palabras_name), 4)
      else:
        print('NO palabras name en', FSN)
        stats[conceptID]['%_pal'] = 0
      
      # Tenemos en cuenta también aquellos vectores que hayan sido vacíos
      if vectors != []:
        stats[conceptID]['Vector'] = True
        word_vector = sum(vectors)
        concepts_ids.append(conceptID)
        concepts_vectors.append(word_vector)
        stats[conceptID]['wv'] = np.array(word_vector)
      else:
        stats[conceptID]['Vector'] = False
    
    t2 = time.time()
    print('Cálculo de similitudes')
    sims_V = cosine_similarity(X=concepts_vectors, Y=embs)
    print('Cálculo hecho')
    t3 = time.time()
    print('Tiempo similitud', t3-t2)
    
    cont = 0
    total = len(concepts_ids)
    
    # Para las analogías del primer concepto
    wordsX = []
    eval_ids_an = []
    rel_ids = []

    # Para las analogías del concepto topRel 
    wordsX2 = []
    eval_ids_an2 = []
    rel_ids2 = []
    
    # Para cada concepto de evaluación, calculamos la categoría semántica
    # y las relaciones y preparamos todo para hacer luego las analogías
    for sim_V, ID in zip(sims_V, concepts_ids):
      # Esto es para mostrar cuánto tiempo queda aproximado en segundos
      cont += 1
      porcentaje_total = round(cont / total * 100, 2)
      t4 = time.time()
      t_aprox = 100 * round(t4 - t3, 2) / porcentaje_total
      print('Llevamos:', porcentaje_total, '% Tiempo restante aproximado:', t_aprox - (t4 - t3))
        
      # Obtenemos la lista para vectores de palabras
      sim_list = list(zip(fsns, ids, sim_V))
      
      # Ordenamos la lista de mayor a menor
      sim_list.sort(key=lambda x: x[2], reverse=True)
      sim_list_V = sim_list[:10]
      
      salida = stats[ID]['FSN'] + '\t' + str(ID) + '\t' + stats[ID]['sem'] + '\t'
      
      # Obtenemos la etiqueta semántica
      st, pos = mismo_semantic_tag(sim_list_V, concepts_training)
      
      # Si se ha acertado el tipo semántico, se marca como 0
      # En caso contrario, se marca con un -1
      top = -1
      if st == stats[ID]['sem']:
        top = 0
      
      # Obtenemos el concepto que use esa etiqueta semántica y no sea de navegación
      topRel = mismo_semantic_tag_rels(stats[ID]['sem'], sim_list_V, concepts_training)
      
      similares_solo_IDs = [id for _, id, _ in sim_list_V]
      salida += str(top) + '\t' + str(topRel) + '\t' + str(similares_solo_IDs) + '\t' + str(stats[ID]['pal_no_c']) + '\t' + str(stats[ID]['%_pal']) + '\t'

      # Para el más similar siempre hacemos el cálculo de relaciones parecidas y de analogía
      ID_conceptB = sim_list_V[0][1]
      porcentajeIgual, iguales = mismas_relaciones(concepts_training[ID_conceptB], concepts_evaluation[ID], metadatos)

      salida += str(len(iguales)) + '\t' + str(porcentajeIgual) + '\t' + str(iguales) + '\t' + str(ID_conceptB) + '\t'

      # Hacemos la analogía
      for relID in iguales:
          idDestino = getIDdestino(concepts_training[ID_conceptB]['relations'], relID)

          if idDestino == -1:
              print('Error, no hay una relación de tipo', relID, 'en el concepto', ID_conceptB)
          else:
              wordX = np.array(dict_concepts[idDestino]['vector']) - np.array(dict_concepts[ID_conceptB]['vector']) + \
                      stats[ID]['wv']
              wordsX.append(wordX)
              eval_ids_an.append(ID)
              rel_ids.append(relID)

      # Además, si hemos encontrado correctamente la categoría semántica
      if top != -1:
        # Esto lo hacemos por si solo hay conceptos de navegación, para que utilice entonces
        # el primero de ellos
        if topRel == -1:
            topRel = pos

        # Si tenemos que el concepto es el mismo que el primero, ya hemos hecho los cálculos para ese
        if topRel != 0:
            # Comprobamos en qué aspectos se parece al concepto que comparte categoría semántica y con relaciones
            ID_conceptB = sim_list_V[topRel][1]

            porcentajeIgual, iguales = mismas_relaciones(concepts_training[ID_conceptB], concepts_evaluation[ID], metadatos)

        salida += str(len(iguales)) + '\t' + str(porcentajeIgual) + '\t' + str(iguales) + '\t' + str(ID_conceptB) + '\t'

        # Hacemos la analogía
        for relID in iguales:
          idDestino = getIDdestino(concepts_training[ID_conceptB]['relations'], relID)

          if idDestino == -1:
            print('Error, no hay una relación de tipo', relID, 'en el concepto', ID_conceptB)
          else:
            wordX = np.array(dict_concepts[idDestino]['vector']) - np.array(dict_concepts[ID_conceptB]['vector']) + stats[ID]['wv']
            wordsX2.append(wordX)
            eval_ids_an2.append(ID)
            rel_ids2.append(relID)

      # Si no hay ninguno que comparta tipo semántico
      else:
        salida += '0\t0\t[]\t-1'
        
      # Escribimos la salida
      log.write(salida + '\n')
    
    # Escribimos la salida de aquellos cuyo vector era vacío, ya que no se puede saber nada de ellos
    for ID, value in stats.items():
      if not value['Vector']:
        salida = value['FSN'] + '\t' + str(ID) + '\t' + value['sem'] + '\t'
        salida += str(-1) + '\t' + str(-1) + '\t[]\t' + str(value['pal_no_c']) + '\t0\t0\t0\t[]\t-1\t0\t0\t[]\t-1'
        log.write(salida + '\n')
    
    log.close()
    
    # Abrimos el log como un dataframe para añadir la nueva información
    df = pd.read_csv(PATH + LOGS + log_name, sep='\t', index_col=False)

    df['Analogias 1'] = np.nan
    df['Analogias 1 Lista'] = np.nan
    df['Analogias Sem'] = np.nan
    df['Analogias Sem Lista'] = np.nan

    df['Analogias 1'] = df['Analogias 1'].astype('object')
    df['Analogias 1 Lista'] = df['Analogias 1 Lista'].astype('object')
    df['Analogias Sem'] = df['Analogias Sem'].astype('object')
    df['Analogias Sem Lista'] = df['Analogias Sem Lista'].astype('object')
    
     # Hacemos el cálculo de similitud para las analogías del top1
    t_step_2 = time.time()
    print('Cálculo de similitud para las analogías de longitud:', len(eval_ids_an))
    sims_analogies = cosine_similarity(X=wordsX, Y=embs)
    t_step_4 = time.time()
    print('Tiempo cálculo similitud analogías:', t_step_4 - t_step_2)
    
    topAns = []
    topAnalog = []
    evalID_actual = eval_ids_an[0]

    cont = 0
    total = len(eval_ids_an)
    
    # Por cada analogía
    for sim, evalID, relID in zip(sims_analogies, eval_ids_an, rel_ids):
      # Esto es para mostrar cuánto tiempo queda aproximado en segundos
      cont += 1
      porcentaje_total = round(cont / total * 100, 2)
      t_step_3 = time.time()
      t_aprox = 100 * round(t_step_3 - t_step_2, 2) / porcentaje_total
      print('Para analogías - Llevamos:', porcentaje_total, '% Tiempo restante aproximado:', t_aprox - (t_step_3 - t_step_2))
      
      # Si estamos mirando aún las analogías de otro concepto, escribimos lo del anterior
      if evalID != evalID_actual:
        df.loc[df['ID'] == evalID_actual, 'Analogias 1'] = pd.Series([topAns] * len(df))
        df.loc[df['ID'] == evalID_actual, 'Analogias 1 Lista'] = pd.Series([topAnalog] * len(df))

        topAns = []
        topAnalog = []
        evalID_actual = evalID
    
      # Obtenemos los 10 conceptos más similares de la analogía
      sim_list = list(zip(ids, sim))
      sim_list.sort(key=lambda x : x[1], reverse = True)
      
      analogias = sim_list[:10]
      
      analogias_solo_IDs = [id for id, _ in analogias]
      
      # Comprobamos si hemos acertado para alguno
      topAn = concepto_destino_correcto(analogias, concepts_evaluation[evalID], relID)

      topAns.append([relID, topAn])
      topAnalog.append(analogias_solo_IDs)

    df.loc[df['ID'] == evalID_actual, 'Analogias 1'] = pd.Series([topAns] * len(df))
    df.loc[df['ID'] == evalID_actual, 'Analogias 1 Lista'] = pd.Series([topAnalog] * len(df))

    #################################
    # REPETIMOS PARA LOS SEMÁNTICOS #
    #################################

    # Cálculo similitud para las analogías de los tipos semánticos
    t_step_2 = time.time()
    print('Cálculo de similitud para las analogías de longitud:', len(eval_ids_an2))
    sims_analogies2 = cosine_similarity(X=wordsX2, Y=embs)
    t_step_4 = time.time()
    print('Tiempo cálculo similitud analogías:', t_step_4 - t_step_2)

    topAns = []
    topAnalog = []
    evalID_actual = eval_ids_an2[0]

    cont = 0
    total = len(eval_ids_an2)

    for sim, evalID, relID in zip(sims_analogies2, eval_ids_an2, rel_ids2):
      # Esto es para mostrar cuánto tiempo queda aproximado en segundos
      cont += 1
      porcentaje_total = round(cont / total * 100, 2)
      t_step_3 = time.time()
      t_aprox = 100 * round(t_step_3 - t_step_2, 2) / porcentaje_total
      print('Para analogías sem - Llevamos:', porcentaje_total, '% Tiempo restante aproximado:', t_aprox - (t_step_3 - t_step_2))

      # Si estamos mirando aún las analogías de otro concepto, escribimos lo del anterior
      if evalID != evalID_actual:
        df.loc[df['ID'] == evalID_actual, 'Analogias Sem'] = pd.Series([topAns] * len(df))
        df.loc[df['ID'] == evalID_actual, 'Analogias Sem Lista'] = pd.Series([topAnalog] * len(df))

        topAns = []
        topAnalog = []
        evalID_actual = evalID

      # Obtenemos los 10 conceptos más similares de la analogía
      sim_list = list(zip(ids, sim))

      sim_list.sort(key=lambda x: x[1], reverse=True)

      analogias = sim_list[:10]

      analogias_solo_IDs = [id for id, _ in analogias]
        
      # Comprobamos si hemos acertado para alguno
      topAn = concepto_destino_correcto(analogias, concepts_evaluation[evalID], relID)

      topAns.append([relID, topAn])
      topAnalog.append(analogias_solo_IDs)

    df.loc[df['ID'] == evalID_actual, 'Analogias Sem'] = pd.Series([topAns] * len(df))
    df.loc[df['ID'] == evalID_actual, 'Analogias Sem Lista'] = pd.Series([topAnalog] * len(df))

    # Guardamos el dataframe
    df.to_csv(PATH + LOGS + log_name, sep='\t', index=False)
    
    
    
    
    



