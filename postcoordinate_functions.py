import re
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

ES_UN_ID = 116680003
TOP_RELATIONS = [762705008, 246061005, 410663007, 408739003]

def get_jerarquia():
  path = './'
  inputs = 'input/'
  logs = 'logs/'
  models = 'models/'
  dicts = 'dicts/'
  corpus = 'corpus/'
  concepts = 'concepts/'
  
  return path, inputs, logs, models, dicts, corpus, concepts


# Función para realizar la poscoordinación
# Necesita el término a poscoordinar, el modelo, el diccionario de vectores,
# el diccionario de conceptos de entrenamiento y el de metadatos 
def post_coordinate(name, model, dic, concepts, metadatos):
  coordinated = {'term' : name, 'semantic_tag' : '', 'relations' : []}
  
  # Obtenemos el vector
  vC = get_vector(name, model)
  
  # Obtenemos los conceptos más similares
  embs, fsns, ids = split_dic(dic)
  sim_list = mas_similar(embs, fsns, ids, vC, n=5)
  
  # Obtenemos la etiqueta semántica
  st, pos = mismo_semantic_tag(sim_list, concepts)
  coordinated['semantic_tag'] = st
  
  # Obtenemos el concepto más similar que no sea de navegación y comparta la etiqueta
  concept_pos = mismo_semantic_tag_rels(st, sim_list, concepts)
  
  # Si solo hay conceptos de navegación, nos quedamos con el primer elemento de ese tipo
  if concept_pos == -1:
    concept_pos = pos
  
  # Obtenemos el identificador del concepto que usaremos como base
  concept_ID = sim_list[concept_pos][1]
  
  # Obtenemos el vector de dicho concepto
  vA = dic[concept_ID]['vector']
  
  # Obtenemos las relaciones
  relations_aux = set([top_relation(metadatos, typeID) for destID, typeID in concepts[concept_ID]['relations']])
  
  relations = []
  for rel in list(relations_aux):
    idDestino = getIDdestino(concepts[concept_ID]['relations'], rel)
    relations.append([idDestino, rel])
  
  # Para cada relación del concepto padre
  rels_aux = []
  for destID, typeID in relations:
    vB = dic[destID]['vector']
    
    rel_destino_id = get_analogias(embs, ids, vA, vB, vC, n=1)[0][0]
    
    rels_aux.append([rel_destino_id, typeID])
  
  coordinated['relations'] = rels_aux
  
  return coordinated

# Obtiene el vector de la palabra que recibe como parámetro utilizando
# el modelo que se le pase
def get_vector(name, model):
  s = preprocesar_frase(name)
  s = nltk.word_tokenize(s, language='english')
  
  v = []
  for n in s:
    try:
      v.append(model.wv.get_vector(n))
    except:
      print('El modelo no reconoce:',n)
  
  return sum(v)


# Separa el diccionario de conceptos en tres listas: embeddings, fsns e ids
def split_dic(dic):
    embs = []
    fsns = []
    ids = []

    for id, ele in dic.items():
        fsns.append(ele['FSN'])
        embs.append(np.array(ele['vector']))
        ids.append(id)

    return embs, fsns, ids

# Función para que las claves del diccionario sean números en lugar de cadenas tras leerlo del JSON
def string_keys_to_int(dic):
  dic_aux = {}

  for key in dic:
    dic_aux[int(key)] = dic[key]

  return dic_aux

# Recibe las descripciones de un concepto y las devuelve
# sin duplicados
def remove_duplicates(descriptions):
    no_duplicates = []

    for elem in descriptions:
        if isinstance(elem, str):
            e = re.sub('\(.+\)', '', elem)
            e = e.strip()

            if e not in no_duplicates:
                no_duplicates.append(e)

    return no_duplicates

# Función para preprocesar frases para W2V y FT
def preprocesar_frase(frase):
  # Pasamos el texto a minúsculas
  s = frase.lower()

  # Aislamos ciertos símbolos
  simbolos = ['(',')','.','[',']',':','-','/']
  for simb in simbolos:
    s = s.replace(simb, ' ' + simb + ' ')

  # Quitamos las palabras entre paréntesis
  s = re.sub('\(.+\)', ' ', s)

  # Quitamos dobles espacios y espacios finales
  s = re.sub(' +', ' ', s)
  s = s.strip()

  return s

# Función para preprocesar textos para W2V y FT
def preprocesar_texto(texto, lang='spanish'):
  sentences = []

  for sentence in texto:
    s = preprocesar_frase(sentence)

    # Dividimos la frase en palabras
    s = nltk.word_tokenize(s, language=lang)

    # El corpus lo dejamos como un array de arrays de palabras, que es el
    # formato que pide Gensim
    sentences.append(s)

  return sentences


# Devuelve los n elementos más similares a vector de entre los conceptos del diccionario
# Recibe la lista con los embeddings del diccionario, sus nombres, sus ids,
# el vector a comparar y un número de elementos similares a devolver
def mas_similar(embeddings, names, ids, vector, n=10):
    # Calculamos la similaridad con los elementos del diccionario
    similaridades = cosine_similarity(X=vector.reshape(1, -1), Y=embeddings)[0]

    # Concatenamos los nombres, ids y valores de similaridad
    sim_list = list(zip(names, ids, similaridades))

    # Ordenamos la lista de mayor a menor
    sim_list.sort(key=lambda x: x[2], reverse=True)

    # Devolvemos los n primeros elementos
    return sim_list[:n]

# Compara cuál es la categoría semántica más común entre los n primeros elementos
# de list_similar. Devuelve esa categoría y cuál es el primer elemento de ese tipo
def mismo_semantic_tag(list_similar, concepts_training, n=5):
    aux = {}

    n = 0
    for _, id, _ in list_similar[:n]:
        st = concepts_training[id]['semantic_tag']
        if st in aux:
            aux[st]['val'] += 1
        else:
            aux[st] = {'val': 1, 'p': n}

        n += 1

    most = list(aux.keys())[0]

    for k, v in aux.items():
        if v['val'] > aux[most]['val']:
            most = k

    return most, aux[most]['p']


# Función para comprobar si en una lista de listas de dos elementos, alguna
# no pertenece al tipo de relación indicado por id
def contiene_relacion_distinta_a(list_lists, id=ES_UN_ID):
    if len(list_lists) == 0:
        return False

    for tup in list_lists:
        if tup[1] != ES_UN_ID:
            return True

    return False


# Compara si algún elemento de list_similar es de tipo semantic_tag y si no es un concepto de navegación y devuelve
# la posición dentro de la lista de dicho elemento. Devuelve -1 en caso contrario
def mismo_semantic_tag_rels(semantic_tag, list_similar, concepts_training):
    n = 0
    for _, id, _ in list_similar:
        if semantic_tag == concepts_training[id]['semantic_tag'] and contiene_relacion_distinta_a(
                concepts_training[id]['relations']):
            return n
        n += 1

    return -1

# A partir de dos conceptos, devuelve el porcentaje de relaciones de A que
# existen en B y cuáles son. En caso de que el concepto A no tenga relaciones,
# devuelve -1 y una lista vacía
def mismas_relaciones(conceptA, conceptB, metadatos):
    relA = set([typeID for destID, typeID in conceptA['relations']])
    relB = set([typeID for destID, typeID in conceptB['relations']])

    if len(relA) == 0:
      return (-1, [])

    relA_aux = []
    relB_aux = []
    dic_aux = {}

    for rel in relA:
        top_rel = top_relation(metadatos, rel)
        dic_aux[top_rel] = rel
        relA_aux.append(top_rel)

    for rel in relB:
        relB_aux.append(top_relation(metadatos, rel))

    mismasRelaciones = set(relA_aux) & set(relB_aux)
    porcentajeRelaciones = round(len(mismasRelaciones) / len(relA_aux), 4)

    mismasRelaciones_aux = []
    for rel in mismasRelaciones:
        mismasRelaciones_aux.append(dic_aux[rel])

    return (porcentajeRelaciones, mismasRelaciones_aux)

# Recibe el diccionario de metadatos y el ID de una relación
# y devuelve cuál es la relación top de esa que mantiene el sentido
def top_relation(metadatos, rel):
  for destID, typeID in metadatos[rel]['relations']:
    if typeID == ES_UN_ID:
      if destID in TOP_RELATIONS:
        return rel
      else:
        return top_relation(metadatos, destID)

# Devuelve los n posibles conceptos destino para la analogía entre los vectores A, B y C
# como tuplas donde el primer elemento es el ID y el segundo la similitud del coseno
# La analogía la hace de la forma A es para B, lo que C es para X
# Recibe como entrada el vector de embeddings de los conceptos de entrenamiento, sus IDs
# y los vectores A, B y C
def get_analogias(embs, ids, wordA, wordB, wordC, n=10):
    wordX = wordB - wordA + wordC
    sims_X = cosine_similarity(X=wordX.reshape(1, -1), Y=embs)[0]

    sim_list = list(zip(ids, sims_X))

    sim_list.sort(key=lambda x: x[1], reverse=True)

    return sim_list[:n]

# Obtiene el primer destino de una relación que sea de ese tipo
# Relaciones tiene que ser una lista de tuplas donde el primer
# elemento es el ID del concepto y el segundo el ID de la relación
def getIDdestino(relaciones, relID):
    for destID, typeID in relaciones:
        if relID == typeID:
            return destID

    return -1

# Comprueba los posibles destinos para un tipo de relación
# y devuelve en qué posición de las analogías está el concepto
# destino correcto. Devuelve -1 en caso contrario
def concepto_destino_correcto(analogias, concepto, relID):
    posibles_destinos = []
    for destID, typeID in concepto['relations']:
        if typeID == relID:
            posibles_destinos.append(destID)

    n = 0
    for conceptID, sim in analogias:
        if conceptID in posibles_destinos:
            return n
        n += 1

    return -1

# Devuelve los vecinos de un concepto
# Recibe el diccionario de conceptos y el ID del concepto
def get_vecinos(diccionario, nodo):
    return diccionario[nodo]['vecinos']

# Realiza todos los caminos aleatorios de profundidad depth
# para el concepto nodo_inicial. Recibe como entrada el diccionario
# de conceptos, el ID del concepto y la profundidad
# Devuelve una lista de caminos
def random_walk(diccionario, nodo_inicial, depth=1):
    walks = {(nodo_inicial,)}

    for i in range(depth):
        walks_copy = walks.copy()

        for walk in walks_copy:
            nodo = walk[-1]
            vecinos = get_vecinos(diccionario, nodo)

            if len(vecinos) > 0:
                walks.remove(walk)

            for vecino in vecinos:
                if i == depth - 1:
                    walks.add(walk + (vecino[0], vecino[1]))
                else:
                    walks.add(walk + (vecino[0], vecino[1],))

    return list(walks)

# Obtiene todos los random_walks de profundidad depth
# para los conceptos del diccionario
def random_walks(diccionario, depth=1):
    all_walks = set()
    for conceptID, concept in diccionario.items():

        walks = random_walk(diccionario, conceptID, depth)
        for walk in walks:
            all_walks.add(walk)

    return all_walks