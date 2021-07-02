import json
import random
from postcoordinate_functions import string_keys_to_int, remove_duplicates, random_walks

PATH = './'
INPUT = 'input/'
LOGS = 'logs/'
MODELS = 'models/'
DICT = 'dicts/'
CORPUS = 'corpus/'
CONCEPTS = 'concepts/'

ES_UN_ID = 116680003

# Conceptos de entrenamiento, validacin, conceptos activos y metadatos
CONCEPTS_FILE = 'active_concepts.json'
METADATOS_FILE = 'metadatos.json'
CONCEPTS_FILE_INTERNATIONAL = 'active_concepts_international.json'
METADATOS_FILE_INTERNATIONAL = 'metadatos_international.json'

for i in range(1):
  print('Estamos por la prueba:', i)
  #############################################
  # ACTUALIZAMOS LOS NOMBRES DE LAS VARIABLES #
  #############################################
  # Identificador de prueba
  id_prueba = '_test' + str(i) 

  # Corpus de texto
  CORPUS_RELATIONS_FILE = 'corpus_rel_' + id_prueba + '.txt'

  # Cargamos los conceptos activos
  active_concepts_file = open(PATH + CONCEPTS + CONCEPTS_FILE)
  active_concepts = string_keys_to_int(json.load(active_concepts_file))
  active_concepts_file.close()

  # Cargamos los metadatos
  metadatos_file = open(PATH + CONCEPTS + METADATOS_FILE)
  metadatos = string_keys_to_int(json.load(metadatos_file))
  metadatos_file.close()

  #######################
  # GENERAMOS El CORPUS #
  #######################
  print('Generando el corpus')
  # Ponemos los conceptos de entrenamiento
  concepts_training = active_concepts

  # Hacemos los random walks
  rws_URI = random_walks(concepts_training, depth=2)
  rws_words = random_walks(concepts_training, depth=2)

  ### CORPUS DE RELACIONES ###
  with open(PATH + CORPUS + CORPUS_RELATIONS_FILE, 'w') as corpus_file:
    # Hacemos primero las líneas del corpus de URIs/IDs
    for rwU in rws_URI:
      # Creamos las lineas del corpus de URIs
      line = ''

      if len(rwU) > 1:
        for w in rwU:
            line += str(w) + ' '
      
        corpus_file.write(line + '\n')

    # Después hacemos las líneas del corpus de palabras
    for rwW in rws_words:
      if rwW is not None:
        if len(rwW) == 5:
          # Obtenemos los nombres involucrados que no estén repetidos
          sources = remove_duplicates(concepts_training[rwW[0]]['description'])
          destinations = remove_duplicates(concepts_training[rwW[2]]['description'])
          destinations2 = remove_duplicates(concepts_training[rwW[4]]['description'])

          for source in sources:
            for dest in destinations:
              for dest2 in destinations2:
                lineLabeled = source + ' ' + str(rwW[1]) + ' ' + dest + ' ' + str(rwW[3]) + ' ' + dest2

                corpus_file.write(lineLabeled + ' \n')
          
        elif len(rwW) == 3:
          # Obtenemos los nombres involucrados que no estén repetidos
          sources = remove_duplicates(concepts_training[rwW[0]]['description'])
          destinations = remove_duplicates(concepts_training[rwW[2]]['description'])

          for source in sources:
            for dest in destinations:
              lineLabeled = source + ' ' + str(rwW[1]) + ' ' + dest

              corpus_file.write(lineLabeled + ' \n')
        elif len(rwW) == 1:
          # Obtenemos los nombres involucrados que no estén repetidos
          sources = remove_duplicates(concepts_training[rwW[0]]['description'])

          for source in sources:
            corpus_file.write(source + ' \n')
          
    # Finalmente hacemos las líneas del corpus combinado   
    for rwW in rws_words:
      if rwW is not None and len(rwW) > 1:
        if len(rwW) == 3:
          # Escogemos qué elemento va a mantenerse como ID que no sea la relación
          URI_n = random.randint(0, len(rwW) - 2)
          if URI_n == 1:
            URI_n == 2

          lineMix = ''

          for i, w in enumerate(rwW):
            if i != URI_n:
              if w in concepts_training:
                lineMix += concepts_training[w]['FSN'] + ' '
              else:
                # Esto es para las relaciones con URI
                lineMix += str(w) + ' '
            else:
              lineMix += str(w) + ' '

          corpus_file.write(lineMix + '\n')
        elif len(rwW) == 5:
          # Escogemos qué elemento va a mantenerse como ID que no sea la relación
          URI_n = random.randint(0, len(rwW) - 3)
          if URI_n == 1:
            URI_n == 4

          lineMix = ''

          for i, w in enumerate(rwW):
            if i != URI_n:
              if w in concepts_training:
                lineMix += concepts_training[w]['FSN'] + ' '
              else:
                # Esto es para las relaciones con URI
                lineMix += str(w) + ' '
            else:
              lineMix += str(w) + ' '

          corpus_file.write(lineMix + '\n')