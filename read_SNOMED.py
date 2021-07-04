import sys
import re
import json
import pandas as pd
from copy import deepcopy
from postcoordinate_functions import get_jerarquia

PATH, INPUT, LOGS, MODELS, DICT, CORPUS, CONCEPTS = get_jerarquia()

# Constantes
FULLY_SPECIFIED_NAME_ID = 900000000000003001
SYNONYM_ID = 900000000000013009
ES_UN_ID = 116680003
RAIZ_JERARQUIA_METADATOS = 900000000000441003

CONCEPTS_FILE = 'active_concepts.json'
METADATOS_FILE = 'metadatos.json'

CONCEPTS_INTERNATIONAL = sys.argv[1]
DESCRIPTION_SNOMED = sys.argv[2]
TEXT_DEFINITION = sys.argv[3]
RELATIONS = sys.argv[4]

##########################################
# OBTENCIÓN DE CONCEPTOS INTERNACIONALES #
##########################################

# Cargamos los conceptos de la versión internacional, que están
# correctamente etiquetados como activos e inactivos y nos lo guardamos
concepts_international = pd.read_csv(PATH+INPUT+CONCEPTS_INTERNATIONAL, delimiter='\t')

concepts_status = {}

for CID, active in zip(concepts_international['id'], concepts_international['active']):
  concepts_status[CID] = active

##############################
# OBTENCIÓN DE DESCRIPCIONES #
##############################

# Leemos el fichero que contiene las descripciones
description_d = pd.read_csv(PATH+INPUT+DESCRIPTION_SNOMED, delimiter='\t')

# Inicializamos el diccionario
active_concepts = {}

# Hay conceptos marcados como activos que no están activos -> Indicado por "RETIRADO" o "no activo"
# Los guardamos para revisarlos
false_active_concepts = {}

# Por cada línea el fichero nos quedamos el identificador del concepto,
# si la descripción está activa, el tipo de descripción (por si es sinónimo o FSN),
# y la descripción a la que corresponde
for CID, active, typeID, description in zip(description_d['conceptId'],
                                         description_d['active'],
                                         description_d['typeId'],
                                         description_d['term']):
  # Para quitarnos los conceptos inactivos
  if active == 1:
    # Si ya tenemos el CID, añadimos la descripción al diccionario
    if CID in active_concepts:
      if typeID == FULLY_SPECIFIED_NAME_ID:
        semantic_tag = re.search('\(.+\)', description).group().split(sep='(')[-1]
        semantic_tag = re.sub('[\(\)]', '', semantic_tag)
        active_concepts[CID]['FSN'] = description
        active_concepts[CID]['semantic_tag'] = semantic_tag
      active_concepts[CID]['description'].append(description)
    # Si no, creamos la entrada en el diccionario
    else:
      if typeID == FULLY_SPECIFIED_NAME_ID:
        semantic_tag = re.search('\(.+\)', description).group().split(sep='(')[-1]
        semantic_tag = re.sub('[\(\)]', '', semantic_tag)
        active_concepts[CID] = {'FSN' : description, 'description' : [description], 'relations' : [], 'relationsAux' : [], 'definition' : '', 'semantic_tag' : semantic_tag, 'vecinos' : []}
      else:
        active_concepts[CID] = {'FSN' : '', 'description' : [description], 'relations' : [], 'relationsAux' : [], 'definition' : '', 'semantic_tag' : '', 'vecinos' : []}

# Hay conceptos inactivos que no están marcados como tal, pero que tienen un FSN
# que lo indica o que en la lista de conceptos activos internacional, no están
active_concepts_aux = deepcopy(active_concepts)
for CID, concept in active_concepts_aux.items():
  if (CID in concepts_status and concepts_status[CID] == 0) or (" no activo" in concept['FSN'] or "RETIRED" in concept['FSN'] or '[X]' in concept['FSN']):
    false_active_concepts[CID] = active_concepts.pop(CID)

#############################
# OBTENCIÓN DE DEFINICIONES #
#############################

# Leemos el fichero que contiene las definiciones de algunos conceptos
definition_d = pd.read_csv(PATH+INPUT+TEXT_DEFINITION, delimiter='\t')

# Por cada línea nos quedamos con si esa definición está activa, a qué concepto
# se refiere y la definición en sí
for active, CID, definition in zip(definition_d['active'],
                                  definition_d['conceptId'],
                                  definition_d['term']):
  # Si la definición está activa y el concepto al que hace referencia lo tenemos
  # entonces nos guardamos dicha definición
  if active == 1 and CID in active_concepts:
    active_concepts[CID]['definition'] = definition

###########################
# OBTENCIÓN DE RELACIONES #
###########################

# Leemos el fichero que contiene las relaciones de la versión internacional
relations_d = pd.read_csv(PATH+INPUT+RELATIONS, delimiter='\t')

# Por cada línea nos quedamos con si esa relación está activa, entre qué conceptos
# ocurre y el tipo de relación que es
for active, sourceID, destinationID, typeID in zip(relations_d['active'],
                                                   relations_d['sourceId'],
                                                   relations_d['destinationId'],
                                                   relations_d['typeId']):
    if active == 1 and sourceID in active_concepts and destinationID in active_concepts:
        active_concepts[sourceID]['relations'].append([destinationID, typeID])
        if typeID == ES_UN_ID:
            active_concepts[destinationID]['relationsAux'].append(sourceID)

        # Metemos los vecinos
        active_concepts[sourceID]['vecinos'].append([typeID, destinationID])
        # active_concepts[destinationID]['vecinos'].append([typeID, sourceID])

##########################
# OBTENCIÓN DE METADATOS #
##########################

metadatos_sin_explorar = [RAIZ_JERARQUIA_METADATOS]
metadatos = {}
active_concepts_no_metadatos = active_concepts.copy()

while len(metadatos_sin_explorar) > 0:
  sourceID = metadatos_sin_explorar.pop(0)

  # Comprobamos que ese concepto no esté ya en metadatos
  if sourceID not in metadatos:
    for destinationID in active_concepts_no_metadatos[sourceID]['relationsAux']:
      metadatos_sin_explorar.append(destinationID)

    metadatos[sourceID] = active_concepts_no_metadatos.pop(sourceID)


active_concepts_file = open(PATH + CONCEPTS + CONCEPTS_FILE, 'w')
json.dump(active_concepts_no_metadatos, active_concepts_file, indent=4)
active_concepts_file.close()

metadatos_file = open(PATH + CONCEPTS + METADATOS_FILE, 'w')
json.dump(metadatos, metadatos_file, indent=4)
metadatos_file.close()