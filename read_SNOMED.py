import sys
import re
import json
import pandas as pd
from copy import deepcopy
from postcoordinate_functions import get_jerarquia

# Import the hierarchy
PATH, INPUT, LOGS, MODELS, DICT, CORPUS, CONCEPTS = get_jerarquia()

# Constants
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
#    OBTAINING INTERNATIONAL CONCEPTS    #
##########################################

# We load the concepts of the international version, which are
# labelled as either active or inactive
concepts_international = pd.read_csv(PATH+INPUT+CONCEPTS_INTERNATIONAL, delimiter='\t')

concepts_status = {}

for CID, active in zip(concepts_international['id'], concepts_international['active']):
    concepts_status[CID] = active

##############################
#   OBTAINING DESCRIPTIONS   #
##############################

# We read the file that contains the descriptions
description_d = pd.read_csv(PATH+INPUT+DESCRIPTION_SNOMED, delimiter='\t')

# Initialize the dictionary
active_concepts = {}

# There are some concepts in the Spanish version that may be labelled as active, but they are actually
# inactive. It's indicated in the name by the words "RETIRADO" or "no activo"
# We will save them in this dictionary
false_active_concepts = {}

# For each line of the file, we keep the ID of the concept if it's active and
# each active description and its type (to identify if it's the FSN or a synonym)
for CID, active, typeID, description in zip(description_d['conceptId'],
                                            description_d['active'],
                                            description_d['typeId'],
                                            description_d['term']):
    # To avoid inactive concepts
    if active == 1:
        # If we already have an entry for the concept, we add the description to the dictionary
        if CID in active_concepts:
            if typeID == FULLY_SPECIFIED_NAME_ID:
                semantic_tag = re.search('\(.+\)', description).group().split(sep='(')[-1]
                semantic_tag = re.sub('[\(\)]', '', semantic_tag)
                active_concepts[CID]['FSN'] = description
                active_concepts[CID]['semantic_tag'] = semantic_tag
            active_concepts[CID]['description'].append(description)
        # Otherwise, we create an entry for it
        else:
            if typeID == FULLY_SPECIFIED_NAME_ID:
                semantic_tag = re.search('\(.+\)', description).group().split(sep='(')[-1]
                semantic_tag = re.sub('[\(\)]', '', semantic_tag)
                active_concepts[CID] = {'FSN' : description, 'description' : [description], 'relations' : [], 'relationsAux' : [], 'definition' : '', 'semantic_tag' : semantic_tag, 'vecinos' : []}
            else:
                active_concepts[CID] = {'FSN' : '', 'description' : [description], 'relations' : [], 'relationsAux' : [], 'definition' : '', 'semantic_tag' : '', 'vecinos' : []}

# As previously mentioned, there are some inactive concept which are mislabelled
# but it is indicated in their FSN
active_concepts_aux = deepcopy(active_concepts)
for CID, concept in active_concepts_aux.items():
    if (CID in concepts_status and concepts_status[CID] == 0) or (" no activo" in concept['FSN'] or "RETIRED" in concept['FSN'] or '[X]' in concept['FSN']):
        false_active_concepts[CID] = active_concepts.pop(CID)

#############################
#   OBTAINING DEFINITIONS   #
#############################

# We read the file that contains the definitions of some concepts
definition_d = pd.read_csv(PATH+INPUT+TEXT_DEFINITION, delimiter='\t')

# For each line, we keep that definition if it's active, as well as to which
# concept is refering
for active, CID, definition in zip(definition_d['active'],
                                   definition_d['conceptId'],
                                   definition_d['term']):
    # Only if the definition is active and the concept is active,
    # we keep said definition
    if active == 1 and CID in active_concepts:
        active_concepts[CID]['definition'] = definition

###########################
# OBTAINING RELATIONSHIPS #
###########################

# We read the file containing the relationships of the international version
relations_d = pd.read_csv(PATH+INPUT+RELATIONS, delimiter='\t')

# For each line, if the relationship is active, we keep its type as well as
# the head and tail concepts involved
for active, sourceID, destinationID, typeID in zip(relations_d['active'],
                                                   relations_d['sourceId'],
                                                   relations_d['destinationId'],
                                                   relations_d['typeId']):
    if active == 1 and sourceID in active_concepts and destinationID in active_concepts:
        active_concepts[sourceID]['relations'].append([destinationID, typeID])
        if typeID == ES_UN_ID:
            active_concepts[destinationID]['relationsAux'].append(sourceID)

        # We also obtain the neighbours
        active_concepts[sourceID]['vecinos'].append([typeID, destinationID])
        # By uncommenting the following line, you can create false symmetry
        # active_concepts[destinationID]['vecinos'].append([typeID, sourceID])

##########################
#   OBTAINING METADATA   #
##########################

metadatos_sin_explorar = [RAIZ_JERARQUIA_METADATOS]
metadatos = {}
active_concepts_no_metadatos = active_concepts.copy()

while len(metadatos_sin_explorar) > 0:
    sourceID = metadatos_sin_explorar.pop(0)

    # Check if the concept is already in the metadata dictionary
    if sourceID not in metadatos:
        for destinationID in active_concepts_no_metadatos[sourceID]['relationsAux']:
            metadatos_sin_explorar.append(destinationID)

        metadatos[sourceID] = active_concepts_no_metadatos.pop(sourceID)


# Saving active concepts and metadata dictionaries
active_concepts_file = open(PATH + CONCEPTS + CONCEPTS_FILE, 'w')
json.dump(active_concepts_no_metadatos, active_concepts_file, indent=4)
active_concepts_file.close()

metadatos_file = open(PATH + CONCEPTS + METADATOS_FILE, 'w')
json.dump(metadatos, metadatos_file, indent=4)
metadatos_file.close()