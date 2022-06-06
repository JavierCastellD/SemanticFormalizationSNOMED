import sys
import json
import random
from postcoordinate_functions import string_keys_to_int, remove_duplicates, random_walks, get_jerarquia

PATH, INPUT, LOGS, MODELS, DICT, CORPUS, CONCEPTS = get_jerarquia()

ES_UN_ID = 116680003

# Path to the concepts file
concepts_path = sys.argv[1]

# Path to the metadata file
metadatos_path = sys.argv[2]

# Depth of URI random walks
uri_depth = sys.argv[3]

# Depth of word random walks
word_depth = sys.argv[4]

for i in range(1):
    print('Test', i)
    ###############################
    # UPDATING RELEVANT VARIABLES #
    ###############################
    # Test ID
    id_prueba = '_test' + str(i)

    # File for the text corpus
    CORPUS_RELATIONS_FILE = 'corpus_rel_' + id_prueba + '.txt'

    # Loading the concepts file
    active_concepts_file = open(PATH + CONCEPTS + concepts_path)
    active_concepts = string_keys_to_int(json.load(active_concepts_file))
    active_concepts_file.close()

    # Load the metadata file
    metadatos_file = open(PATH + CONCEPTS + metadatos_path)
    metadatos = string_keys_to_int(json.load(metadatos_file))
    metadatos_file.close()

    #######################
    # GENERATE THE CORPUS #
    #######################
    print('Generating the corpus')

    # We choose the training concepts
    concepts_training = active_concepts

    # We perform the random walks for each type of corpora
    rws_URI = random_walks(concepts_training, depth=uri_depth)
    rws_words = random_walks(concepts_training, depth=word_depth)

    ### RELATIONSHIP CORPUS ###
    with open(PATH + CORPUS + CORPUS_RELATIONS_FILE, 'w') as corpus_file:
        # We first obtain the sentences of only IDs/URIs
        for rwU in rws_URI:
            line = ''

            if len(rwU) > 1:
                for w in rwU:
                    line += str(w) + ' '

                corpus_file.write(line + '\n')

        # We then obtain the sentence of only words
        # Relationships are represented by their IDs
        for rwW in rws_words:
            if rwW is not None:
                if len(rwW) == 5:
                    # Obtaining the names involved in the sentence and remove the duplicates between their synonyms
                    sources = remove_duplicates(concepts_training[rwW[0]]['description'])
                    destinations = remove_duplicates(concepts_training[rwW[2]]['description'])
                    destinations2 = remove_duplicates(concepts_training[rwW[4]]['description'])

                    for source in sources:
                        for dest in destinations:
                            for dest2 in destinations2:
                                lineLabeled = source + ' ' + str(rwW[1]) + ' ' + dest + ' ' + str(rwW[3]) + ' ' + dest2

                                corpus_file.write(lineLabeled + ' \n')

                elif len(rwW) == 3:
                    # Obtaining the names involved in the sentence and remove the duplicates between their synonyms
                    sources = remove_duplicates(concepts_training[rwW[0]]['description'])
                    destinations = remove_duplicates(concepts_training[rwW[2]]['description'])

                    for source in sources:
                        for dest in destinations:
                            lineLabeled = source + ' ' + str(rwW[1]) + ' ' + dest

                            corpus_file.write(lineLabeled + ' \n')
                elif len(rwW) == 1:
                    # Obtaining the names involved in the sentence and remove the duplicates between their synonyms
                    sources = remove_duplicates(concepts_training[rwW[0]]['description'])

                    for source in sources:
                        corpus_file.write(source + ' \n')

        # Finally, we obtain the lines of combining IDs and words
        # Relationships are represented by their IDs
        for rwW in rws_words:
            if rwW is not None and len(rwW) > 1:
                if len(rwW) == 3:
                    # We choose what element is going to appear as the ID
                    # It can not be the relationship, since that will always be an ID
                    URI_n = random.randint(0, len(rwW) - 2)
                    if URI_n == 1:
                        URI_n = 2

                    lineMix = ''

                    for i, w in enumerate(rwW):
                        if i != URI_n:
                            if w in concepts_training:
                                lineMix += concepts_training[w]['FSN'] + ' '
                            else:
                                # This is for the relationships represented as an ID
                                lineMix += str(w) + ' '
                        else:
                            lineMix += str(w) + ' '

                    corpus_file.write(lineMix + '\n')
                elif len(rwW) == 5:
                    # We choose what element is going to appear as the ID
                    # It can not be the relationship, since that will always be an ID
                    URI_n = random.randint(0, len(rwW) - 3)
                    if URI_n == 1:
                        URI_n = 4

                    lineMix = ''

                    for i, w in enumerate(rwW):
                        if i != URI_n:
                            if w in concepts_training:
                                lineMix += concepts_training[w]['FSN'] + ' '
                            else:
                                # This is for the relationships represented as an ID
                                lineMix += str(w) + ' '
                        else:
                            lineMix += str(w) + ' '

                    corpus_file.write(lineMix + '\n')