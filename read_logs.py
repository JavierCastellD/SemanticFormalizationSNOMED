import sys
import json
import pandas as pd
from postcoordinate_functions import get_jerarquia, top_relation, string_keys_to_int, read_list_of_lists

# Import the hierarchy
PATH, INPUT, LOGS, MODELS, DICT, CORPUS, CONCEPTS = get_jerarquia()

# Constants
FULLY_SPECIFIED_NAME_ID = 900000000000003001
SYNONYM_ID = 900000000000013009
ES_UN_ID = 116680003
RAIZ_JERARQUIA_METADATOS = 900000000000441003

# Path to the logs file
LOG = sys.argv[1]

# Path to the concepts file
CONCEPTS_FILE = sys.argv[2]

# Path to the metadata file
METADATOS_FILE = sys.argv[3]

# Number of concepts of evaluation
total_concepts_ev = int(sys.argv[4])

# Concepts where we could not create the
# word embedding
elementos_vacios = 0

## VARIABLES FOR SEMANTIC TAG ACCURACY ##
# Hits @1
n_top1 = 0
# Hits @5
n_top5 = 0
# Hits @10
n_top10 = 0

# Dictionary whose key is the semantic tag
# and which has two elements: hits @1 and misses @1
# This is to identify hits per semantic category
n_cat = {}

clases = {}

# Hits @10 when the word is not complete
n_no_comp = 0
# Misses @10 when the word is not complete
fallos_no_comp = 0

## VARIABLES FOR RELATIONSHIPS ##
# When choosing the top similar concept of the most frequent semantic category
# Mean percentage of relationships
media_rel_igual_1 = 0

# When choosing the top similar concept of the correct semantic category (semantic concept)
# Mean percentage of relationships for sem
media_rel_igual_sem = 0

## VARIABLES FOR ANALOGIES ##
# Mean analogies hit @1, @5 and @10
media_ana_top1 = 0
media_ana_top5 = 0
media_ana_top10 = 0

media_completitud = 0

# Mean analogies hit @1, @5 and @10 for semantic
media_ana_top1_sem = 0
media_ana_top5_sem = 0
media_ana_top10_sem = 0

media_rels_test = 0

# Loading the metadata
metadatos_file = open(PATH + CONCEPTS + METADATOS_FILE)
metadatos = string_keys_to_int(json.load(metadatos_file))
metadatos_file.close()

for i in range(1):
    # Open the log file
    log = pd.read_csv(PATH + LOGS + LOG, sep='\t')

    # We load the concepts
    concepts_file = open(PATH + CONCEPTS + CONCEPTS_FILE)
    concepts = string_keys_to_int(json.load(concepts_file))
    concepts_file.close()

    log['Analogias 1'] = log['Analogias 1'].fillna('[]')
    log['Analogias Sem'] = log['Analogias Sem'].fillna('[]')

    # For each line in the log file, we obtain the ID of the evaluation concept, the semantic tag, if it was hit (top),
    # the number of words that were not in the dictionary, the percentage of words that appear in the dictionary
    # the mean percentage of relationships shared with the top concept or the semantic concept,
    # the relationships shared with the top or semantic concept, whether the analogies were correct for each type of
    # relationship, the list of most similar concepts, the most similar concept of the correct semantic category,
    # and the IDs for the most similar concept and the most similar concept of the correct semantic category
    for id, sem_tag, top, pal_no_cont, porc_cont, porc_igual_1, porc_igual_sem, rel_igual_1, rel_igual_sem, analogias, ids_similares, topRel, analogias_sem, id1, id_match in zip(
            log['ID'], log['Semantic tag'],
            log['Top'], log['Palabras no contenida'],
            log['% palabras contenidas'],
            log['% igual 1'], log['% igual match'],
            log['Nombres igual 1'], log['Nombres igual match'],
            log['Analogias 1'], log['Similares'],
            log['Top Rel'], log['Analogias Sem'],
            log['ID 1'], log['ID match']):

        top = int(top)
        porc_cont = float(porc_cont)

        # This is done to be able to read old logs and is no longer used
        # It checks if the mistake is due to not being able to create the WE
        if top == -2 or porc_cont == 0:
            elementos_vacios += 1
        else:
            # The lists from the log are preprocessed so that they can be used correctly
            rel_igual_1 = read_list_of_lists(rel_igual_1)
            rel_igual_sem = read_list_of_lists(rel_igual_sem)

            analogias = read_list_of_lists(analogias)
            analogias_sem = read_list_of_lists(analogias_sem)

            pal_no_cont = float(pal_no_cont)
            porc_igual_1 = float(porc_igual_1)
            porc_igual_sem = float(porc_igual_sem)

            # To calculate the accuracy for analogies of the most similar
            # concept of the correct semantic type
            if len(analogias_sem) > 0:
                ana_top1_sem = 0
                ana_top5_sem = 0
                ana_top10_sem = 0
                for rel, topAna in analogias_sem:
                    if topAna != -1:
                        ana_top10_sem += 1
                        if topAna == 0:
                            ana_top5_sem += 1
                            ana_top1_sem += 1
                        elif topAna < 5:
                            ana_top5_sem += 1

                media_ana_top1_sem += ana_top1_sem / len(analogias_sem)
                media_ana_top5_sem += ana_top5_sem / len(analogias_sem)
                media_ana_top10_sem += ana_top10_sem / len(analogias_sem)

            # To calculate the accuracy of our method for analogies
            # If the semantic type is correct (top == 0), we use the analogies of analogias_sem
            # Otherwise, we use those of analogias
            if top == 0 and len(analogias_sem) > 0:
                ana_top1 = 0
                ana_top5 = 0
                ana_top10 = 0
                for rel, topAna in analogias_sem:
                    if topAna != -1:
                        ana_top10 += 1
                        if topAna == 0:
                            ana_top5 += 1
                            ana_top1 += 1
                        elif topAna < 5:
                            ana_top5 += 1
                media_ana_top1 += ana_top1 / len(analogias_sem)
                media_ana_top5 += ana_top5 / len(analogias_sem)
                media_ana_top10 += ana_top10 / len(analogias_sem)
            elif len(analogias) > 0:
                ana_top1 = 0
                ana_top5 = 0
                ana_top10 = 0
                for rel, topAna in analogias:
                    if topAna != -1:
                        ana_top10 += 1
                        if topAna == 0:
                            ana_top5 += 1
                            ana_top1 += 1
                        elif topAna < 5:
                            ana_top5 += 1
                media_ana_top1 += ana_top1 / len(analogias)
                media_ana_top5 += ana_top5 / len(analogias)
                media_ana_top10 += ana_top10 / len(analogias)

            ## This and the next one are old methods to
            ## calculate the percentage of relationships and it are
            ## not used in the paper

            # To calculate the percentage of relationships for sem
            media_rel_igual_sem += porc_igual_sem

            # To calculate the percentage of relationships for
            # our method. If we have the semantic type, we use sem
            if top == 0:
                media_rel_igual_1 += porc_igual_sem
            else:
                media_rel_igual_1 += porc_igual_1


            # This is to calculate the accuracy at predicting
            # the semantic type. If top == -1, it is a miss
            if top == -1:
                # This is to count the number of misses when
                # some word was missing from the dictionary
                if pal_no_cont > 0:
                    fallos_no_comp += 1

                # This is to count the number of hits/misses (acierto/fallo)
                # per semantic type. So if the semantic type is not registered
                # in the dictionary, we create the entry adding a miss
                if sem_tag not in n_cat:
                    n_cat[sem_tag] = {'acierto': 0, 'fallo': 1}
                # Otherwise, we add the miss to the corresponding semantic type
                else:
                    n_cat[sem_tag]['fallo'] += 1

            # If top == 0, then it is a hit for semantic type
            # Previously, if top >= 0, it was a hit and it represented
            # if it was hit @1, @5 or @10
            else:
                # We check if the hit was @1, @5 or @10
                n_top10 += 1
                if top == 0:
                    n_top1 += 1
                    n_top5 += 1
                elif top < 5:
                    n_top5 += 1

                # This is to count the number of hits when
                # a word is missing from the dictionary
                if pal_no_cont > 0:
                    n_no_comp += 1

                # This is to count the number of hits/misses (acierto/fallo)
                # per semantic type. So if the semantic type is not registered
                # in the dictionary, we create the entry adding a hit
                if top == 0:
                    if sem_tag not in n_cat:
                        n_cat[sem_tag] = {'acierto': 1, 'fallo': 0}
                    # Otherwise, we add the hit to the corresponding semantic type
                    else:
                        n_cat[sem_tag]['acierto'] += 1
                # This was done this way to consider as miss when the hit was not @1
                # This is no longer used since top can only be 0 or -1 now
                else:
                    if sem_tag not in n_cat:
                        n_cat[sem_tag] = {'acierto': 0, 'fallo': 1}
                    # En caso contrario, apuntamos el fallo para esa clase
                    else:
                        n_cat[sem_tag]['fallo'] += 1

            ## This is to calculate the completeness of the postcoordinated concept
            #                   semantic type hit + number of relationships predicted + number of correct analogies
            # Completeness is: -------------------------------------------------------------------------------------
            #                            1 + 2 * number of relationships of the evaluation concept

            # First we obtain the relationships of the evaluation concept
            relations = concepts[id]['relations']

            # We obtain the maximum value of completeness
            # which is 1 (semantic type) + 2 * number of relationships (because of relationships and analogies)
            total = 1 + 2 * len(relations)

            # Initialize the score to 0
            punt = 0

            if top == 0:
                # If top == 0, it was a hit for semantic type
                # so we add 1 to the score
                punt += 1

                # We extract the predicted relationships and obtain their top relationship value to be able
                # to compare them
                relations_igual_top_sem = [top_relation(metadatos, typeID) for typeID in rel_igual_sem]
                # We also extract the predicted analogies for each predicted relationship
                analogias_top_sem = [[top_relation(metadatos, typeID), top] for typeID, top in analogias_sem]

                # We check each relationship in the evaluation concept
                # If a concept has the same relationship twice, our method can
                # only predict one analogy for that relationship
                for _, typeID in relations:
                    typeID = top_relation(metadatos, typeID)
                    # If that relationship was predicted, we add 1 to the score
                    if typeID in relations_igual_top_sem:
                        punt += 1

                    # If the analogy for that relationship was predicted, we add 1 to the score
                    if [typeID, 0] in analogias_top_sem:
                        punt += 1
            # If it was a miss for semantic type, we repeat the same but using
            # the relationships and analogies for the most similar concept
            else:

                relations_igual_top = [top_relation(metadatos, typeID) for typeID in rel_igual_1]
                analogias_top = [[top_relation(metadatos, typeID), top] for typeID, top in analogias]

                # We check each relationship in the evaluation concept
                for _, typeID in relations:
                    typeID = top_relation(metadatos, typeID)
                    # If that relationship was predicted, we add 1 to the score
                    if typeID in relations_igual_top:
                        punt += 1

                    # If the analogy for that relationship was predicted, we add 1 to the score
                    if [typeID, 0] in analogias_top:
                        punt += 1

            # Then we finally calculate the completeness
            media_completitud += punt / total

            # This is to correctly calculate the percentage of accuracy of relationships
            if top == -1:
                id_aux = id1
            else:
                id_aux = id_match

            # We extract the relationships of the evaluation concept
            rels_eval = [top_relation(metadatos, typeID) for _, typeID in relations]
            n_rels_eval = len(rels_eval)

            # We extract the relationships of the predicted concept
            rels_concept = [top_relation(metadatos, typeID) for _, typeID in concepts[id_aux]['relations']]
            rels_concept_aux = []

            # For each predicted relationship
            for ele in rels_concept:
                # If it is found in the evaluation concept
                if ele in rels_eval:
                    # We added it to the list of correct relationships
                    rels_concept_aux.append(ele)

                    # We remove it from this list to avoid duplicates
                    rels_eval.remove(ele)

            media_rels_test += len(rels_concept_aux) / n_rels_eval

            # This is to have statistics per semantic type
            # If there is already an entry for this type
            if sem_tag in clases:
                clases[sem_tag]['total'] += 1

                if top == 0:
                    clases[sem_tag]['top1'] += 1
                    clases[sem_tag]['%rel'] += porc_igual_sem
                    if len(analogias_sem) > 0:
                        clases[sem_tag]['an1'] += ana_top1_sem / len(analogias_sem)
                else:
                    clases[sem_tag]['%rel'] += porc_igual_1
                    if len(analogias) > 0:
                        clases[sem_tag]['an1'] += ana_top1 / len(analogias)

                clases[sem_tag]['complet'] += punt / total
                clases[sem_tag]['%rel real'] += len(rels_concept_aux) / len(n_rels_eval)
            # Otherwise, we create an entry for this semantic type
            else:
                clases[sem_tag] = {'top1': 0, '%rel': 0, 'an1': 0, 'complet': 0, 'total': 0, '%rel real': 0}

                clases[sem_tag]['total'] += 1
                if top == 0:
                    clases[sem_tag]['top1'] += 1
                    clases[sem_tag]['%rel'] += porc_igual_sem
                    if len(analogias_sem) > 0:
                        clases[sem_tag]['an1'] += ana_top1_sem / len(analogias_sem)
                else:
                    clases[sem_tag]['%rel'] += porc_igual_1
                    if len(analogias) > 0:
                        clases[sem_tag]['an1'] += ana_top1 / len(analogias)

                clases[sem_tag]['complet'] += punt / total
                clases[sem_tag]['%rel real'] += len(rels_concept_aux) / len(n_rels_eval)


# This just prints the information from the log that we have previously obtained
print('% Hit @10:', round(n_top10 / total_concepts_ev * 100, 2))
print('% Hit @5:', round(n_top5 / total_concepts_ev * 100, 2))
print('% Hit @1:', round(n_top1 / total_concepts_ev * 100, 2))
print('--------------------------------------')
print('Percentage of correctly predicted relationships:', round(media_rels_test / total_concepts_ev * 100, 2))

print('--------------------------------------')
print('% Hit @1 (only completed WE):', round(n_top1 / (total_concepts_ev - elementos_vacios) * 100, 2))
print('Mean percentages of analogies @1 (only completed WE):', round(media_ana_top1 / (total_concepts_ev - elementos_vacios) * 100, 2))
print('Mean completeness (only completed WE):', round(media_completitud / (total_concepts_ev - elementos_vacios) * 100, 2))

total_rel_igual = n_top10

''' This is an old version of calculating relationships
print('--------------------------------------')
print('Media relaciones igual 1:', round(media_rel_igual_1 / total_concepts_ev * 100, 2))
print('Media relaciones igual sem:', round(media_rel_igual_sem / total_rel_igual * 100, 2))
'''

print('--------------------------------------')
print('Mean analogies @10:', round(media_ana_top10 / total_concepts_ev * 100, 2))
print('Mean analogies @5:', round(media_ana_top5 / total_concepts_ev * 100, 2))
print('Mean analogies @1:', round(media_ana_top1 / total_concepts_ev * 100, 2))
print('Mean completeness:', round(media_completitud / total_concepts_ev * 100, 2))

print('--------------------------------------')
print('Mean analogies sem @10:', round(media_ana_top10_sem / total_concepts_ev * 100, 2))
print('Mean analogies sem @5:', round(media_ana_top5_sem / total_concepts_ev * 100, 2))
print('Mean analogies sem @1:', round(media_ana_top1_sem / total_concepts_ev * 100, 2))

for key, value in clases.items():
    print('--------------------------------------')
    print(key, 'Total:', value['total'], '(' + str(round(value['total'] / total_concepts_ev * 100, 2)) + '%)')
    print(key, '% Hit @1:', round(value['top1'] / value['total'] * 100, 2))
    #print(key, '% rel padre:', round(value['%rel'] / value['total'] * 100, 2))
    print(key, '% Analogíes @1:', round(value['an1'] / value['total'] * 100, 2))
    print(key, '% Completeness:', round(value['complet'] / value['total'] * 100, 2))
    print(key, '% Relationships:', round(value['%rel real'] / value['total'] * 100, 2))