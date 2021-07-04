import sys
import json
import pandas as pd
from postcoordinate_functions import get_jerarquia, top_relation, string_keys_to_int, read_list_of_lists

PATH, INPUT, LOGS, MODELS, DICT, CORPUS, CONCEPTS = get_jerarquia()

# Constantes
FULLY_SPECIFIED_NAME_ID = 900000000000003001
SYNONYM_ID = 900000000000013009
ES_UN_ID = 116680003
RAIZ_JERARQUIA_METADATOS = 900000000000441003

LOG = sys.argv[1]
CONCEPTS_FILE = sys.argv[2]
METADATOS_FILE = sys.argv[3]

# Número de conceptos de evaluación
total_concepts_ev = int(sys.argv[4])

# Conceptos para los que no se ha podido crear un WE
elementos_vacios = 0

## VARIABLES ACIERTO CATEGORÍA SEMÁNTICA ##
# Número de aciertos @1
n_top1 = 0
# Número de aciertos @5
n_top5 = 0
# Número de aciertos @10
n_top10 = 0

# Diccionario cuya clave es la categoría semántica
# y tiene dos elementos: aciertos @1 y fallos @1
n_cat = {}

clases = {}

# Número de aciertos @10 cuando la palabra no está completa
n_no_comp = 0
# Número de fallos @10 cuando la palabra no está completa
fallos_no_comp = 0

## VARIABLES RELACIONES ##
# PARA IGUAL 1
# Porcentaje medio de relaciones acertadas para igual 1
media_rel_igual_1 = 0

# PARA SEM
# Porcentaje medio de relaciones acertadas para igual 1
media_rel_igual_sem = 0

## VARIABLES ANALOGIAS ##
# Media de aciertos de analogias @1, @5 y @10
media_ana_top1 = 0
media_ana_top5 = 0
media_ana_top10 = 0

media_completitud = 0

# Analogías semántica
media_ana_top1_sem = 0
media_ana_top5_sem = 0
media_ana_top10_sem = 0

media_rels_test = 0

# Cargamos los metadatos
metadatos_file = open(PATH + CONCEPTS + METADATOS_FILE)
metadatos = string_keys_to_int(json.load(metadatos_file))
metadatos_file.close()

for i in range(1):

  log = pd.read_csv(PATH + LOGS + LOG, sep='\t')

  # Cargamos los conceptos
  concepts_file = open(PATH + CONCEPTS + CONCEPTS_FILE)
  concepts = string_keys_to_int(json.load(concepts_file))
  concepts_file.close()

  log['Analogias 1'] = log['Analogias 1'].fillna('[]')
  log['Analogias Sem'] = log['Analogias Sem'].fillna('[]')

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

    # Si se trata de un fallo porque no se ha podido generar el vector
    if top == -2 or porc_cont == 0:
      elementos_vacios += 1
    else:
      # Preprocesamos las listas de relaciones para que las lea correctamente
      rel_igual_1 = read_list_of_lists(rel_igual_1)
      rel_igual_sem = read_list_of_lists(rel_igual_sem)

      analogias = read_list_of_lists(analogias)
      analogias_sem = read_list_of_lists(analogias_sem)

      pal_no_cont = float(pal_no_cont)
      porc_igual_1 = float(porc_igual_1)
      porc_igual_sem = float(porc_igual_sem)

      #######################
      ## PARA VECTOR WORDS ##
      #######################

      # Para las analogías independientemente de fallo o no, cogemos las semánticas
      # En caso de que topRel == -1, analogías_sem == analogias
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

      # Para las analogías que utilizamos para completitud, si top == 0
      # Entonces tenemos la categoría semántica. En caso contrario cogemos analogias
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

      # Para las relaciones, independientemente del fallo o no, cogemos las semánticas
      media_rel_igual_sem += porc_igual_sem

      if top == 0:
        media_rel_igual_1 += porc_igual_sem
      else:
        media_rel_igual_1 += porc_igual_1

      if top == -1:
        # Si la palabra no está completa
        if pal_no_cont > 0:
          fallos_no_comp += 1

        # Si la clase semántica no está registrada, la creamos
        # añadiendo el fallo
        if sem_tag not in n_cat:
          n_cat[sem_tag] = {'acierto': 0, 'fallo': 1}
        # En caso contrario, apuntamos el fallo para esa clase
        else:
          n_cat[sem_tag]['fallo'] += 1

      # En caso contrario, se trata de un acierto
      else:
        # Comprobamos si el acierto se ha dado @1, @5 o @10
        n_top10 += 1
        if top == 0:
          n_top1 += 1
          n_top5 += 1
        elif top < 5:
          n_top5 += 1

        # Comprobamos si el concepto estaba completo
        if pal_no_cont > 0:
          n_no_comp += 1

        # Si la clase semántica no está registrada, la creamos
        # añadiendo el acierto si es top == 0
        if top == 0:
          if sem_tag not in n_cat:
            n_cat[sem_tag] = {'acierto': 1, 'fallo': 0}
          # En caso contrario, apuntamos el fallo para esa clase
          else:
            n_cat[sem_tag]['acierto'] += 1
        else:
          if sem_tag not in n_cat:
            n_cat[sem_tag] = {'acierto': 0, 'fallo': 1}
          # En caso contrario, apuntamos el fallo para esa clase
          else:
            n_cat[sem_tag]['fallo'] += 1

      # Media completitud
      relations = concepts[id]['relations']

      relations_igual_top = [top_relation(metadatos, typeID) for typeID in rel_igual_1]
      analogias_top = [[top_relation(metadatos, typeID), top] for typeID, top in analogias]

      total = 1 + 2 * len(relations)
      punt = 0

      if top == 0:
        punt += 1

        relations_igual_top_sem = [top_relation(metadatos, typeID) for typeID in rel_igual_sem]
        analogias_top_sem = [[top_relation(metadatos, typeID), top] for typeID, top in analogias_sem]

        for _, typeID in relations:
          typeID = top_relation(metadatos, typeID)
          if typeID in relations_igual_top_sem:
            punt += 1

          if [typeID, 0] in analogias_top_sem:
            punt += 1
      else:
        for _, typeID in relations:
          typeID = top_relation(metadatos, typeID)
          if typeID in relations_igual_top:
            punt += 1

          if [typeID, 0] in analogias_top:
            punt += 1

      media_completitud += punt / total

      # Para la medida sobre cuántas relaciones son correctas del hijo
      if top == -1:
        id_aux = id1
      else:
        id_aux = id_match

      rels_hijo = [top_relation(metadatos, typeID) for _, typeID in relations]
      rels_hijo_aux = [top_relation(metadatos, typeID) for _, typeID in relations]

      rels_padre = [top_relation(metadatos, typeID) for _, typeID in concepts[id_aux]['relations']]
      rels_padre_aux = []

      for ele in rels_padre:
        if ele in rels_hijo_aux:
          rels_padre_aux.append(ele)
          rels_hijo_aux.remove(ele)

      media_rels_test += len(rels_padre_aux) / len(rels_hijo)

      # Para tener estadísticas por clase semántica
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
        clases[sem_tag]['%rel real'] += len(rels_padre_aux) / len(rels_hijo)
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
        clases[sem_tag]['%rel real'] += len(rels_padre_aux) / len(rels_hijo)

print('############')
print('### Word ###')
print('############')
print('% acierto @10:', round(n_top10 / total_concepts_ev * 100, 2))
print('% acierto @5:', round(n_top5 / total_concepts_ev * 100, 2))
print('% acierto @1:', round(n_top1 / total_concepts_ev * 100, 2))
print('--------------------------------------')
print('Porcentaje de relaciones del hijo acertadas:', round(media_rels_test / total_concepts_ev * 100, 2))

print('--------------------------------------')
print('% acierto @1 (sin vacíos):', round(n_top1 / (total_concepts_ev - elementos_vacios) * 100, 2))
print('Media analogías @1 (sin vacíos):', round(media_ana_top1 / (total_concepts_ev - elementos_vacios) * 100, 2))
print('Media completitud (sin vacíos):', round(media_completitud / (total_concepts_ev - elementos_vacios) * 100, 2))

total_rel_igual = n_top10

print('--------------------------------------')
print('Media relaciones igual 1:', round(media_rel_igual_1 / total_concepts_ev * 100, 2))
print('Media relaciones igual sem:', round(media_rel_igual_sem / total_rel_igual * 100, 2))

print('--------------------------------------')
print('Media analogías @10:', round(media_ana_top10 / total_concepts_ev * 100, 2))
print('Media analogías @5:', round(media_ana_top5 / total_concepts_ev * 100, 2))
print('Media analogías @1:', round(media_ana_top1 / total_concepts_ev * 100, 2))
print('Media completitud:', round(media_completitud / total_concepts_ev * 100, 2))

print('--------------------------------------')
print('Media analogías sem @10:', round(media_ana_top10_sem / total_concepts_ev * 100, 2))
print('Media analogías sem @5:', round(media_ana_top5_sem / total_concepts_ev * 100, 2))
print('Media analogías sem @1:', round(media_ana_top1_sem / total_concepts_ev * 100, 2))

for key, value in clases.items():
  print('--------------------------------------')
  print(key, 'Total:', value['total'], '(' + str(round(value['total'] / total_concepts_ev * 100, 2)) + '%)')
  print(key, '% acierto @1:', round(value['top1'] / value['total'] * 100, 2))
  print(key, '% rel padre:', round(value['%rel'] / value['total'] * 100, 2))
  print(key, '% analogías @1:', round(value['an1'] / value['total'] * 100, 2))
  print(key, '% completitud:', round(value['complet'] / value['total'] * 100, 2))
  print(key, '% rel real:', round(value['%rel real'] / value['total'] * 100, 2))