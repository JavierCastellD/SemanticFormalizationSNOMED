from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
import numpy as np
import torch

ES_UN_ID = 116680003 
TOP_RELATIONS = [762705008, 246061005, 410663007, 408739003]
TOP_CONCEPT = 138875005

# This functions searches if a concept ID
# of two concept dictionaries can be found on a string
def id_in_line(line, concepts, concepts_2):
      # Searching for a concept ID of the first concepts dictionary
      for c in concepts.keys():
        if str(c) in line:
          return True

      # Searching for a concept ID of the second concepts dictionary
      for c in concepts_2.keys():
        if str(c) in line:
          return True
          
      return False

# Given tuples of relationships (tail concept ID, type of relationship ID),
# returns the tail concept ID of a certain relationship ID
def getIDdestinoTopRel(relaciones, relID, metadatos):
  # For each tuple in relationships
  for destID, typeID in relaciones:
    # If the relationship ID is the same as relID
    if relID == top_relation(metadatos, typeID):
      # Return the tail concept ID
      return destID
  
  return -1


def get_analogias_bert(embs, ids, wordA, wordB, wordC):
    wordX = np.array(wordB) - np.array(wordA) + np.array(wordC)
    sims_X = cosine_similarity(X=wordX.reshape(1, -1), Y=embs)[0]

    sim_list = list(zip(ids, sims_X))

    sim_list.sort(key=lambda x: x[1], reverse=True)

    analogias = sim_list[:100]

    return analogias


def get_vector_bert(name, model, tokenizer):
    marked_name = "[CLS]" + name + "[SEP]"

    # La tokenizamos usando el tokenizer
    tokenized_name = tokenizer.tokenize(marked_name)

    # Indexamos esos tokens en el vocabulario del tokenizador
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_name)

    # Esto se debe a que BERT est� entrenado para pares de frases distinguiendo entre
    # ambas porque una tiene IDS de todo 1s y otra de todo 0s
    # Al solo entrenar una, solo necesitamos 1s
    segments_ids = [1] * len(tokenized_name)

    # Convertimos la entrada a tensores de Pytorch
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Lo pasamos por el modelo
    outputs = model(tokens_tensor, segments_tensors)
    hidden_states = outputs[2]

    # Obtenemos los vectores de las palabras
    token_vecs = hidden_states[-2][0]

    # Mediante su media obtenemos el WE
    sentence_embedding = torch.mean(token_vecs, dim=0)

    return np.array(sentence_embedding.tolist())


def get_jerarquia():
    path = './'
    inputs = 'input/'
    logs = 'logs/'
    models = 'models/'
    dicts = 'dicts/'
    corpus = 'corpus/'
    concepts = 'concepts/'

    return path, inputs, logs, models, dicts, corpus, concepts


def is_relative(conceptA, conceptB, concepts):
    # Si el concepto es el mismo, devolvemos verdadero
    if conceptA == conceptB:
        return True
    # Si llegamos al concepto superior, entonces no son padre/hijo
    if conceptA == TOP_CONCEPT:
        return False

    # Obtenemos los padres
    rels_es_un = [destID for destID, typeID in concepts[conceptA]['relations'] if typeID == ES_UN_ID]

    # Por cada padre, hacemos una llamada recursiva
    for destRel in rels_es_un:
        # Si alguno de los padres es padre/hijo, devolvemos verdadero
        if is_relative(destRel, conceptB, concepts):
            return True

    # Devolvemos falso en caso contrario
    return False

def is_relative_multiples(conceptA, conceptsB, concepts):
    # Si el concepto es el mismo, devolvemos verdadero
    if conceptA in conceptsB:
        return True
    # Si llegamos al concepto superior, entonces no son padre/hijo
    if conceptA == TOP_CONCEPT:
        return False

    # Obtenemos los padres
    rels_es_un = [destID for destID, typeID in concepts[conceptA]['relations'] if typeID == ES_UN_ID]

    # Por cada padre, hacemos una llamada recursiva
    for destRel in rels_es_un:
        # Si alguno de los padres coincide, devolvemos verdadero
        if is_relative_multiples(destRel, conceptsB, concepts):
            return True

    # Devolvemos falso en caso contrario
    return False

# Aquí la idea es que en función del tipo origen y de la relación, devolvemos
# los posibles tipos válidos (habría que buscar luego si es descendiente)
# Origen es el concepto de referencia, ya que asumimos que va a ser del mismo tipo
def valid_dest_relations(origen, rel, concepts, metadatos):
    # TODO: Habría que ver si tomamos las relaciones top o las hijas
    #   (ya que pueden tener distintos conceptos destino -> solo un caso)
    top_level_rel = top_relation(metadatos, rel)

    ## Basic dose form: 736478001
    if is_relative(origen, 736478001, concepts):
        # Is a: 116680003
        if top_level_rel == ES_UN_ID:
            # Basic dose form: 736478001
            return [736478001]
        # Has state of matter: 736518005
        if top_level_rel == 736518005:
            # State of matter (736471007)
            return [736471007]
    ## Body Structure: 123037004
    if is_relative(origen, 123037004, concepts):
        # Is a: 116680003
        if top_level_rel == ES_UN_ID:
            # Body Structure: 123037004
            return [123037004]
        # All or part of: 733928003 - TODO: Aquí tenemos subtipos de relación, pero el destino no cambia
        if top_level_rel == 733928003:
            # Body Structure (123037004)
            return [123037004]
        # Lateral half of:
        if top_level_rel == 733933004:
            # Body Structure (123037004)
            return [123037004]
        # Laterality: 272741003 (Es solo para un subtipo de body structure)
        if top_level_rel == 272741003:
            # Side (182353008)
            return [182353008]
    ## Clinical finding: 404684003
    if is_relative(origen, 404684003, concepts):
        # Is a: 116680003
        if top_level_rel == ES_UN_ID:
            # Clinical finding: 404684003
            return [404684003]
        # Associated morphology: 116676008
        if top_level_rel == 116676008:
            # Morphologically abnormal structure (49755003)
            return [49755003]
        # Associated with: 47429007 - TODO: Aquí tenemos 3 subtipos de la relación donde va cambiando el destino
        if top_level_rel == 47429007:
            # Physical object (260787004), Event (272379006), Clinical finding (404684003)
            # Organism (410607006), Procedure (71388002), Physical force (78621006)
            return [260787004, 272379006, 404684003, 410607006, 71388002, 78621006]
        # Clinical course: 263502005
        if top_level_rel == 263502005:
            # Courses (288524001)
            return [288524001]
        # Episodicity: 246456000
        if top_level_rel == 246456000:
            # Episodicities (288526004)
            return [288526004]
        # Finding informer: 419066007
        if top_level_rel == 419066007:
            # Subject of record or other provider of history (419358007), Performer of method (420158005)
            # Person with characteristic related to subject of record (444018008)
            return [419358007, 420158005, 444018008]
        # Finding method: 418775008
        if top_level_rel == 418775008:
            # Procedure (71388002)
            return [71388002]
        # Finding site: 363698007
        if top_level_rel == 363698007:
            # Anatomical or acquired body structure (442083009)
            return [442083009]
        # Has interpretation: 363713009
        if top_level_rel == 363713009:
            # Finding value (260245000), Colors (263714004)
            # Environment or geographical location (308916002)
            return [260245000, 263714004, 308916002]
        # Has realization: 719722006
        if top_level_rel == 719722006:
            # Process (719982003)
            return [719982003]
        # Interprets: 363714003
        if top_level_rel == 363714003:
            # Laboratory procedure (108252007), Observable entity (363787002),
            # Evaluation procedure (386053000)
            return [108252007, 363787002, 386053000]
        # Occurrence: 246454002
        if top_level_rel == 246454002:
            # Periods of life (282032007)
            return [282032007]
        # Pathological process: 370135005
        if top_level_rel == 370135005:
            # Pathological developmental process (308490002), Infectious process (441862004)
            # Hypersensitivity process (472963003), Abnormal immune process (769247005)
            return [308490002, 441862004, 472963003, 769247005]
        # Severity: 246112005
        if top_level_rel == 246112005:
            # Severities (272141005)
            return [272141005]
    ## Event: 272379006
    if is_relative(origen, 272379006, concepts):
        # Is a: 116680003
        if top_level_rel == ES_UN_ID:
            # Event: 272379006
            return [272379006]
        # Associated with: 47429007 - TODO: Aquí tenemos 3 subtipos de la relación donde va cambiando el destino
        if top_level_rel == 47429007:
            # Physical object (260787004), Event (272379006), Clinical finding (404684003)
            # Organism (410607006), Procedure (71388002), Physical force (78621006)
            return [260787004, 272379006, 404684003, 410607006, 71388002, 78621006]
        # Occurrence: 246454002
        if top_level_rel == 246454002:
            # Periods of life (282032007)
            return [282032007]
    ## Observable entity: 363787002
    if is_relative(origen, 363787002, concepts):
        # Is a: 116680003
        if top_level_rel == ES_UN_ID:
            # Observable entity: 363787002
            return [363787002]
        # Characterizes: 704321009
        if top_level_rel == 704321009:
            # Procedure (71388002), Process (719982003)
            return [71388002, 719982003]
        # Component: 246093002
        if top_level_rel == 246093002:
            # Substance (105590001), Body structure (123037004), Specimen (123038009)
            # Physical object (260787004), Pharmaceutical / biologic product (373873005)
            # Organism (410607006), Record artifact (419891008)
            return [105590001, 123037004, 123038009, 260787004, 373873005, 410607006, 419891008]
        # Direct site: 704327008
        if top_level_rel == 704327008:
            # Substance (105590001), Body structure (123037004), Specimen (123038009)
            # Physical object (260787004), Pharmaceutical / biologic product (373873005)
            # Organism (410607006), Record artifact (419891008)
            return [105590001, 123037004, 123038009, 260787004, 373873005, 410607006, 419891008]
        # Has realization: 719722006
        if top_level_rel == 719722006:
            # Process (719982003)
            return [719982003]
        # Inherent location: 718497002
        if top_level_rel == 718497002:
            # Substance (105590001), Body structure (123037004), Specimen (123038009)
            # Physical object (260787004), Pharmaceutical / biologic product (373873005)
            # Organism (410607006), Record artifact (419891008)
            return [105590001, 123037004, 123038009, 260787004, 373873005, 410607006, 419891008]
        # Inheres in: 704319004
        if top_level_rel == 704319004:
            # Substance (105590001), Body structure (123037004), Specimen (123038009), Person (125676002)
            # Physical object (260787004), Pharmaceutical / biologic product (373873005)
            # Organism (410607006), Record artifact (419891008)
            return [105590001, 123037004, 125676002, 123038009, 260787004, 373873005, 410607006, 419891008]
        # Precondition: 704326004
        if top_level_rel == 704326004:
            # Clinical finding (404684003), Precondition value (703763000), Procedure (71388002)
            return [404684003, 703763000, 71388002]
        # Procedure device: 405815000 - TODO: Aquí tenemos subtipos de relación, pero el destino no cambia
        if top_level_rel == 405815000:
            # Device (49062001)
            return [49062001]
        # Process agent: 704322002
        if top_level_rel == 704322002:
            # Substance (105590001), Body structure (123037004), Organism (410607006)
            # Physical object (260787004), Pharmaceutical / biologic product (373873005)
            return [105590001, 123037004, 410607006, 260787004, 373873005]
        # Process duration: 704323007
        if top_level_rel == 704323007:
            # Time frame (7389001)
            return [7389001]
        # Process extends to: 1003703000
        if top_level_rel == 1003703000:
            # Body structure (123037004)
            return [123037004]
        # Process output: 704324001
        if top_level_rel == 704324001:
            # Substance (105590001), Process (719982003)
            return [105590001, 719982003]
        # Property: 370130000
        if top_level_rel == 370130000:
            # Property (118598001)
            return [118598001]
        # Relative to: 704325000
        if top_level_rel == 704325000:
            # Substance (105590001), Body structure (123037004), Specimen (123038009)
            # Physical object (260787004), Pharmaceutical / biologic product (373873005)
            # Organism (410607006), Record artifact (419891008)
            return [105590001, 123037004, 123038009, 260787004, 373873005, 410607006, 419891008]
        # Relative to part of: 719715003
        if top_level_rel == 719715003:
            # Substance (105590001), Body structure (123037004), Specimen (123038009)
            # Physical object (260787004), Pharmaceutical / biologic product (373873005)
            # Organism (410607006), Record artifact (419891008)
            return [105590001, 123037004, 123038009, 260787004, 373873005, 410607006, 419891008]
        # Scale type: 370132008
        if top_level_rel == 370132008:
            # Nominal value (117362005), Ordinal value (117363000), Narrative value (117364006)
            # Ordinal OR quantitative value (117365007), Text value (117444000), Qualitative (26716007)
            # Quantitative (30766002)
            return [117362005, 117363000, 117364006, 117365007, 117444000, 26716007, 30766002]
        # Technique: 246501002
        if top_level_rel == 246501002:
            # Staging and scales (254291000), Technique (272394005)
            return [254291000, 272394005]
        # Time aspect: 370134009
        if top_level_rel == 370134009:
            # Time frame (7389001)
            return [7389001]
        # Towards: 704320005
        if top_level_rel == 704320005:
            # Substance (105590001), Body structure (123037004), Specimen (123038009)
            # Physical object (260787004), Pharmaceutical / biologic product (373873005)
            # Organism (410607006), Record artifact (419891008)
            return [105590001, 123037004, 123038009, 260787004, 373873005, 410607006, 419891008]
        # Units: 246514001
        if top_level_rel == 246514001:
            # Unit of measure (767524001)
            return [767524001]
    ## Pharmaceutical / biologic product: 373873005
    if is_relative(origen, 373873005, concepts):
        # Is a: 116680003
        if top_level_rel == ES_UN_ID:
            # Pharmaceutical / biologic product: 373873005
            return [373873005]
        # Has basis of strength substance: 732943007
        if top_level_rel == 732943007:
            # Substance (105590001)
            return [105590001]
        # Has concentration strength denominator unit: 733722007
        if top_level_rel == 733722007:
            # Unit of measure (767524001)
            return [767524001]
        # Has concentration strength numerator unit: 733725009
        if top_level_rel == 733725009:
            # Unit of measure (767524001)
            return [767524001]
        # Has ingredient: 762951001 - TODO: Aquí tenemos subtipos de relación, pero el destino no cambia
        if top_level_rel == 762951001:
            # Substance (105590001)
            return [105590001]
        # Has ingredient characteristic: 860779006
        if top_level_rel == 860779006:
            # Qualifier value (362981000)
            return [362981000]
        # Has ingredient qualitative strength: 1149366004
        if top_level_rel == 1149366004:
            # Ingredient qualitative strength (1149484003)
            return [1149484003]
        # Has manufactured dose form: 411116001
        if top_level_rel == 411116001:
            # Pharmaceutical dose form (736542009)
            return [736542009]
        # Has presentation strength denominator unit: 732947008
        if top_level_rel == 732947008:
            # Unit of measure (767524001)
            return [767524001]
        # Has presentation strength numerator unit: 732945000
        if top_level_rel == 732945000:
            # Unit of measure (767524001)
            return [767524001]
        # Has product characteristic: 860781008
        if top_level_rel == 860781008:
            # Qualifier value (362981000)
            return [362981000]
        # Has product name: 774158006
        if top_level_rel == 774158006:
            # Product name (774167006)
            return [774167006]
        # Has supplier: 774159003
        if top_level_rel == 774159003:
            # Supplier (774164004)
            return [774164004]
        # Has target population: 1149367008
        if top_level_rel == 1149367008:
            # Product target population (27821000087106)
            return [27821000087106]
        # Has unit of presentation: 763032000
        if top_level_rel == 763032000:
            # Unit of presentation (732935002)
            return [732935002]
        # Plays role: 766939001
        if top_level_rel == 766939001:
            # Role (766940004)
            return [766940004]
        # Unit of presentation size unit: 320091000221107
        if top_level_rel == 320091000221107:
            # Unit of measure
            return [767524001]
        # Contains clinical drug: 774160008 - Esta es de un subtipo
        if top_level_rel == 774160008:
            # Medicinal product (763158003)
            return [763158003]
        # Has pack size unit: 774163005 - Esta es de un subtipo
        if top_level_rel == 774163005:
            # Unit of measure (767524001)
            return [767524001]
    ## Pharmaceutical dose form: 736542009
    if is_relative(origen, 736542009, concepts):
        # Is a: 116680003
        if top_level_rel == ES_UN_ID:
            # Pharmaceutical dose form: 736542009
            return [736542009]
        # Has basic dose form: 736476002
        if top_level_rel == 736476002:
            # Basic dose form (736478001)
            return [736478001]
        # Has dose form administration: 736472000
        if top_level_rel == 736472000:
            # Dose form administration method (736665006)
            return [736665006]
        # Has dose form intended site: 736474004
        if top_level_rel == 736474004:
            # Dose form intended site (736479009)
            return [736479009]
        # Has dose form release characteristic: 736475003
        if top_level_rel == 736475003:
            # Dose form characteristic (736480007)
            return [736480007]
        # Has dose form transformation: 736473005
        if top_level_rel == 736473005:
            # Dose form transformation (736477006)
            return [736477006]
    ## Physical object: 260787004
    if is_relative(origen, 260787004, concepts):
        # Is a: 116680003
        if top_level_rel == ES_UN_ID:
            # Physical object: 260787004
            return [260787004]
        # Has absorbability: 1148969005
        if top_level_rel == 1148969005:
            # Bioabsorbable (860574003), Nonbioabsorbable (863965006), Partially bioabsorbable (863968008)
            return [860574003, 863965006, 863968008]
        # Has coating material: 1148967007
        if top_level_rel == 1148967007:
            # Substance (105590001)
            return [105590001]
        # Has compositional material: 840560000
        if top_level_rel == 840560000:
            # Substance (105590001)
            return [105590001]
        # Has device intended site: 836358009
        if top_level_rel == 836358009:
            # Body structure (123037004)
            return [123037004]
        # Has filling: 827081001
        if top_level_rel == 827081001:
            # Substance (105590001)
            return [105590001]
        # Has surface texture: 1148968002
        if top_level_rel == 1148968002:
            # Smooth (82280004), Textured (860647008)
            return [82280004, 860647008]
        # Is sterile: 1148965004
        if top_level_rel == 1148965004:
            # True (31874001), False (64100000)
            return [31874001, 64100000]
    ## Procedure: 71388002
    if is_relative(origen, 71388002, concepts):
        # Is a: 116680003
        if top_level_rel == ES_UN_ID:
            # Procedure: 71388002
            return [71388002]
        # Access: 260507000
        if top_level_rel == 260507000:
            # Surgical access values (309795001)
            return [309795001]
        # Direct substance: 363701004
        if top_level_rel == 363701004:
            # Substance (105590001), Pharmaceutical / biologic product (373873005)
            return [105590001, 373873005]
        # Has focus: 363702006
        if top_level_rel == 363702006:
            # Clinical finding (404684003), Procedure (71388002)
            return [404684003, 71388002]
        # Has intent: 363703001
        if top_level_rel == 363703001:
            # Intents (363675004)
            return [363675004]
        # Method: 260686004
        if top_level_rel == 260686004:
            # Action (129264002)
            return [129264002]
        # Priority: 260870009
        if top_level_rel == 260870009:
            # Priorities (272125009)
            return [272125009]
        # Procedure device: 405815000 - TODO: Aquí tenemos subtipos de relación, pero el destino no cambia
        if top_level_rel == 405815000:
            # Device (49062001)
            return [49062001]
        # Procedure morphology: 405816004 - TODO: Aquí tenemos subtipos de relación, pero el destino no cambia
        if top_level_rel == 405816004:
            # Morphologically abnormal structure (49755003)
            return [49755003]
        # Procedure site: 363704007
        if top_level_rel == 363704007:
            # Anatomical or acquired body structure (442083009)
            return [442083009]
        # Recipient category: 370131001
        if top_level_rel == 370131001:
            # Person (125676002), Community (133928008), Family (35359004), Group (389109008)
            return [125676002, 133928008, 35359004, 389109008]
        # Revision status: 246513007
        if top_level_rel == 246513007:
            # Revision - value (255231005), Part of multistage procedure (257958009)
            # Primary operation (261424001)
            return [255231005, 257958009, 261424001]
        # Using energy: 424244007
        if top_level_rel == 424244007:
            # Physical force (78621006)
            return [78621006]
        # Using substance: 424361007
        if top_level_rel == 424361007:
            # Substance (105590001)
            return [105590001]
        ## Surgical procedure: 387713003 - Subtipo de Procedure
        # Surgical approach: 424876005
        if top_level_rel == 424876005:
            # Procedural approach (103379005)
            return [103379005]
        ## Evaluation procedure: 386053000 - Subtipo de Procedure
        # Component: 246093002
        if top_level_rel == 246093002:
            # Substance (105590001), Body structure (123037004), Specimen (123038009)
            # Physical object (260787004), Pharmaceutical / biologic product (373873005)
            # Organism (410607006), Record artifact (419891008)
            return [105590001, 123037004, 123038009, 260787004, 373873005, 410607006, 419891008]
        # Has specimen: 116686009
        if top_level_rel == 116686009:
            # Specimen (123038009)
            return [123038009]
        # Measurement method: 370129005
        if top_level_rel == 370129005:
            # Laboratory procedure categorized by method (127789004)
            return [127789004]
        # Property: 370130000
        if top_level_rel == 370130000:
            # Property (118598001)
            return [118598001]
        # Scale type: 370132008
        if top_level_rel == 370132008:
            # Nominal value (117362005), Ordinal value (117363000), Narrative value (117364006)
            # Ordinal OR quantitative value (117365007), Text value (117444000), Qualitative (26716007)
            # Quantitative (30766002)
            return [117362005, 117363000, 117364006, 117365007, 117444000, 26716007, 30766002]
        # Time aspect: 370134009
        if top_level_rel == 370134009:
            # Time frame (7389001)
            return [7389001]
        ## Administration of substance via specific route: 433590000 - Subtipo de Procedure
        # Route of administration: 410675002
        if top_level_rel == 410675002:
            # Route of administration value (284009009)
            return [284009009]
    ## Situation with explicit context: 243796009
    if is_relative(origen, 243796009, concepts):
        # Is a: 116680003
        if top_level_rel == ES_UN_ID:
            # Situation with explicit context: 243796009
            return [243796009]
        # Subject relationship context: 408732007
        if top_level_rel == 408732007:
            # Person (125676002)
            return [125676002]
        # Temporal context: 408731000
        if top_level_rel == 408731000:
            # Temporal context value (410510008)
            return [410510008]
        ## Procedure with explicit context: 129125009 - Subtipo de Situation with explicit context
        # Associated procedure: 363589002
        if top_level_rel == 363589002:
            # Procedure (71388002)
            return [71388002]
        # Procedure context: 408730004
        if top_level_rel == 408730004:
            # Context values for actions (288532009)
            return [288532009]
        ## Finding with explicit context: 413350009 - Subtipo de Situation with explicit context
        # Associated finding: 246090004
        if top_level_rel == 246090004:
            # Event (272379006), Clinical finding (404684003)
            return [272379006, 404684003]
        # Finding context: 408729009
        if top_level_rel == 408729009:
            # Finding context value (410514004)
            return [410514004]
    ## Specimen: 123038009
    if is_relative(origen, 123038009, concepts):
        # Is a: 116680003
        if top_level_rel == ES_UN_ID:
            # Specimen: 123038009
            return [123038009]
        # Specimen procedure: 118171006
        if top_level_rel == 118171006:
            # Procedure (71388002)
            return [71388002]
        # Specimen source identity: 118170007
        if top_level_rel == 118170007:
            # Person (125676002), Community (133928008), Physical object (260787004)
            # Environment (276339004), Family (35359004)
            return [125676002, 133928008, 260787004, 276339004, 35359004]
        # Specimen source morphology: 118168003
        if top_level_rel == 118168003:
            # Morphologically abnormal structure (49755003)
            return [49755003]
        # Specimen source topography: 118169006
        if top_level_rel == 118169006:
            # Anatomical or acquired body structure (442083009)
            return [442083009]
        # Specimen substance: 370133003
        if top_level_rel == 370133003:
            # Substance (105590001), Physical object (260787004), Pharmaceutical / biologic product (373873005)
            return [105590001, 260787004, 373873005]
    ## Substance: 105590001
    if is_relative(origen, 105590001, concepts):
        # Is a: 116680003
        if top_level_rel == ES_UN_ID:
            # Substance: 105590001
            return [105590001]
        # Has disposition: 726542003
        if top_level_rel == 726542003:
            # Disposition (726711005)
            return [726711005]
        # Is modification of: 738774007
        if top_level_rel == 738774007:
            # Substance (105590001
            return [105590001]

    # Si no es de ninguno de esos tipos, pues devolvemos el conjunto vacío
    return []

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
            print('El modelo no reconoce:', n)

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


# Función para convertir correctamente las listas de listas o listas de cadena
# de texto a lista de listas
def read_list_of_lists(string):
    l = []

    if string == '[]':
        return l

    elements = string.split('],')

    for ele in elements:
        ele = re.sub('[\]\[]', '', ele)
        ns = ele.split(',')
        l2 = []
        for ele in ns:
            l2.append(int(ele))
        l.append(l2)

    return l


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
    simbolos = ['(', ')', '.', '[', ']', ':', '-', '/']
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


# Compares which is the most common semantic type among the first list_len elements
# of list_similar. Returns the most common semantic type and the position of the 
# first element of that type
def mismo_semantic_tag(list_similar, concepts_training, list_len=5):
    aux = {}

    n = 0
    for _, id, _ in list_similar[:list_len]:
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


# Receives as input a list of tuples (concept ID, relationship ID) and a relationship ID.
# Checks if there is a tuple in the list that is of a different relationship ID
def contiene_relacion_distinta_a(list_lists, id=ES_UN_ID):
    if len(list_lists) == 0:
        return False

    for tup in list_lists:
        if tup[1] != ES_UN_ID:
            return True

    return False


# Compares if any element in list_similar is of the same semantic type (semantic_tag)
# and if it is not a navigation concept (if it has a relationship different than is a).
# Returns the position in the list of the first element that shares those characteristics.
# Returns -1 if there is none
def mismo_semantic_tag_rels(semantic_tag, list_similar, concepts_training):
    n = 0
    for _, id, _ in list_similar:
        if semantic_tag == concepts_training[id]['semantic_tag'] and contiene_relacion_distinta_a(
                concepts_training[id]['relations']):
            return n
        n += 1

    return -1


# It receives two concepts A and B and returns the percentage of relationships
# of A that exist in B and which relationship they share. If A does not contain
# any relationship, it returns -1 and an empty array
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


# Receives the metadata dictionary and a relationship ID
# and returns the ID of the top relationship that maintains
# the same meaning. We consider the top relationship as the
# relationship that is on higher position in the hierarchy
def top_relation(metadatos, rel):
    for destID, typeID in metadatos[rel]['relations']:
        if typeID == ES_UN_ID:
            if destID in TOP_RELATIONS:
                return rel
            else:
                return top_relation(metadatos, destID)


# Returns the first n possible tail concepts for an analogy between three word embeddings
# A, B, and C. Those tail concepts are returned in a tuple (conceptID, cosine similarity). 
# An analogy is what A is to B is what C is to X
# Receives as input the word embedding for the training concepts (embs), their IDs (ids),
# the word embeddings for A, B, and C, and the number of analogies to return
def get_analogias(embs, ids, wordA, wordB, wordC, n=10):
    wordX = wordB - wordA + wordC
    sims_X = cosine_similarity(X=wordX.reshape(1, -1), Y=embs)[0]

    sim_list = list(zip(ids, sims_X))

    sim_list.sort(key=lambda x: x[1], reverse=True)

    return sim_list[:n]


# Returns the first tail concept of a given relatinoship for a certain
# concept. It receives an array of relationship tuples (concept ID, relationship ID)
def getIDdestino(relaciones, relID):
    for destID, typeID in relaciones:
        if relID == typeID:
            return destID

    return -1

# Receives a head concept, an analogies array, a relationship ID, the valid concept types
# and all the concepts. Checks in which position of the analogies array we can find the 
# correct tail concept and returns it, but filtering using the valid concept types. 
# Returns -1 otherwise 
def concepto_destino_valido(analogias, concepto, conceptos_validos, relID, concepts):
    posibles_destinos = []
    for destID, typeID in concepto['relations']:
        if typeID == relID:
            posibles_destinos.append(destID)

    n = 0
    for conceptID, _ in analogias:
        # We first check if the concept is valid
        if conceptos_validos == [] or is_relative_multiples(conceptID, conceptos_validos, concepts):
            if conceptID in posibles_destinos:
                return n
            n += 1

    return -1

# Receives a head concept, an analogies array and relationship ID. Checks in which position
# of the analogies array we can find the correct tail concept and returns it. 
# Returns -1 otherwise
def concepto_destino_correcto(analogias, concepto, relID):
    posibles_destinos = []
    for destID, typeID in concepto['relations']:
        if typeID == relID:
            posibles_destinos.append(destID)

    n = 0
    for conceptID, _ in analogias:
        if conceptID in posibles_destinos:
            return n
        n += 1

    return -1


# Return the neighbours of a concept
# Receives as input the dictionary of concepts
# and the concept ID
def get_vecinos(diccionario, nodo):
    return diccionario[nodo]['vecinos']


# Performs each random walk of certain depth
# for the concept nodo_inicial. It receives as input
# the dictionary of concepts, the ID of the initial concept and
# the depth. It returns a list of random walks
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


# Obtains every random_walk of a certain depth
# for each concept in the dictionary
def random_walks(diccionario, depth=1):
    all_walks = set()
    for conceptID, concept in diccionario.items():

        walks = random_walk(diccionario, conceptID, depth)
        for walk in walks:
            all_walks.add(walk)

    return all_walks
