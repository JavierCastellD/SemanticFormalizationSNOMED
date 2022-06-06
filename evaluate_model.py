import sys
import torch
import gensim
import json
import nltk
import numpy  as np
import pandas as pd
from transformers             import BertModel, BertTokenizer
from sentence_transformers   import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from postcoordinate_functions import string_keys_to_int, preprocesar_frase, get_jerarquia, mismo_semantic_tag, mismo_semantic_tag_rels, mismas_relaciones, getIDdestino, concepto_destino_correcto, is_relative, valid_dest_relations, concepto_destino_valido

# We load the hierarchy
PATH, INPUT, LOGS, MODELS, DICT, CORPUS, CONCEPTS = get_jerarquia()
ES_UN_ID = 116680003
METADATOS_FILE = 'metadatos.json'

# Name of the model (ft, w2v, bert, sbert)
model_name = sys.argv[1]

# Path to the model relative to the models folder (write anything if the model is sbert)
model_path = sys.argv[2]

# Language of the corpus
language = sys.argv[3]

# We load the metadata
metadatos_file = open(PATH + CONCEPTS + METADATOS_FILE)
metadatos = string_keys_to_int(json.load(metadatos_file))
metadatos_file.close()

for i in range(1):
    print('Paso:', i)
    ##############################
    #     IMPORTANT VARIABLES    #
    ##############################

    # Test ID
    id_prueba = '_test' + str(i)
    
    # Concepts dictionaries names
    CONCEPTS_TRAINING = 'active_concepts_testing.json'
    CONCEPTS_EVALUATION = 'concepts_test.json'

    # Load training concepts
    training_concepts_file = open(PATH + CONCEPTS + CONCEPTS_TRAINING)
    concepts_training = string_keys_to_int(json.load(training_concepts_file))
    training_concepts_file.close()

    # Load evaluation concepts
    evaluation_concepts_file = open(PATH + CONCEPTS + CONCEPTS_EVALUATION)
    concepts_evaluation = string_keys_to_int(json.load(evaluation_concepts_file))
    evaluation_concepts_file.close()

    ##########################
    #    MODEL INFORMATION   #
    ##########################
    if model_name == 'w2v':
        print('Model: Word2Vec')
        model = gensim.models.Word2Vec.load(PATH + MODELS + model_path)
        log_name = 'log_w2v' + id_prueba + '.tsv'
    elif model_name == 'ft':
        print('Model: FastText')
        model = gensim.models.FastText.load(PATH + MODELS + model_path)
        log_name = 'log_ft' + id_prueba + '.tsv'
    elif model_name == 'bert':
        print('Model: BERT')
        model = BertModel.from_pretrained(PATH + MODELS + model_path, output_hidden_states=True)
        tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.2', do_lower_case=False)
        model.eval()
        log_name = 'log_bert' + id_prueba + '.tsv'
    elif model_name == 'sbert':
        print('Model: SBERT')
        model = SentenceTransformer('all-mpnet-base-v2')
        log_name = 'log_sbert' + id_prueba + '.tsv'
        
    #########################################
    #   GENERATION OF CONCEPTS DICTIONARY   #
    #########################################
    print('Generating the concepts dictionary')

    # If the dictionary concept was already created, you can load it instead
    # dict_concepts = string_keys_to_int(json.load(dictionary_concepts_file))

    dict_concepts = {}
    if model_name == 'ft' or model_name == 'w2v':
        for conceptID, concept in concepts_training.items():
            # We obtain the word embedding from the FSN
            name = preprocesar_frase(concept['FSN'])
            palabras_name = nltk.word_tokenize(name, language=language)
          
            # Generate the word embedding using the language model
            vectors = []
            for palabra in palabras_name:
                try:
                    vectors.append(model.wv.get_vector(palabra))
                except:
                    print(palabra + ' not found')


            v = sum(vectors)
            if isinstance(v, np.ndarray):
                dict_concepts[conceptID] = {'vector': v.tolist(), 'vectorURI': np.array([]), 'FSN': name}

    elif model_name == 'bert':
        for conceptID, concept in concepts_training.items():
            # We obtain the word embedding from the FSN
            name = concept['FSN']
            marked_name = "[CLS]" + name + "[SEP]"

            # We tokenize it using the corresponding tokenizer
            tokenized_name = tokenizer.tokenize(marked_name)

            # We index those tokens in the tokenizer vocabulary
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_name)

            # This is because BERT is trained for sentence pairs and it distinguish them
            # by generating segment IDs of 1s and 0s
            # We only use one sentence per concept, so we set all 1s
            segments_ids = [1] * len(tokenized_name)

            # Transform the input to PyTorch tensors
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])

            # We evaluate the name to generate the output
            outputs = model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]

            # Obtain the vector of each word
            token_vecs = hidden_states[-2][0]

            # Use the mean of the vectors to obtain the word embedding
            sentence_embedding = torch.mean(token_vecs, dim=0)

            vector_URI = np.array([])

            dict_concepts[conceptID] = {'vector': sentence_embedding.tolist(), 'vectorURI': vector_URI.tolist(),
                                        'FSN': name}

    elif model_name == 'sbert':
        ids = []
        names = []
        for conceptID, concept in concepts_training.items():
            ids.append(conceptID)
            names.append(concept['FSN'])

        embeddings = model.encode(names, convert_to_tensor=True)

        for embedding, conceptID, name in zip(embeddings, ids, names):
            dict_concepts[conceptID] = {'vector': embedding.tolist(), 'vectorURI': [], 'FSN': name}

    #####################
    # OBTAINING THE LOG #
    #####################
    print('Obtaining the evaluation log')

    # We extract the list of embeddings, names and ids
    embs = []
    embs_URI = []
    fsns = []
    ids = []
    for id, ele in dict_concepts.items():
        embs_URI.append(np.array(ele['vectorURI']))
        fsns.append(ele['FSN'])
        embs.append(np.array(ele['vector']))
        ids.append(id)

    # Intermediary structure to create the log
    stats = {}
    
    # Open the log file
    log = open(PATH + LOGS + log_name, 'w')

    # Write the header
    log.write('Concepto\tID\tSemantic tag\tTop\tTop Rel\tSimilares\tPalabras no contenida\t% palabras contenidas\t'
              'Relaciones igual 1\t% igual 1\tNombres igual 1\tID 1\t'
              'Relaciones igual match\t% igual match\tNombres igual match\tID match\n')
    
    concepts_vectors = []
    concepts_ids = []
    
    # For each evaluation concept
    for conceptID, concept in list(concepts_evaluation.items()):
        stats[conceptID] = {'sem' : '', 'FSN' : '', 'pal_no_c' : 0, '%_pal' : 0, 'Vector' : True}
        # Obtain the FSN
        FSN = concept['FSN']
    
        # Obtain the semantic tag
        semantic_tag = concept['semantic_tag']
      
        stats[conceptID]['sem'] = semantic_tag
        stats[conceptID]['FSN'] = FSN

        # Aux variable to identify the number of words which are not in the dictionary
        n_palabras_no_diccionario = 0

        if model_name == 'ft' or model_name == 'w2v':
            # We preprocess the name and separate each word
            name = preprocesar_frase(FSN)
            palabras_name = nltk.word_tokenize(name, language=language)

            # Obtain the vector for each word
            vectors = []
            for palabra in palabras_name:
                try:
                    vectors.append(model.wv.get_vector(palabra))
                except:
                    n_palabras_no_diccionario += 1

            sentence_embedding = sum(vectors)

        elif model_name == 'bert':
            name = FSN
            marked_name = "[CLS]" + name + "[SEP]"

            # We tokenize it using the corresponding tokenizer
            tokenized_name = tokenizer.tokenize(marked_name)

            # We index those tokens in the tokenizer vocabulary
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_name)

            # Create the segment IDs of 1s
            segments_ids = [1] * len(tokenized_name)

            # Transform the input to PyTorch
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])

            # Evaluate it using the model
            outputs = model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]

            # Obtain the vector of the concept name
            token_vecs = hidden_states[-2][0]

            # Create the word embedding from the mean of the vectors
            sentence_embedding = torch.mean(token_vecs, dim=0)
            sentence_embedding = sentence_embedding.tolist()

        elif model_name == 'sbert':
            name = FSN

            sentence_embedding = model.encode(name, convert_to_tensor=True)


        # For each evaluation concept, we store inforamtion that might be useful
        stats[conceptID]['pal_no_c'] = n_palabras_no_diccionario
        if len(palabras_name) != 0:
            stats[conceptID]['%_pal'] = 1 - round(n_palabras_no_diccionario / len(palabras_name), 4)
        else:
            print('NO palabras name en', FSN)
            stats[conceptID]['%_pal'] = 0
      
        # We consider that some vectors might be empty
        if vectors != []:
            stats[conceptID]['Vector'] = True
            word_vector = sentence_embedding
            concepts_ids.append(conceptID)
            concepts_vectors.append(word_vector)
            stats[conceptID]['wv'] = np.array(word_vector)
        else:
            stats[conceptID]['Vector'] = False

    print('Calculating similarity of the concepts')
    sims_V = cosine_similarity(X=concepts_vectors, Y=embs)
    
    # For the analogies of the first concept
    wordsX = []
    eval_ids_an = []
    rel_ids = []

    # For the analogies of the topRel concept
    wordsX2 = []
    eval_ids_an2 = []
    rel_ids2 = []
    
    # For each evaluation concept, we obtain the semantic category and relationships
    # Then we perform the analogies
    for sim_V, ID in zip(sims_V, concepts_ids):
        # We sort the list of similarities from most similar to less
        sim_list = list(zip(fsns, ids, sim_V))

        sim_list.sort(key=lambda x: x[2], reverse=True)
        sim_list_V = sim_list[:10]
      
        salida = stats[ID]['FSN'] + '\t' + str(ID) + '\t' + stats[ID]['sem'] + '\t'
      
        # Obtain the semantic tag
        st, pos = mismo_semantic_tag(sim_list_V, concepts_training)
      
        # If the semantic tag is correct, the value of top is 0
        # Otherwise it is 1
        top = -1
        if st == stats[ID]['sem']:
            top = 0
      
        # Obtain the first concept which uses that semantic tag and it is not a navigational concept
        topRel = mismo_semantic_tag_rels(stats[ID]['sem'], sim_list_V, concepts_training)
      
        similares_solo_IDs = [id for _, id, _ in sim_list_V]
        salida += str(top) + '\t' + str(topRel) + '\t' + str(similares_solo_IDs) + '\t' + str(stats[ID]['pal_no_c']) + '\t' + str(stats[ID]['%_pal']) + '\t'

        # For the most similar concept, we always calculate the relationships and analogies
        ID_conceptB = sim_list_V[0][1]
        porcentajeIgual, iguales = mismas_relaciones(concepts_training[ID_conceptB], concepts_evaluation[ID], metadatos)

        salida += str(len(iguales)) + '\t' + str(porcentajeIgual) + '\t' + str(iguales) + '\t' + str(ID_conceptB) + '\t'

        # Perform the analogy
        for relID in iguales:
            idDestino = getIDdestino(concepts_training[ID_conceptB]['relations'], relID)

            if idDestino == -1:
                print('Error, there is no relationship of type', relID, 'for the concept', ID_conceptB)
            else:
                wordX = np.array(dict_concepts[idDestino]['vector']) - np.array(dict_concepts[ID_conceptB]['vector']) + \
                        stats[ID]['wv']
                wordsX.append(wordX)
                eval_ids_an.append(ID)
                rel_ids.append(relID)

        # If we found the semantic category
        if top != -1:
            # This is just in case there are only navigational concepts
            if topRel == -1:
                topRel = pos

            # If topRel is the first concept, we don't need to repeat this
            if topRel != 0:
                ID_conceptB = sim_list_V[topRel][1]

                porcentajeIgual, iguales = mismas_relaciones(concepts_training[ID_conceptB], concepts_evaluation[ID], metadatos)

            salida += str(len(iguales)) + '\t' + str(porcentajeIgual) + '\t' + str(iguales) + '\t' + str(ID_conceptB) + '\t'

            # Perform the analogy
            for relID in iguales:
                idDestino = getIDdestino(concepts_training[ID_conceptB]['relations'], relID)

                if idDestino == -1:
                    print('Error, there is no relationship of type', relID, 'for the concept', ID_conceptB)
                else:
                    wordX = np.array(dict_concepts[idDestino]['vector']) - np.array(dict_concepts[ID_conceptB]['vector']) + stats[ID]['wv']
                    wordsX2.append(wordX)
                    eval_ids_an2.append(ID)
                    rel_ids2.append(relID)

        # If none shares the semantic tag
        else:
            salida += '0\t0\t[]\t-1'
        
        # Write the ouput
        log.write(salida + '\n')
    
    # Write the output for those whose vector was empty
    for ID, value in stats.items():
        if not value['Vector']:
            salida = value['FSN'] + '\t' + str(ID) + '\t' + value['sem'] + '\t'
            salida += str(-1) + '\t' + str(-1) + '\t[]\t' + str(value['pal_no_c']) + '\t0\t0\t0\t[]\t-1\t0\t0\t[]\t-1'
            log.write(salida + '\n')
    
    log.close()
    
    # Open the log file as a dataframe to add new information
    df = pd.read_csv(PATH + LOGS + log_name, sep='\t', index_col=False)

    df['Analogias 1'] = np.nan
    df['Analogias 1 Lista'] = np.nan
    df['Analogias Sem'] = np.nan
    df['Analogias Sem Lista'] = np.nan

    df['Analogias 1'] = df['Analogias 1'].astype('object')
    df['Analogias 1 Lista'] = df['Analogias 1 Lista'].astype('object')
    df['Analogias Sem'] = df['Analogias Sem'].astype('object')
    df['Analogias Sem Lista'] = df['Analogias Sem Lista'].astype('object')
    
     # Perform the similarity for the analogies of the first concept
    sims_analogies = cosine_similarity(X=wordsX, Y=embs)
    
    topAns = []
    topAnalog = []
    evalID_actual = eval_ids_an[0]
    
    # For each analogy
    for sim, evalID, relID in zip(sims_analogies, eval_ids_an, rel_ids):
        # If we are still looking at the analogies of a prior concept, we write it
        if evalID != evalID_actual:
            df.loc[df['ID'] == evalID_actual, 'Analogias 1'] = pd.Series([topAns] * len(df))
            df.loc[df['ID'] == evalID_actual, 'Analogias 1 Lista'] = pd.Series([topAnalog] * len(df))

            topAns = []
            topAnalog = []
            evalID_actual = evalID
    
        # Obtain the top 100 most similar concepts of the analogy
        sim_list = list(zip(ids, sim))
        sim_list.sort(key=lambda x : x[1], reverse = True)
      
        analogias = sim_list[:100]

        id_concepto_similar = list(df.loc[df['ID'] == evalID_actual, 'ID match'])[0]
        if id_concepto_similar == -1:
            id_concepto_similar = list(df.loc[df['ID'] == evalID_actual, 'ID 1'])[0]

        # Obtain the valid tail concepts for this relationship
        destinos_validos = valid_dest_relations(id_concepto_similar, relID, concepts_training, metadatos)

        analogias_solo_IDs = [id for id, _ in analogias[:10]]
      
        # Check if we found the correct one
        topAn = concepto_destino_valido(analogias, concepts_evaluation[evalID], destinos_validos, relID, concepts_training)
        topAns.append([relID, topAn])
        topAnalog.append(analogias_solo_IDs)

    df.loc[df['ID'] == evalID_actual, 'Analogias 1'] = pd.Series([topAns] * len(df))
    df.loc[df['ID'] == evalID_actual, 'Analogias 1 Lista'] = pd.Series([topAnalog] * len(df))

    ###################################
    # REPEAT IT FOR THE SEMANTIC TYPE #
    ###################################

    # Perform the similarity for the analogies of the semantic types
    sims_analogies2 = cosine_similarity(X=wordsX2, Y=embs)

    topAns = []
    topAnalog = []
    evalID_actual = eval_ids_an2[0]

    for sim, evalID, relID in zip(sims_analogies2, eval_ids_an2, rel_ids2):
        # If we are still looking at the analogies of a prior concept, we write it
        if evalID != evalID_actual:
            df.loc[df['ID'] == evalID_actual, 'Analogias Sem'] = pd.Series([topAns] * len(df))
            df.loc[df['ID'] == evalID_actual, 'Analogias Sem Lista'] = pd.Series([topAnalog] * len(df))

            topAns = []
            topAnalog = []
            evalID_actual = evalID

        # Obtain the top 100 most similar concepts of the analogy
        sim_list = list(zip(ids, sim))

        sim_list.sort(key=lambda x: x[1], reverse=True)

        analogias = sim_list[:100]

        id_concepto_similar = list(df.loc[df['ID'] == evalID_actual, 'ID match'])[0]
        if id_concepto_similar == -1:
            id_concepto_similar = list(df.loc[df['ID'] == evalID_actual, 'ID 1'])[0]

        # Obtain the valid tail concepts for this relationship
        destinos_validos = valid_dest_relations(id_concepto_similar, relID, concepts_training, metadatos)

        analogias_solo_IDs = [id for id, _ in analogias[:10]]
      
        # Check if we found the correct one
        topAn = concepto_destino_valido(analogias, concepts_evaluation[evalID], destinos_validos, relID, concepts_training)

        topAns.append([relID, topAn])
        topAnalog.append(analogias_solo_IDs)

    df.loc[df['ID'] == evalID_actual, 'Analogias Sem'] = pd.Series([topAns] * len(df))
    df.loc[df['ID'] == evalID_actual, 'Analogias Sem Lista'] = pd.Series([topAnalog] * len(df))

    # Save the dataframe
    df.to_csv(PATH + LOGS + log_name, sep='\t', index=False)