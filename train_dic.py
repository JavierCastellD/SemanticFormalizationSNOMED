import sys
import torch
import gensim
import json
import nltk
import numpy   as np
from transformers               import BertModel, BertTokenizer
from sentence_transformers      import SentenceTransformer
from postcoordinate_functions   import string_keys_to_int, preprocesar_frase, get_jerarquia

PATH, INPUT, LOGS, MODELS, DICT, CORPUS, CONCEPTS = get_jerarquia()

model_name = sys.argv[1]
model_path = sys.argv[2]
concepts_path = sys.argv[3]
language = sys.argv[4]

for i in range(1):
    print('Paso:', i)
    #######################
    # IMPORTANT VARIABLES #
    #######################

    # Test ID
    id_prueba = '_test' + str(i)

    # Concepts dictionaries names
    CONCEPTS_FILE = concepts_path

    # Load training concepts
    training_concepts_file = open(PATH + CONCEPTS + CONCEPTS_FILE)
    concepts_training = string_keys_to_int(json.load(training_concepts_file))
    training_concepts_file.close()

    #####################
    # MODEL INFORMATION #
    #####################

    if model_name == 'w2v':
        print('Model: Word2Vec')
        model = gensim.models.Word2Vec.load(PATH + MODELS + model_path)
        dict_name = 'concepts_dictionary_w2v'+id_prueba+'.json'
    elif model_name == 'ft':
        print('Model: FastText')
        model = gensim.models.FastText.load(PATH + MODELS + model_path)
        dict_name = 'concepts_dictionary_ft'+id_prueba+'.json'
    elif model_name == 'bert':
        print('Model: BERT')
        model = BertModel.from_pretrained(PATH + MODELS + model_path, output_hidden_states=True)
        tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.2', do_lower_case=False)
        model.eval()
        dict_name = 'concepts_dictionary_bert' + id_prueba + '.json'
    elif model_name == 'sbert':
        print('Model: SBERT')
        model = SentenceTransformer('all-mpnet-base-v2')
        dict_name = 'concepts_dictionary_sbert' + id_prueba + '.json'


    #########################################
    #   GENERATION OF CONCEPTS DICTIONARY   #
    #########################################
    print('Generating the concepts dictionary')

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


    concepts_dict_file = open(PATH + DICT + dict_name, 'w')
    json.dump(dict_concepts, concepts_dict_file, indent=4)
    concepts_dict_file.close()