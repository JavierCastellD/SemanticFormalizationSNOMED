import sys
import gensim
from pathlib import Path
from postcoordinate_functions import preprocesar_texto, get_jerarquia

# Importing hierarchy
PATH, INPUT, LOGS, MODELS, DICT, CORPUS, CONCEPTS = get_jerarquia()

model_name = sys.argv[1]
corpus_path = sys.argv[2]

# Hyparameters for language model
embed_size = int(sys.argv[3])
window_size = int(sys.argv[4])

language = sys.argv[5]

print('Embed_size =', embed_size, '| Window_size =', window_size)

for i in range(1):
    print('Test:', i)

    # Folder for the model
    id_prueba = '_test' + str(i)
    TEST_FOLDER = 'embed' + str(embed_size) + '_window' + str(window_size) + '_' + str(i) + '/'

    # Path to the generated models
    W2V_REL = 'word2vec/word2vec' + id_prueba + '.model'
    FT_REL = 'ft/ft' + id_prueba + '.model'

    ######################
    # TRAINING THE MODEL #
    ######################

    print('Preprocessing the sentences of the corpus')
    with open(PATH + CORPUS + corpus_path) as corpus_pruebas_file:
        sentences_corpus = corpus_pruebas_file.readlines()

    sentences = preprocesar_texto(sentences_corpus, language)

    if model_name == 'w2v':
        print('Training Word2Vec')
        w2v_model = gensim.models.Word2Vec(sentences=sentences,
                                           size=embed_size,
                                           window=window_size,
                                           min_count=1,
                                           workers=4, sg=1)
        Path(PATH + MODELS + TEST_FOLDER + 'w2v/').mkdir(parents=True, exist_ok=True)
        w2v_model.save(PATH + MODELS + TEST_FOLDER + W2V_REL)

    else:
        print('Training FastText')
        ft_model = gensim.models.FastText(sentences=sentences,
                                          size=embed_size,
                                          window=window_size,
                                          min_count=1,
                                          workers=4, sg=1)
        Path(PATH + MODELS + TEST_FOLDER + 'ft/').mkdir(parents=True, exist_ok=True)
        ft_model.save(PATH + MODELS + TEST_FOLDER + FT_REL)