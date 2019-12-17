import numpy as np


def encode_sequence(list_of_text, nlp_model, pad_string="xxxpad",
                   n_of_dims=96):
    """
    Encodes a list of text with an nlp_model
    Pads text with a string `pad_string` so they all have the same length
    
    n_of_dims: number of dimensions per word that the nlp_model brings
    """
    max_length = 0
    n_of_sentences = len(list_of_text)
    lengths = np.empty(n_of_sentences, dtype='int')
    for i, sentence in enumerate(list_of_text):
        tokens = nlp_model(sentence)
        lengths[i] = len(tokens)
        max_length = max(max_length, len(tokens))
    
    array_of_encodings = np.empty((len(list_of_text), max_length, n_of_dims))
    
    for i, sentence in enumerate(list_of_text):
        #pad text
        sentence += (' ' + pad_string)*(max_length - lengths[i])
        tokens = nlp_model(sentence)
        array_of_encodings[i, :, :] = np.array([token.vector for token in tokens])

    return array_of_encodings


def get_embedding_and_index_item_matrix(list_of_text, nlp_model, pad_string="xxxpad",
                   n_of_embedding_dims=96):
    """
    Encodes a list of text with an nlp_model to produce
    + index_item_matrix: (n_of_sentences, max_length)
    + embedding matrix: (n_of_word_in_vocabulary, n_of_embedding_dims)
    + unique_tokens_d: dictionary of tokens to indices
    
    Pads text with a string `pad_string` so they all have the same length
    
    n_of_embedding_dims: number of dimensions per word that the nlp_model brings
    """
    max_length = 0
    n_of_sentences = len(list_of_text)
    lengths = np.empty(n_of_sentences, dtype='int')
    unique_tokens_d = {}
    ind = 0
    for i, sentence in enumerate(list_of_text):
        tokens = nlp_model(sentence)
        lengths[i] = len(tokens)
        max_length = max(max_length, len(tokens))
        for token in tokens:
            if token.text not in unique_tokens_d.keys():
                unique_tokens_d[token.text] = ind
                ind += 1
    
    #Add the padded string
    unique_tokens_d[pad_string] = ind
    
    index_item_matrix = np.empty( (len(list_of_text), max_length), dtype='int' )
    embedding_matrix = np.empty(  (len(unique_tokens_d.values()), n_of_embedding_dims)  )
    
    #Index_item_matrix
    for i, sentence in enumerate(list_of_text):
        #pad text
        sentence += (' ' + pad_string)*(max_length - lengths[i])
        tokens = nlp_model(sentence)
        index_item_matrix[i, :] = [unique_tokens_d[token.text] for token in tokens  ]
        
    #Embedding_matrix
    for text, ind in unique_tokens_d.items():
        embedding_matrix[ind, :] = nlp_model(text).vector
        
    return index_item_matrix, embedding_matrix, unique_tokens_d
    