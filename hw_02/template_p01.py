import numpy as np

def softmax(vector):
    '''
    vector: np.array of shape (n, m)
    
    return: np.array of shape (n, m)
        Matrix where softmax is computed for every row independently
    '''
    nice_vector = vector - vector.max()
    exp_vector = np.exp(nice_vector)
    exp_denominator = np.sum(exp_vector, axis=1)[:, np.newaxis]
    softmax_ = exp_vector / exp_denominator
    return softmax_

def multiplicative_attention(decoder_hidden_state, encoder_hidden_states, W_mult):
    '''
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    W_mult: np.array of shape (n_features_dec, n_features_enc)
    
    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    '''
    # your code here
    e = decoder_hidden_state.T @ W_mult @ encoder_hidden_states  # Shape (1, n_states)
    softmax_vector = softmax(e) # Shape (1, n_states)
    attention_vector = encoder_hidden_states @ softmax_vector.T  # Shape (n_features_enc, 1)
    return attention_vector
    

def additive_attention(decoder_hidden_state, encoder_hidden_states, v_add, W_add_enc, W_add_dec):
    '''
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    v_add: np.array of shape (n_features_int, 1)
    W_add_enc: np.array of shape (n_features_int, n_features_enc)
    W_add_dec: np.array of shape (n_features_int, n_features_dec)
    
    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    '''
    # your code here
    enc_proj = W_add_enc @ encoder_hidden_states  # Shape (n_features_int, n_states)
    dec_proj = W_add_dec @ decoder_hidden_state  # Shape (n_features_int, 1)
    tanh_combined = np.tanh(enc_proj + dec_proj)
    e = v_add.T @ tanh_combined  # Shape (1, n_states)
    softmax_vector = softmax(e) # Shape (1, n_states)
    attention_vector = encoder_hidden_states @ softmax_vector.T  # Shape (n_features_enc, 1)
    return attention_vector