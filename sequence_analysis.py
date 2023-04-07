import numpy as np
import matplotlib.pyplot as plt
import itertools

def check_matched_seqences(prediction, ground_truth):
    
    prediction_boolean = prediction>0.5
    prediction_binary = np.where(prediction_boolean, 1, 0)


    prediction_binary = prediction_binary

    unmatched_SequenceSteps = np.any(prediction_binary!=ground_truth,axis=2)
    unmatched_Sequence =  np.any( unmatched_SequenceSteps == True,axis=1)
    unmatched_Sequence_idx = np.where(np.any( unmatched_SequenceSteps == True,axis=1))
    print("Unmatched sequence index: ", unmatched_Sequence_idx)
    matchedSequences = ~unmatched_Sequence
    print(" Matched Sequences: ", np.sum(matchedSequences),"/",len(matchedSequences), " ",  np.sum(matchedSequences)/len(matchedSequences)*100)
    return unmatched_Sequence_idx

def display_2d_sequence(sequence_data, sequence_steps, sequence_display_begin = 0):
    plt.figure(figsize=(8,16))
    n_sequences_to_show = 10
    sequence_steps =3 
    f, axs = plt.subplots(sequence_steps, n_sequences_to_show)
    for i,j in itertools.product(range(sequence_steps), range(n_sequences_to_show)):
        axs[i,j].imshow(sequence_data[j+sequence_display_begin,i,:,:],cmap='gray')
        axs[i,j].axes.get_xaxis().set_visible(False)
        axs[i,j].axes.get_yaxis().set_visible(False)
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = -0.8, wspace = 0)