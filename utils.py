import matplotlib.pyplot as plt
import numpy as np

def plot_scores(scores, scores_means, execution_info):
    
    ## Plot the scores
    fig = plt.figure(figsize=(20,10))
    
    for key in execution_info:
        print(f'{key}: {execution_info[key]}')

    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.plot(np.arange(len(scores_means)), scores_means)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend(('Score', 'Mean'), fontsize='xx-large')

    plt.show()