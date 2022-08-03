import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, epsilon_history, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilon_history, label="Scores", color="red")
    ax.set_xlabel("Training Steps", color="red")
    ax.set_ylabel("Epsilon", color="red")
    ax.tick_params(axis='x', colors="red")
    ax.tick_params(axis='y', colors="red")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-100):(t+1)])
    
    ax2.scatter(x, running_avg, color="blue", label="Scores")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="blue")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="blue")
    
    plt.savefig(filename)
    plt.close()
    plt.show()