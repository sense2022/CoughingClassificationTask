import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=False,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None,
                          x_title=None,
                          save_path=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Arguments
    ---------
    cf:            confusion matrix to be passed in

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    normalize:     If True, show the proportions for each category. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.

    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        # group_labels = ["{}\n".format(value) for value in group_names]
        group_labels = []
        print(len(cf))
        for i in range(len(cf)):
            for j in range(len(cf)):
                group_labels.append(cf[j])
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        
        # group_percentages = ["{0:.1%}".format(value) for value in cf.flatten()/10]
        group_percentages = []
        for row in cf:
            total = row.sum()
            for num in row:
                value = num / total
                group_percentages.append("{0:.1%}".format(value))
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    ax = sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories, annot_kws={"size": 58, "weight": "bold"})
    # cbar = ax.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=20)
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 30, fontsize=60, fontweight="bold")
    ax.set_yticklabels(ax.get_yticklabels(),rotation = 30, fontsize=60, fontweight="bold")

    if xyplotlabels:
        plt.ylabel('True', fontsize=60, fontweight="bold")
        plt.xlabel(x_title, fontsize=60, fontweight="bold")
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title, fontsize=60, fontweight="bold")
    plt.tight_layout()
    plt.rcParams["font.weight"] = "bold"
    plt.savefig(save_path)


cf = [[60,0,0,0,0], 
       [0, 60, 0, 0, 0],
       [0,0,60,0,0],
       [0,0,0,60,0],
       [0,0,0,0,60]]

title = 'Accuracy of Fold 1'
saved_path = './volunteers/Fold1.png'

acc = np.array([cf[x][x] for x in range(len(cf))]).sum() / np.array(cf).sum()
x_title = 'Prediction (Accuracy=%.2f)'% acc
cf = np.array(cf)
print(cf)
print(cf.size)

categories = ['Coughing', 'Laughing', 'Throat\nCleaning', 'Speaking', 'Walking']
make_confusion_matrix(cf,
                          group_names=None,
                          categories=categories,
                          count=True,
                          percent=True,
                          cbar=False,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=(21,17),
                          cmap='Blues',
                          title=title,
                          x_title = x_title,
                          save_path=saved_path)