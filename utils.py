import numpy as np
import matplotlib.pyplot as plt 

def label_with_highest_prob(probabilities):
  return np.argmax(probabilities, axis = 1)

def visualize_points(X, y):
  plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
  plt.show()

def generate_data_points(number_of_classes, points_per_class):
  N = points_per_class
  K = number_of_classes
  X = np.zeros((N*K,2)) # data matrix (each row = single example)
  y = np.zeros(N*K, dtype='uint8') # class labels
  for j in np.arange(K):
    ix = range(N*j,N*(j+1))
    r = np.linspace(0.0,1,N) # radius
    t = np.linspace(j*(K+1),(j+1)*(K+1),N) + np.random.randn(N)*0.2 # theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j
  return X, y

def plot_decision_boundary(X, y, model, steps=100, cmap='Paired'):
    """
    Function to plot the decision boundary and data points of a model.
    Data points are colored based on their actual label.
    """

    # Define region of interest by data limits
    x_span = np.linspace(-1, 1, steps)
    y_span = np.linspace(-1, 1, steps)
    xx, yy = np.meshgrid(x_span, y_span)

    # Make predictions across region of interest
    labels = np.argmax(model.predict(np.c_[xx.ravel(), yy.ravel()]), axis=1)

    # Plot decision boundary in region of interest
    z = labels.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, z, cmap=plt.cm.Spectral, alpha=0.25)

    # Get predicted labels on training data and plot
    train_labels = model.predict(X)
    ax.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Spectral, lw=0)

    return fig, ax
  
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
def plot_confusion_matrix(y_true, y_pred, classes=None,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Confusion matrix'
        else:
            title = 'Confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    unique = unique_labels(y_true, y_pred)
    if(classes is None):
      classes = unique
    else:
      classes = classes[unique]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Confusion matrix")
    else:
        print('Confusion matrix')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def plot_accuracy_and_loss(history):
  # Plot training & validation accuracy values
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Validation'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Validation'], loc='upper left')
  plt.show()

  
def draw_model(model, view=True, filename="network.gv", title="Neural Network"):
    """Vizualizes a Sequential model.
    # Arguments
        model: A Keras model instance.
        view: whether to display the model after generation.
        filename: where to save the vizualization. (a .gv file)
        title: A title for the graph
    """
    from graphviz import Digraph;
    import tensorflow as tf;
    import tensorflow.keras as keras;
    from tensorflow.keras.models import Sequential;
    from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten;
    import json;
    input_layer = 0;
    hidden_layers_nr = 0;
    layer_types = [];
    hidden_layers = [];
    output_layer = 0;
    for layer in model.layers:
        if(layer == model.layers[0]):
            input_layer = int(str(layer.input_shape).split(",")[1][1:-1]);
            hidden_layers_nr += 1;
            if (type(layer) == keras.layers.Dense):
                hidden_layers.append(int(str(layer.output_shape).split(",")[1][1:-1]));
                layer_types.append("Dense");
            else:
                hidden_layers.append(1);
                if (type(layer) == keras.layers.Conv2D):
                    layer_types.append("Conv2D");
                elif (type(layer) == keras.layers.MaxPooling2D):
                    layer_types.append("MaxPooling2D");
                elif (type(layer) == keras.layers.Dropout):
                    layer_types.append("Dropout");
                elif (type(layer) == keras.layers.Flatten):
                    layer_types.append("Flatten");
                elif (type(layer) == keras.layers.Activation):
                    layer_types.append("Activation");
        else:
            if(layer == model.layers[-1]):
                output_layer = int(str(layer.output_shape).split(",")[1][1:-1]);
            else:
                hidden_layers_nr += 1;
                if (type(layer) == keras.layers.Dense):
                    hidden_layers.append(int(str(layer.output_shape).split(",")[1][1:-1]));
                    layer_types.append("Dense");
                else:
                    hidden_layers.append(1);
                    if (type(layer) == keras.layers.Conv2D):
                        layer_types.append("Conv2D");
                    elif (type(layer) == keras.layers.MaxPooling2D):
                        layer_types.append("MaxPooling2D");
                    elif (type(layer) == keras.layers.Dropout):
                        layer_types.append("Dropout");
                    elif (type(layer) == keras.layers.Flatten):
                        layer_types.append("Flatten");
                    elif (type(layer) == keras.layers.Activation):
                        layer_types.append("Activation");
        last_layer_nodes = input_layer;
        nodes_up = input_layer;
        if(type(model.layers[0]) != keras.layers.Dense):
            last_layer_nodes = 1;
            nodes_up = 1;
            input_layer = 1;

        g = Digraph('g', filename=filename);
        n = 0;
        g.graph_attr.update(splines="false", nodesep='1', ranksep='2');
        #Input Layer
        with g.subgraph(name='cluster_input') as c:
            if(type(model.layers[0]) == keras.layers.Dense):
                the_label = title+'\n\n\n\nInput Layer';
                if (int(str(model.layers[0].input_shape).split(",")[1][1:-1]) > 10):
                    the_label += " (+"+str(int(str(model.layers[0].input_shape).split(",")[1][1:-1]) - 10)+")";
                    input_layer = 10;
                c.attr(color='white')
                for i in range(0, input_layer):
                    n += 1;
                    c.node(str(n));
                    c.attr(label=the_label)
                    c.attr(rank='same');
                    c.node_attr.update(color="#2ecc71", style="filled", fontcolor="#2ecc71", shape="circle");
                    
            elif(type(model.layers[0]) == keras.layers.Flatten):
                #Conv2D Input visualizing
                the_label = title+'\n\n\n\nInput Layer';
                c.attr(color="white", label=the_label);
                c.node_attr.update(shape="square");
                pxls = str(model.layers[0].input_shape).split(',');
                clr = 1
                if(len(pxls) >3):
                  clr = int(pxls[3][1:-1]);
                if (clr == 1):
                    clrmap = "Grayscale";
                    the_color = "black:white";
                elif (clr == 3):
                    clrmap = "RGB";
                    the_color = "#e74c3c:#3498db";
                else:
                    clrmap = "";
                c.node_attr.update(fontcolor="white", fillcolor=the_color, style="filled");
                n += 1;
                c.node(str(n), label="Image\n"+pxls[1]+" x"+pxls[2]+" pixels\n"+clrmap, fontcolor="white");

            elif(type(model.layers[0]) == keras.layers.Conv2D):
                #Conv2D Input visualizing
                the_label = title+'\n\n\n\nInput Layer';
                c.attr(color="white", label=the_label);
                c.node_attr.update(shape="square");
                pxls = str(model.layers[0].input_shape).split(',');
                clr = int(pxls[3][1:-1]);
                if (clr == 1):
                    clrmap = "Grayscale";
                    the_color = "black:white";
                elif (clr == 3):
                    clrmap = "RGB";
                    the_color = "#e74c3c:#3498db";
                else:
                    clrmap = "";
                c.node_attr.update(fontcolor="white", fillcolor=the_color, style="filled");
                n += 1;
                c.node(str(n), label="Image\n"+pxls[1]+" x"+pxls[2]+" pixels\n"+clrmap, fontcolor="white");
            else:
                raise ValueError("ANN Visualizer: Layer not supported for visualizing");
        for i in range(0, hidden_layers_nr):
            with g.subgraph(name="cluster_"+str(i+1)) as c:
                if (layer_types[i] == "Dense"):
                    c.attr(color='white');
                    c.attr(rank='same');
                    #If hidden_layers[i] > 10, dont include all
                    the_label = "";
                    if (int(str(model.layers[i].output_shape).split(",")[1][1:-1]) > 10):
                        the_label += " (+"+str(int(str(model.layers[i].output_shape).split(",")[1][1:-1]) - 10)+")";
                        hidden_layers[i] = 10;
                    c.attr(labeljust="right", labelloc="b", label=the_label);
                    for j in range(0, hidden_layers[i]):
                        n += 1;
                        c.node(str(n), shape="circle", style="filled", color="#3498db", fontcolor="#3498db");
                        for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                            g.edge(str(h), str(n));
                    last_layer_nodes = hidden_layers[i];
                    nodes_up += hidden_layers[i];
                elif (layer_types[i] == "Conv2D"):
                    c.attr(style='filled', color='#5faad0');
                    n += 1;
                    kernel_size = str(model.layers[i].get_config()['kernel_size']).split(',')[0][1] + "x" + str(model.layers[i].get_config()['kernel_size']).split(',')[1][1 : -1];
                    filters = str(model.layers[i].get_config()['filters']);
                    c.node("conv_"+str(n), label="Convolutional Layer\nKernel Size: "+kernel_size+"\nFilters: "+filters, shape="square");
                    c.node(str(n), label=filters+"\nFeature Maps", shape="square");
                    g.edge("conv_"+str(n), str(n));
                    for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                        g.edge(str(h), "conv_"+str(n));
                    last_layer_nodes = 1;
                    nodes_up += 1;
                elif (layer_types[i] == "MaxPooling2D"):
                    c.attr(color="white");
                    n += 1;
                    pool_size = str(model.layers[i].get_config()['pool_size']).split(',')[0][1] + "x" + str(model.layers[i].get_config()['pool_size']).split(',')[1][1 : -1];
                    c.node(str(n), label="Max Pooling\nPool Size: "+pool_size, style="filled", fillcolor="#8e44ad", fontcolor="white");
                    for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                        g.edge(str(h), str(n));
                    last_layer_nodes = 1;
                    nodes_up += 1;
                elif (layer_types[i] == "Flatten"):
                    n += 1;
                    c.attr(color="white");
                    c.node(str(n), label="Flattening", shape="invtriangle", style="filled", fillcolor="#2c3e50", fontcolor="white");
                    for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                        g.edge(str(h), str(n));
                    last_layer_nodes = 1;
                    nodes_up += 1;
                elif (layer_types[i] == "Dropout"):
                    n += 1;
                    c.attr(color="white");
                    c.node(str(n), label="Dropout Layer", style="filled", fontcolor="white", fillcolor="#f39c12");
                    for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                        g.edge(str(h), str(n));
                    last_layer_nodes = 1;
                    nodes_up += 1;
                elif (layer_types[i] == "Activation"):
                    n += 1;
                    c.attr(color="white");
                    fnc = model.layers[i].get_config()['activation'];
                    c.node(str(n), shape="octagon", label="Activation Layer\nFunction: "+fnc, style="filled", fontcolor="white", fillcolor="#00b894");
                    for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                        g.edge(str(h), str(n));
                    last_layer_nodes = 1;
                    nodes_up += 1;


        with g.subgraph(name='cluster_output') as c:
            if (type(model.layers[-1]) == keras.layers.Dense):
                c.attr(color='white')
                c.attr(rank='same');
                c.attr(labeljust="1");
                for i in range(1, output_layer+1):
                    n += 1;
                    c.node(str(n), shape="circle", style="filled", color="#e74c3c", fontcolor="#e74c3c");
                    for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                        g.edge(str(h), str(n));
                c.attr(label='Output Layer', labelloc="bottom")
                c.node_attr.update(color="#2ecc71", style="filled", fontcolor="#2ecc71", shape="circle");

        g.attr(arrowShape="none");
        g.edge_attr.update(arrowhead="none", color="#707070");
        if view == True:
            g.view();


    if view == True:
        import graphviz

        with open(filename) as f:
            dot_graph = f.read()

        return graphviz.Source(dot_graph)

      
def plot_single_image_correct(i, predictions, true_labels, images, class_names=None, cmap=plt.cm.binary):
  predictions_array, true_label, img = predictions[i], true_labels[i], images[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=cmap)
  
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  class_name = true_label if class_names is None else class_names[true_label]
  class_name_predicted = predicted_label if class_names is None else class_names[predicted_label]
  plt.xlabel("{} {:2.0f}% ({})".format(class_name_predicted,
                                100*np.max(predictions_array),
                                class_name),
                                color=color)

def plot_value_array(i, predictions, true_labels):
  predictions_array, true_label = predictions[i], true_labels[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
  
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

  
def plot_image_and_prob(predictions, test_labels, test_images, class_names=None, i = 0, cmap=plt.cm.binary):
  plt.figure(figsize=(6,3))
  plt.subplot(1,2,1)
  plot_single_image_correct(i, predictions, test_labels, test_images, class_names, cmap=cmap)
  plt.subplot(1,2,2)
  plot_value_array(i, predictions,  test_labels)
  plt.show()

def plot_multi_images_prob(predictions, labels, images, class_names=None, start=0, num_rows=5, num_cols=3, cmap=plt.cm.binary ):
  num_rows = 5
  num_cols = 3
  num_images = num_rows*num_cols
  plt.figure(figsize=(2*2*num_cols, 2*num_rows))
  for i in range(num_images):
    index = i + start
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_single_image_correct(index, predictions, labels, images, class_names, cmap=cmap)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(index, predictions, labels)
  plt.show()

def plot_multi_images(images, labels, class_names=None, start=0, num_rows=5, num_cols=5, cmap=plt.cm.binary):
  plt.figure(figsize=(2*num_cols, 2*num_rows))
  for i in range(num_cols*num_rows):
    index = i + start
    plt.subplot(num_rows,num_cols,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[index], cmap=cmap)
    label = labels[index] if class_names is None else class_names[labels[index]]
    plt.xlabel(label)
  plt.show()


def plot_image(images, labels, class_names = None, index=0, colorbar=True, cmap='Greys'):
  plt.figure()
  plt.imshow(images[index], cmap=cmap) # print the image
  if(colorbar == True):
    plt.colorbar()
  plt.show()
  label = labels[index] if class_names is None else class_names[labels[i]]
  print("the label is:", labels[index]) # The train label
