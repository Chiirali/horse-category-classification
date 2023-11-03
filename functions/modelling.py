from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import pandas as pd
from pandas import DataFrame
import seaborn as sns

def recompile_model(base_model, optimizer, loss):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(3, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model
    
    
def extract_labels_categorical(dataset):
    import numpy as np

    test_ls = list(dataset.as_numpy_iterator())
    true_labels = []
    for batch in test_ls:
        for batchitem in batch[1].tolist():
            true_labels.append(np.argmax(batchitem))
    return true_labels
    
    
def plot_cm(confusion_matrix, class_names):
    import pandas as pd
    import seaborn as sn
    import matplotlib.pyplot as plt

    df_cm = pd.DataFrame(confusion_matrix, columns=class_names, index = class_names)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (10,7))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, cmap="Reds", annot=True, annot_kws={"size": 12}, fmt='g')