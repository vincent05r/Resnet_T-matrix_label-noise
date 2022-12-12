import numpy as np

def norm(T):
    row_sum = np.sum(T, 1)
    T_norm = T / row_sum
    return T_norm

def error(T, T_true):
    error = np.sum(np.abs(T-T_true)) / np.sum(np.abs(T_true))
    return error




def find_transition_matrix(M, num_classes, filter_outlier=True):

    T = np.empty((num_classes, num_classes))
    temp_M = M #becareful if it actually copies the matrix.


    for i in np.arange(num_classes):

        if filter_outlier == False:
            anchor_index = np.argmax(temp_M[: , i])
        else:
            eta_thresh = np.percentile(temp_M[:, i], 97,interpolation='higher')
            robust_eta = temp_M[:, i]
            robust_eta[robust_eta >= eta_thresh] = 0.0
            anchor_index = np.argmax(robust_eta)
        for j in np.arange(num_classes):
            T[i, j] = temp_M[anchor_index, j]
    

    return T





#load training data and split them according to corresponding random seed
def dataset_split(train_images, train_labels, split_per=0.9, random_seed = 5):

    #debug
    print()
    print("shape checking before data split")
    print(train_images.shape)
    print(train_labels.shape)
    print()

    num_samples = int(train_labels.shape[0])
    np.random.seed(random_seed)
    train_set_index = np.random.choice(num_samples, int(num_samples*split_per), replace=False)
    index = np.arange(train_images.shape[0])
    val_set_index = np.delete(index, train_set_index)

    train_set, val_set = train_images[train_set_index, :], train_images[val_set_index, :]
    train_labels_mod, val_labels = train_labels[train_set_index], train_labels[val_set_index]

    
    #debug
    print("\nshape checking after data split")
    print("train_set" + str(train_set.shape))
    print("val_set" + str(val_set.shape))
    print("train_labels_mod" + str(train_labels_mod.shape))
    print("val_labels" + str(val_labels.shape) + '\n')


    print("\n\n data composition check for split")
    t0 = 0
    t1 = 0
    t2 = 0
    for i in train_labels_mod:
        if i == 0:
            t0 += 1

        elif i == 1:
            t1 += 1

        elif i == 2:
            t2 += 1
    
    v0 = 0
    v1 = 0
    v2 = 0
    for i in val_labels:
        if i == 0:
            v0 += 1

        elif i == 1:
            v1 += 1

        elif i == 2:
            v2 += 1


    print("Totol len of training data is : {} Distribution in training data is {}  {}  {} ".format(len(train_labels_mod), t0, t1, t2))

    print("Totol len of val data is : {} Distribution in val data is {}  {}  {} ".format(len(val_labels), v0, v1, v2))

    return train_set, val_set, train_labels_mod, val_labels

