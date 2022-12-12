import numpy as np
# import data_load


# train set and val set split

# def dataset_split(train_images, train_labels, split_per=0.9, random_seed = 5):

#     print()
#     print("shape checking")
#     print(train_images.shape)
#     print(train_labels.shape)
#     print()

#     num_samples = int(train_labels.shape[0])
#     np.random.seed(random_seed)
#     train_set_index = np.random.choice(num_samples, int(num_samples*split_per), replace=False)
#     index = np.arange(train_images.shape[0])
#     val_set_index = np.delete(index, train_set_index)

#     train_set, val_set = train_images[train_set_index, :], train_images[val_set_index, :]
#     train_labels_mod, val_labels = train_labels[train_set_index], train_labels[val_set_index]

#     return train_set, val_set, train_labels_mod, val_labels