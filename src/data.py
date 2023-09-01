# code to load data for indexing using the Catlin Seaview annotation format (csv + images in directory)

import pandas as pd
from PIL import Image
from tensorflow.keras.utils import Sequence, img_to_array

import numpy as np
import os
import matplotlib.pyplot as plt

from datetime import datetime
import pickle

from utils import load_pickle
from sklearn import preprocessing

class IdxConfig:
    def __init__(self, training_dir, input_dir, output_dir='../data/output_data/indexed_data', patch_size=256,
                 batch_size=16, model='b0'):
        self.csv_path = os.path.join(input_dir, 'annotations_PAC_AUS.csv')
        self.image_dir = os.path.join(input_dir, 'PAC_AUS/PAC_AUS')
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.mean_image_path = os.path.join(training_dir, 'mean_image.jpg')
        self.model_path = os.path.join(training_dir, 'model', model + '.weights.best.hdf5')
        self.model = model

        self.output_dir = output_dir
        self.output_csv_path = self.create_output_csv_path()

    def create_output_csv_path(self):
        dt = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        outpth = os.path.join(self.output_dir, "indexing_results_" + self.model + '_' + dt + ".csv")
        return outpth


class IdxDatagen:
    def __init__(self, CFG, crop_size=None):  # csv_path, image_dir, patch_size, mean_image_path, batch_size):
        self.csv_path = CFG.csv_path
        self.image_dir = CFG.image_dir

        self.patch_size = CFG.patch_size

        if crop_size==None:
            self.crop_size = self.patch_size
        else:
            self.crop_size = crop_size


        self.mean_image = self.load_mean_image(CFG.mean_image_path)
        self.batch_size = CFG.batch_size
        self.X, self.point_ids = self.get_data(pd.read_csv(CFG.csv_path),
                                               image_path_key="quadratid",
                                               point_x_key="x",
                                               point_y_key="y")
        # self.gen = self.make_generator(self.X, self.ids)

    # generate a list of image_dicts from the input csv to pass to the cropping_functions. Also generate a list of
    # ids for each point
    def get_data(self, df, image_path_key, point_x_key, point_y_key):
        X = []
        ids = df[image_path_key].to_list()
        for index, row in df.iterrows():
            X.append({"quadratid": row[image_path_key],
                      "point_x": row[point_x_key],
                      "point_y": row[point_y_key]})

        return X, ids

    def load_mean_image(self, mean_image_path):
        mean_image = Image.open(mean_image_path)
        return mean_image

    def preprocess_img(self, img):
        img -= self.mean_image

        # normalise for faster training times
        img /= 255.0
        return img

    def check_cache(self, image_path, x, y):

        base_path = "../data/thumbnail-cache-frontiers/"
        image_key = os.path.join( base_path, os.path.basename(image_path), str(x), str(y), ".jpg" )

        if os.path.exists(image_key):
            return Image.open(image_key)

        return None

    def save_to_cache(self, image_path, x, y, image):
        base_path = "../data/thumbnail-cache-frontiers/"
        image_key = os.path.join(base_path, os.path.basename(image_path), str(x), str(y), ".jpg")

        print("saving to cache")
        image.save(image_key)

    def cropping_function(self, image_dict):
        images_dir = self.image_dir
        crop_size = self.crop_size
        patch_size = self.patch_size
        # print("cropping function")
        image_path = os.path.join(images_dir, str(image_dict["quadratid"]) + '.jpg')

        patch = None #self.check_cache(image_path, image_dict["point_x"], image_dict["point_y"])

        if patch is None:
            patch = load_image_and_crop(image_path, image_dict["point_x"], image_dict["point_y"], crop_size, patch_size)
            #self.save_to_cache(image_path, image_dict["point_x"], image_dict["point_y"], patch)

        patch = img_to_array(patch)
        patch = self.preprocess_img(patch)
        return patch

    def make_generator(self):
        X = self.X
        point_ids = self.point_ids
        datagen = FullImagePointCroppingLoader_Index(X,
                                                     self.batch_size,
                                                     self.cropping_function,
                                                     point_ids)

        return datagen


class InfConfig:
    def __init__(self,
                 training_dir,
                 input_dir,
                 datasplit_dir='../data/output_data/PAC_AUS_training_data/train-data_2022-03-25_10-58-08/data-fixed_labels.p',
                 patch_size=256, batch_size=16, model='b0'):

        self.csv_path =  datasplit_dir #os.path.join(input_dir, 'annotations_PAC_AUS.csv')
        self.image_dir = os.path.join(input_dir, 'PAC_AUS/PAC_AUS')
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.mean_image_path = os.path.join(training_dir, 'mean_image.jpg')
        self.model_path = os.path.join(training_dir, 'model', model + '.weights.best.hdf5')
        self.class_list = load_pickle(os.path.join(training_dir, 'classes.p'))
        self.output_csv_path = os.path.join(training_dir, "retraining_test_set_predictions_1111.csv")

class InfDatagen:
    def __init__(self, CFG):
        self.csv_path = CFG.csv_path
        self.df = load_pickle(self.csv_path).df

        self.df =  self.df[self.df["set"]=="test"].reset_index().loc[:300]

        self.image_dir = CFG.image_dir
        self.patch_size = CFG.patch_size
        self.mean_image = self.load_mean_image(CFG.mean_image_path)
        self.batch_size = CFG.batch_size

        self.X, self.point_ids = self.get_data(self.df,
                               image_path_key="quadratid",
                               point_x_key="x",
                               point_y_key="y",
                               set="test")

   # generate a list of image_dicts from the input csv to pass to the cropping_functions. Also generate a list of
    # ids for each point
    def get_data(self, df, image_path_key, point_x_key, point_y_key, set, label_key="label_name"):
        set_df = df[df["set"] == set]
        # print(set_df.head())
        # print(len(set_df[label_key].unique()))
        X = []
        y = []
        ids = set_df[image_path_key].to_list()
        for index, row in set_df.iterrows():
            X.append({"quadratid": row[image_path_key],
                      "point_x": row[point_x_key],
                      "point_y": row[point_y_key]
                      })

        return X, ids

    def load_mean_image(self, mean_image_path):
        mean_image = Image.open(mean_image_path)
        return mean_image

    def preprocess_img(self, img):
        img -= self.mean_image

        # normalise for faster training times
        img /= 255.0
        return img

    # def check_cache(self, image_path, x, y):
    #
    #     base_path = "../data/thumbnail-cache-frontiers/"
    #     image_key = os.path.join( base_path, os.path.basename(image_path), x, y, ".jpg" )
    #
    #     if os.path.exists(image_key):
    #         return Image.open(image_key)
    #
    #     return None
    #
    # def save_to_cache(self, image_path, x, y, image):
    #     base_path = "../data/thumbnail-cache-frontiers/"
    #     image_key = os.path.join(base_path, os.path.basename(image_path), x, y, ".jpg")
    #
    #     print("saving to cache")
    #     image.save(image_key)

    def cropping_function(self, image_dict):
        images_dir = self.image_dir
        crop_size = self.patch_size
        patch_size = self.patch_size
        # print("cropping function")
        image_path = os.path.join(images_dir, str(image_dict["quadratid"]) + '.jpg')

        patch = None #self.check_cache(image_path, image_dict["point_x"], image_dict["point_y"])

        if patch is None:
            patch = load_image_and_crop(image_path, image_dict["point_x"], image_dict["point_y"], crop_size, patch_size)
            # self.save_to_cache(image_path, image_dict["point_x"], image_dict["point_y"], patch)

        patch = img_to_array(patch)
        patch = self.preprocess_img(patch)
        return patch

    def make_generator(self):
        X = self.X
        point_ids = self.point_ids
        datagen = FullImagePointCroppingLoader_Index(X,
                                                     self.batch_size,
                                                     self.cropping_function,
                                                     point_ids)

        return datagen

class RetrainingConfig:
    def __init__(self,
                 training_dir,
                 input_dir,
                 datasplit_dir='../data/output_data/PAC_AUS_training_data/train-data_2022-03-25_10-58-08/data-fixed_labels.p',
                 patch_size=256, batch_size=16, model='b0'):

        self.csv_path =  datasplit_dir #os.path.join(input_dir, 'annotations_PAC_AUS.csv')
        self.image_dir = os.path.join(input_dir, 'PAC_AUS/PAC_AUS')
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.mean_image_path = os.path.join(training_dir, 'mean_image.jpg')
        self.model_path = os.path.join(training_dir, 'model', model + '.weights.best.hdf5')



class RetrainingDatagen:
    def __init__(self, CFG):  # csv_path, image_dir, patch_size, mean_image_path, batch_size):
        self.csv_path = CFG.csv_path
        self.df = load_pickle(self.csv_path).df
        self.image_dir = CFG.image_dir
        self.patch_size = CFG.patch_size
        self.mean_image = self.load_mean_image(CFG.mean_image_path)
        self.batch_size = CFG.batch_size

        from sklearn.preprocessing import LabelBinarizer
        from sklearn.preprocessing import OneHotEncoder
        le = preprocessing.LabelEncoder()
        self.df["label_id"] = le.fit_transform(self.df.label_name)
        #self.df["label_onehot"] = LabelBinarizer().fit_transform(self.df.label_id)
        #self.df["label_onehot"] = OneHotEncoder().fit_transform(self.df["label_id", "label"]).toarray()

        #self.df["label_onehot"] = np.zeros((186420, len(self.df["label"].unique())), dtype="float16")
        #self.df["label_onehot"] = 0
        #for i, label_index in enumerate(y):
        #    onehot_y[i, label_index] = 1.
        #y = onehot_y
        one_hot = []
        len_labels = len(self.df["label_name"].unique())
        print(len_labels)
        for index, row in self.df.iterrows():
            #temp = row["label_onehot"]
            temp = np.zeros((len_labels), dtype="float16")
            temp[row["label_id"]] = 1.
            #self.df.at[index,'label_onehot'] = temp
            one_hot.append(temp)

        self.df["label_onehot"] = one_hot

        self.X_train, self.y_train = self.get_data(self.df,
                                                       image_path_key="quadratid",
                                                       point_x_key="x",
                                                       point_y_key="y",
                                                       set="train")

        self.X_val, self.y_val = self.get_data(self.df,
                                                   image_path_key="quadratid",
                                                   point_x_key="x",
                                                   point_y_key="y",
                                                   set="val")
        # self.gen = self.make_generator(self.X, self.ids)

    # generate a list of image_dicts from the input csv to pass to the cropping_functions. Also generate a list of
    # ids for each point
    def get_data(self, df, image_path_key, point_x_key, point_y_key, set, label_key="label_name"):
        set_df = df[df["set"] == set]
        print(set_df.head())
        print(len(set_df[label_key].unique()))
        X = []
        y = []
        ids = set_df[image_path_key].to_list()
        for index, row in set_df.iterrows():
            X.append({"quadratid": row[image_path_key],
                      "point_x": row[point_x_key],
                      "point_y": row[point_y_key]
                      })
            y.append(row["label_onehot"])



        print("--------")
        print(y[0])
        print(len(y[0]))
        print("--------")
        return X, y

    def load_mean_image(self, mean_image_path):
        mean_image = Image.open(mean_image_path)
        return mean_image

    def preprocess_img(self, img):
        img -= self.mean_image

        # normalise for faster training times
        img /= 255.0
        return img

    def check_cache(self, image_path, x, y):

        base_path = "../data/thumbnail-cache-frontiers/"
        image_key = os.path.join( base_path, os.path.basename(image_path), str(x), str(y), ".jpg" )

        if os.path.exists(image_key):
            return Image.open(image_key)

        return None

    def save_to_cache(self, image_path, x, y, image):
        base_path = "../data/thumbnail-cache-frontiers/"
        new_name = "%s-%s-%s%s" % (os.path.basename(image_path), str(x), str(y), ".jpg")
        image_key = os.path.join(base_path, new_name)
        #print(image_key)
        #print("saving to cache")
        image.save(image_key, "JPEG")

    def cropping_function(self, image_dict):
        images_dir = self.image_dir
        crop_size = self.patch_size
        patch_size = self.patch_size

        image_path = os.path.join(images_dir, str(image_dict["quadratid"]) + '.jpg')

        patch = self.check_cache(image_path, image_dict["point_x"], image_dict["point_y"])

        if patch is None:
            patch = load_image_and_crop(image_path, image_dict["point_x"], image_dict["point_y"], crop_size, patch_size)
            self.save_to_cache(image_path, image_dict["point_x"], image_dict["point_y"], patch)

        #patch = load_image_and_crop(image_path, image_dict["point_x"], image_dict["point_y"], crop_size, patch_size)
        patch = img_to_array(patch)
        patch = self.preprocess_img(patch)
        return patch

    def make_train_generator(self):
        X = self.X_train
        y = self.y_train

        datagen = FullImagePointCroppingLoader(X,
                                               y,
                                               self.batch_size,
                                               self.cropping_function
                                               )

        return datagen

    def make_val_generator(self):
        X = self.X_val
        y = self.y_val

        datagen = FullImagePointCroppingLoader(X,
                                               y,
                                               self.batch_size,
                                               self.cropping_function
                                               )

        return datagen


def get_rect_dimensions_pixels(patchwidth, patchheight, pointx, pointy):
    return [int((pointx) - (patchwidth / 2)), int((pointy) - (patchheight / 2)),
            int((pointx) + (patchwidth / 2)), int((pointy) + (patchheight / 2))]


def cut_patch(image, patch_width, patch_height, x, y):
    dimensions = get_rect_dimensions_pixels(patch_width, patch_height, x, y)
    new_image = image.crop(dimensions)

    return new_image


def load_image_and_crop(image_path, point_x, point_y, crop_size, final_patch_size):
    img = Image.open(image_path)
    img = cut_patch(img, crop_size, crop_size, point_x, point_y)
    #img.save("/home/mat/Dev/reefcloud-tl-frontiers/data/sampleimg/%s.jpg" % (os.path.basename(image_path)+"-"+str(point_x)+"-"+str(point_y)))
    if not crop_size == final_patch_size:
        img = img.resize((final_patch_size, final_patch_size), Image.NEAREST)
    return img


class FullImagePointCroppingLoader_Index(Sequence):

    def __init__(self, x_set, batch_size, load_image_func, point_ids):
        self.x = x_set
        self.batch_size = batch_size
        self.load_image_func = load_image_func
        self.ids = point_ids

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]

        indexes = []
        for index in range(idx * self.batch_size, (idx + 1) * self.batch_size):
            indexes.append(index)

        return np.array([
            self.load_image_func(dict_item)
            for dict_item in batch_x])


class FullImagePointCroppingLoader(Sequence):

    def __init__(self, x_set, y_set, batch_size, load_image_func):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.load_image_func = load_image_func

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        indexes = []
        for index in range(idx * self.batch_size, (idx + 1) * self.batch_size):
            indexes.append(index)

        X = np.array([self.load_image_func(dict_item) for dict_item in batch_x])
        y = np.array(batch_y)
        #print("returning x/y")
        return X, y


class trainingData:

    def __init__(self,
                 df,
                 vectors_pth='../data/output_data/indexed_data/indexing_results_b0_2022-03-09_16-44-00.csv',
                 save_dir='../data/output_data/PAC_AUS_training_data/'):  # , labels_df, label_col="label_name", save_model=True):
        """
        Sets up data for training the classifier model. Generates a training directory with train/val/test splits
        :param df: pandas DataFrame, loaded from Catlin annotations csv
        :param vectors_pth: path to indexing results matching the annotations file
        """
        self.df = df
        self.vectors_pth = vectors_pth
        self.save_dir = save_dir

        self.train_val_test()
        self.create_train_dir()

    def train_val_test(self):
        """ Split data into training, validation and test sets. Evenly sample across surveys, where
        surveyid= quadratid[:5]. Keeps all points from an image in the same set """

        print()
        imsdf = pd.DataFrame(columns=["quadratid", "surveyid"])
        imsdf["quadratid"] = self.df["quadratid"].unique()
        imsdf["surveyid"] = imsdf["quadratid"].apply(lambda x: str(x)[:5])

        imsdf_train = imsdf.groupby("surveyid").sample(frac=0.6)

        non_train = imsdf.drop(imsdf_train.index)
        imsdf_test = non_train.groupby("surveyid").sample(frac=0.5)
        imsdf_val = non_train.drop(imsdf_test.index)

        trainims = imsdf_train["quadratid"].unique()
        testims = imsdf_test["quadratid"].unique()
        valims = imsdf_val["quadratid"].unique()

        # add column for set in data

        self.df.loc[self.df["quadratid"].isin(trainims), "set"] = "train"
        self.df.loc[self.df["quadratid"].isin(testims), "set"] = "test"
        self.df.loc[self.df["quadratid"].isin(valims), "set"] = "val"

        self.df = self.df.loc[:, ['quadratid', 'y', 'x', 'label_name', 'label', 'func_group', 'method', 'set']]

    def create_train_dir(self):
        dt = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        outpth = os.path.join(self.save_dir, "train-data_{}".format(dt))
        os.mkdir(outpth)

        pickle.dump(self, open(os.path.join(outpth, "data.p"), "wb"))


if __name__ == "__main__":
    df = pd.read_csv('../data/input_data/annotations_PAC_AUS.csv')
    print(df.columns)
    traindataobj = trainingData(df)
    traindataobj.train_val_test()
    print(traindataobj.df.columns)

    a = 10
