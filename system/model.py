#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import pickle
from typing import Optional

import face_recognition
import yaml
from PIL import Image
from omegaconf import OmegaConf
from sklearn import svm


class Model:
    """
    Model for embedding retrieval. Wrapper for face_recognition
    """

    def __init__(self, model_config_path: Optional[str] = None):
        """
        Load model from pickle files or create new model.
        :param model_config_path: Path to model configuration with encodings_path, names_path, svm_pickle_path.
        """
        if model_config_path:
            self.config = OmegaConf.load(model_config_path)
            encodings_path = self.config.get("encodings_path")
            names_path = self.config.get("names_path")
            svm_path = self.config.get("svm_pickle_path")

            self.encodings = pickle.load(open(encodings_path, "rb"))
            self.names = pickle.load(open(names_path, "rb"))
            self.clf_svm = pickle.load(open(svm_path, "rb"))

        else:
            self.encodings = []
            self.names = []
            self.clf_svm = svm.SVC()

    def _image_preprocessing(self, image: Image):
        # TODO: I dont know if this is even needed.
        return NotImplementedError

    def _train_embeddings_names(self, train_dir_path: str):
        """
        Method to fit the biometric model. Copy from example face_recognition_svm.py
        """
        # The training data would be all the face encodings from all the known images and the labels are their names

        for person in os.listdir(train_dir_path):
            pix = os.listdir(train_dir_path + "/" + person)

            # Loop through each training image for the current person
            for person_img in pix:
                # Get the face encodings for the face in each image file
                face = face_recognition.load_image_file(
                    train_dir_path + "/" + person + "/" + person_img
                )
                face_bounding_boxes = face_recognition.face_locations(face)

                # If training image contains exactly one face
                if len(face_bounding_boxes) == 1:
                    face_enc = face_recognition.face_encodings(face)[0]
                    # Add face encoding for current image with corresponding label (name) to the training data
                    self.encodings.append(face_enc)
                    self.names.append(person)
                else:
                    print(
                        person
                        + "/"
                        + person_img
                        + " was skipped and can't be used for training"
                    )

    def save_system(self, save_dir: str, system_configuration_name: str = "system"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        paths_set = {
            "svm_pickle_path": os.path.join(save_dir, "svm.pkl"),
            "encodings_path": os.path.join(save_dir, "encodings.pkl"),
            "names_path": os.path.join(save_dir, "names.pkl"),
        }
        with open(os.path.join(save_dir, "svm.pkl"), "wb") as file:
            pickle.dump(self.clf_svm, file)
        with open(os.path.join(save_dir, "encodings.pkl"), "wb") as file:
            pickle.dump(self.encodings, file)
        with open(os.path.join(save_dir, "names.pkl"), "wb") as file:
            pickle.dump(self.names, file)
        print(f"System saved to directory: {save_dir}")
        with open(f"config/{system_configuration_name}.yaml", "w") as config_file:
            yaml.dump(paths_set, config_file, sort_keys=True)
        print(
            f"Config file created and saved, filename: config/{system_configuration_name}.yaml"
        )

    def fit_svm(
        self,
        train_dir_path: str,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        verbose=True,
    ) -> svm.SVC:

        self._train_embeddings_names(train_dir_path)

        self.clf_svm = svm.SVC(
            C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, verbose=verbose
        )
        self.clf_svm.fit(self.encodings, self.names)

        return self.clf_svm

    def add_user(self, name: str, image_path: str):
        """
        Method to add user.
        :param name: Name of the user
        :param image_path: Path to the User's image.
        """
        img = face_recognition.load_image_file(image_path)
        face_enc = face_recognition.face_encodings(img)[0]
        self.encodings.append(face_enc)
        self.names.append(name)
        # TODO: maybe it is needed to retrain SVM ???
        # TODO: create new instance of model and retrain it? Very long xD
        self.clf_svm = svm.SVC(
            **self.clf_svm.get_params()
        )  # create new instance with the same configuration
        self.clf_svm.fit(self.encodings, self.names)  # TODO: is it the only option?

    def verify_user(self, image_path: str):
        """
        Method to verify user's image.
        :param image_path: Path to the User's image.
        :return: Name of the user.
        """
        img = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(img)
        name = self.clf_svm.predict([encoding])

        return name
