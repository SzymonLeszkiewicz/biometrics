#!/usr/bin/python
# -*- coding: utf-8 -*-
from PIL import Image
import os
import face_recognition
from sklearn import svm


class Model:
    """
    Model for embedding retrieval. Wrapper for face_recognition
    """

    def __init__(self):
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

    def fit_svm(
        self,
        train_dir_path: str,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
    ) -> svm.SVC:

        self._train_embeddings_names(train_dir_path)

        self.clf_svm = svm.SVC(
            C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0
        )
        self.clf_svm.fit(self.encodings, self.names)

        return self.clf_svm

    def add_user(self, image: Image):
        return NotImplementedError

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
