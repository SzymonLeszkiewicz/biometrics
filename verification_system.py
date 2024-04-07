import os
from pathlib import Path

import pandas as pd
from deepface import DeepFace
from tqdm.autonotebook import tqdm

"""
Current database format
data/
    database/
            authorized_users/
                            /1
                                img1.jpg
                                img2.jpg
                            ...
                            /100
                                img1.jpg
                                img2.jpg
            incoming_users/
                            authorized_users/
                                            /1
                                                img1.jpg
                                                img2.jpg
                                            ...
                                            /50
                                                img1.jpg
                                                img2.jpg
                            unauthorized_users/
                                            /101
                                                img1.jpg
                                                img2.jpg
                                            ...
                                            /150
                                                img1.jpg
                                                img2.jpg
"""


class VerificationSystem:
    def __init__(self, database_path: str, acceptance_threshold: float = 0.5, model_name: str = "Facenet"):
        self.database_path = database_path
        self.acceptance_threshold = acceptance_threshold
        self.model_name = model_name
        self.initialize_mutiple_databases()

    def initialize_database(self, destination) -> None:
        DeepFace.find(
            img_path=self.get_incoming_authorized_user_path(),
            db_path=os.path.join(self.database_path, destination),
            threshold=self.acceptance_threshold,
            enforce_detection=False,
            model_name=self.model_name,
        )

    def initialize_mutiple_databases(self) -> None:
        databases = os.listdir(os.path.join(self.database_path))
        for db in databases:
            if db in [".DS_Store", "incoming_users"]:
                continue
            print(db)
            self.initialize_database(db)

    def verify_user(self, user_name: str, user_photo_path: str, destination: str = "authorized_users") -> bool:
        faces_found = DeepFace.find(
            img_path=user_photo_path,
            db_path=os.path.join(self.database_path, destination),
            threshold=self.acceptance_threshold,
            enforce_detection=False,
            silent=True,
            model_name=self.model_name,
        )

        # no face detected or above acceptance threshold
        if faces_found[0].empty:
            return False

        # TODO: find a way to make it path independent
        # assumption that only one face is in the image
        predicted_user_name = faces_found[0]["identity"].apply(
            lambda x: Path(x).parts[3]
        )[
            0
        ]  # get the distance closest match

        is_access_granted = user_name == predicted_user_name

        return is_access_granted

    def verify_multiple_users(self, incoming_users_path: str, destination: str = "authorized_users") -> pd.DataFrame:
        df_users = pd.DataFrame(
            columns=[
                "image_path",
                "is_access_granted",
            ]
        )

        for user_name in os.listdir(incoming_users_path):
            for user_photo in os.listdir(os.path.join(incoming_users_path, user_name)):
                is_access_granted = self.verify_user(
                    user_name=user_name,
                    user_photo_path=os.path.join(
                        incoming_users_path, user_name, user_photo
                    ),
                    destination=destination,
                )

                df_user = pd.DataFrame(
                    {
                        "image_path": os.path.join(
                            incoming_users_path, user_name, user_photo
                        ),
                        "is_access_granted": is_access_granted,
                    },
                    index=[0],
                )

                df_users = pd.concat([df_users, df_user], ignore_index=True)

        return df_users

    @staticmethod
    def calculate_access_granted_rate(
            df_users: pd.DataFrame,
    ) -> float:
        return df_users["is_access_granted"].sum() / len(df_users)

    def get_incoming_authorized_user_path(self) -> str:
        return os.path.join(
            self.database_path, "incoming_users", "authorized_users", "25", "010802.jpg"
        )

    def get_incoming_unauthorized_user_path(self):
        return os.path.join(
            self.data,
            base_path,
            "incoming_users",
            "unauthorized_users",
            "69",
            "032251.jpg",
        )

        def get_problematic_incoming_authorized_user_path(self):
            return os.path.join(
                self.database_path, "incoming_users", "authorized_users", "22", "001677.jpg"
            )
