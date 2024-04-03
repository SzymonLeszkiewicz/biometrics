import os
from pathlib import Path

import pandas as pd
from deepface import DeepFace
from tqdm.autonotebook import tqdm


class VerificationSystem:
    def __init__(self, database_path: str, acceptance_threshold: float = 0.5):
        self.database_path = database_path
        self.acceptance_threshold = acceptance_threshold

        self.initialize_database()

    def initialize_database(self) -> None:
        DeepFace.find(
            img_path=self.get_incoming_authorized_user_path(),
            db_path=self.database_path,
            threshold=self.acceptance_threshold,
            enforce_detection=False,
        )

    def get_incoming_authorized_user_path(self) -> str:
        return os.path.join(
            self.database_path, "incoming_users", "authorized_users", "1", "000023.jpg"
        )

    def get_incoming_unauthorized_user_path(self):
        return os.path.join(
            self.database_path,
            "incoming_users",
            "unauthorized_users",
            "101",
            "020633.jpg",
        )

    def get_problematic_incoming_authorized_user_path(self):
        return os.path.join(
            self.database_path, "incoming_users", "authorized_users", "22", "001677.jpg"
        )

    def calculate_users_verification_matrix(
        self, incoming_users_path: str
    ) -> pd.DataFrame:
        df_users = pd.DataFrame(
            columns=[
                "image_path",
                "truth_label",
                "predicted_label",
                "cosine_distance",
                "is_access_granted",
                "is_matched",
            ]
        )

        for user_name in tqdm(
            iterable=os.listdir(incoming_users_path), desc="Processing users"
        ):
            for person_photo in tqdm(
                iterable=os.listdir(os.path.join(incoming_users_path, user_name)),
                desc="Processing user photos",
                leave=False,
            ):
                faces_found = DeepFace.find(
                    img_path=os.path.join(incoming_users_path, user_name, person_photo),
                    db_path=self.database_path,
                    threshold=self.acceptance_threshold,
                    enforce_detection=False,
                    silent=True,
                )

                # TODO: find a way to make it path independent
                predicted_user_name = faces_found[0]["identity"].apply(
                    lambda x: Path(x).parts[3]
                )[
                    0
                ]  # get the distance closest match

                predicted_cosine_distance = faces_found[0]["distance"]
                is_access_granted = (
                    predicted_cosine_distance < self.acceptance_threshold
                ) * 1
                is_matched = (user_name == predicted_user_name) * 1

                df_user = pd.DataFrame(
                    {
                        "image_path": os.path.join(
                            incoming_users_path, user_name, person_photo
                        ),
                        "truth_label": user_name,
                        "predicted_label": predicted_user_name,
                        "cosine_distance": predicted_cosine_distance,
                        "is_access_granted": is_access_granted,
                        "is_matched": is_matched,
                    },
                    index=[0],
                )

                df_users = pd.concat([df_users, df_user], ignore_index=True)

                return df_users

    @staticmethod
    def calculate_accuracy(df_users_verification_matrix: pd.DataFrame) -> float:
        return df_users_verification_matrix["is_match"].sum() / len(
            df_users_verification_matrix
        )

    @staticmethod
    def calculate_grant_accessed_rate(
        df_users_verification_matrix: pd.DataFrame,
    ) -> float:
        return df_users_verification_matrix["grant_access"].sum() / len(
            df_users_verification_matrix
        )
