from noise_addition import GaussianTransformer, luminance_transform_directory
import os

gaussian_transformer = GaussianTransformer()

path_fine_tuning = os.path.join("..", "data", "database")
path_incoming = os.path.join("..", "data", "database", "incoming_users")

# Noisy incommers
gaussian_transformer.transform_directory(
    os.path.join(path_incoming, "authorized_users"),
    os.path.join(path_incoming, "gauss_authorized_users"),
)

gaussian_transformer.transform_directory(
    os.path.join(path_incoming, "unauthorized_users"),
    os.path.join(path_incoming, "gauss_unauthorized_users"),
)

luminance_transform_directory(
    os.path.join(path_incoming, "authorized_users"),
    os.path.join(path_incoming, "lum_authorized_users"),
)

luminance_transform_directory(
    os.path.join(path_incoming, "unauthorized_users"),
    os.path.join(path_incoming, "lum_unauthorized_users"),
)

# Fine-tuned Database
# parametry jakie chcemy zfinetunować - stworzy sie taka baza danych z oryginalnymi zdjeciami i zdjeciami i szumem
luminance_parameters = {"linear": [1.5], "constant": [-100]}

luminance_transform_directory(
    os.path.join(path_fine_tuning, "authorized_users"),
    os.path.join(path_fine_tuning, "lum_fine_authorized_users"),
    True,
    luminance_parameters,
)

gaussian_parameters = [10, 20]

gaussian_transformer.transform_directory(
    os.path.join(path_fine_tuning, "authorized_users"),
    os.path.join(path_fine_tuning, "gauss_fine_authorized_users"),
    True,
    gaussian_parameters,
)
