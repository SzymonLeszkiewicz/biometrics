from noise_addition import GaussianTransformer, luminance_transform_directory
import os

gaussian_transformer = GaussianTransformer()

path_fine = os.path.join('..', 'data', 'database')
path_incoming = os.path.join('..', 'data', 'database', 'incoming_users')

# Noisy incommers
# gaussian_transformer.transform_directory(os.path.join(path_incoming, 'authorized_users'), os.path.join(path_incoming, 'gauss_authorized_users'))
#
# gaussian_transformer.transform_directory(os.path.join(path_incoming, 'unauthorized_users'), os.path.join(path_incoming, 'gauss_unauthorized_users'))
#
# luminance_transform_directory(os.path.join(path_incoming, 'authorized_users'), os.path.join(path_incoming, 'lum_authorized_users'))
#
# luminance_transform_directory(os.path.join(path_incoming, 'unauthorized_users'), os.path.join(path_incoming, 'lum_unauthorized_users'))

# Fine-tuned Database
# parametry jakie chcemy zfinetunować - stworzy sie taka baza danych z oryginalnymi zdjeciami i zdjeciami i szumem
lum_par = {
    "linear": [0.5],
}
luminance_transform_directory(os.path.join(path_fine, 'authorized_users'), os.path.join(path_fine, 'lum_fine_authorized_users'), True, lum_par)

gaus_par = [20]
gaussian_transformer.transform_directory(os.path.join(path_fine, 'authorized_users'), os.path.join(path_fine, 'gauss_fine_authorized_users'), True, gaus_par)
