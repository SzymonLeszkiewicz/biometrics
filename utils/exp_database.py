from noise_addition import GaussianTransformer, luminance_transform_directory

gaussian_transformer = GaussianTransformer()
gaussian_transformer.transform_directory('../data/database/incoming_users/authorized_users',
                                         '../data/database/incoming_users/trans_authorized_users')

gaussian_transformer.transform_directory('../data/database/incoming_users/unauthorized_users',
                                         '../data/database/incoming_users/trans_unauthorized_users')

luminance_transform_directory('../data/database/incoming_users/authorized_users',
                              '../data/database/incoming_users/lum_trans_authorized_users')

luminance_transform_directory('../data/database/incoming_users/unauthorized_users',
                              '../data/database/incoming_users/lum_trans_unauthorized_users')
