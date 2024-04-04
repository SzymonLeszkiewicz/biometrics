import os

path = '../data/'
subset = ['train', 'test', 'val']
new_name = 'all'

def join_subsets(path, subset, new_name):
    # Create a new folder
    os.makedirs(os.path.join(path, new_name), exist_ok=True)

    for sun in subset:
        for per in os.listdir(os.path.join(path, sun)):
            os.makedirs(os.path.join(path, new_name, per), exist_ok=True)
            for img in os.listdir(os.path.join(path, sun, per)):
                # Copy the image
                os.rename(os.path.join(path, sun, per, img), os.path.join(path, new_name, per, img))

join_subsets(path, subset, new_name)

os.makedirs(os.path.join(path, 'new_test'), exist_ok=True)
os.makedirs(os.path.join(path, 'new_train'), exist_ok=True)

test_size = 0.2
for person in os.listdir(os.path.join(path, 'all')):
    photos = os.listdir(os.path.join(path, 'all', person))
    size_test = int(len(photos) * test_size)
    size_train = len(photos) - size_test
    os.makedirs(os.path.join(path, 'new_test', person), exist_ok=True)
    os.makedirs(os.path.join(path, 'new_train', person), exist_ok=True)
    if size_test == 0:
        size_test = 1
        size_train -= 1
    for i in range(size_test):
        os.rename(os.path.join(path, 'all', person, photos[i]), os.path.join(path, 'new_test', person, photos[i]))
    for i in range(size_train):
        os.rename(os.path.join(path, 'all', person, photos[i + size_test]), os.path.join(path, 'new_train', person, photos[i + size_test]))


#%%
