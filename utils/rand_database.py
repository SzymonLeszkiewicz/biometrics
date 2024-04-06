import os
import glob
import cv2
from deepface import DeepFace
import random

'''
Random database generator
- random specified number of persons

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
'''

random.seed(42)
not_auth_size = 100  # liczba zdjęć nieautoryzowanych
num_person = 150  # liczba wdrożonych profili
incoming_auth_size = 500  # test na wdrożonych

photos_counter = 0
auth_ids = set()

autho = '../data/database/authorized_users'
incoming = '../data/database/incoming_users'
incoming_auth = '../data/database/incoming_users/authorized_users'
incoming_unauth = '../data/database/incoming_users/unauthorized_users'

# if any(os.path.exists(path) for path in [autho, incoming, incoming_auth, incoming_unauth]):
#     os.system(f'rm -r {autho}')
#     os.system(f'rm -r {incoming}')
#     os.system(f'rm -r {incoming_auth}')
#     os.system(f'rm -r {incoming_unauth}')

os.makedirs(autho)
os.makedirs(incoming)
os.makedirs(incoming_auth)
os.makedirs(incoming_unauth)

rand_person = random.sample(range(1, 10177), num_person)
print(rand_person)

# Add unauthorized users
for p in rand_person:
    os.makedirs(f'{autho}/{p}')
    for img in glob.glob(f'../data/new_train/{p}/*.jpg'):
        if len(auth_ids) < num_person:
            try:
                emb = DeepFace.represent(img, model_name='Facenet')
                auth_ids.add(p)
                if img not in glob.glob(f'{autho}/{p}/*.jpg'):
                    os.system(f'cp {img} {autho}/{p}/')
                    photos_counter += 1
            except:
                print(f'Error with {img}')

print(photos_counter)
print(auth_ids)

# Add incoming users authorized
tested = 0
for i in auth_ids:
    os.makedirs(f'{incoming_auth}/{i}')
    for img in glob.glob(f'../data/new_test/{i}/*.jpg'):
        os.system(f'cp {img} {incoming_auth}/{i}/')
        tested += 1

    if tested >= incoming_auth_size:
        break

print("Size of incoming authorized users photos: ", tested)
if tested < incoming_auth_size:
    print("Not enough photos in incoming authorized users")

# Add incoming users unauthorized
unauth_ids = set(range(1, 10177)) - auth_ids
unauth_ids = random.sample(list(unauth_ids), len(unauth_ids))
incoming_unauth_counter = 0
for i in unauth_ids:
    os.makedirs(f'{incoming_unauth}/{i}')
    for img in glob.glob(f'../data/new_test/{i}/*.jpg'):
        os.system(f'cp {img} {incoming_unauth}/{i}/')
        incoming_unauth_counter += 1

    if incoming_unauth_counter >= not_auth_size:
        break

if incoming_unauth_counter < not_auth_size:
    print("Not enough photos in incoming unauthorized users")
