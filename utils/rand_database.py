import os
import glob
import cv2
from deepface import DeepFace
import random

'''random database generator
- random specified number of persons'''

auth_size = 15
not_auth_size = 15

num_person = 5
photos_counter = 0
auth_ids = set()

autho = '../data/database/authorized_users'
incoming = '../data/database/incoming_users'
incoming_auth = '../data/database/incoming_users/authorized_users'
incoming_unauth = '../data/database/incoming_users/unauthorized_users'

if any(os.path.exists(path) for path in [autho, incoming, incoming_auth, incoming_unauth]):
    os.system(f'rm -r {autho}')
    os.system(f'rm -r {incoming}')
    os.system(f'rm -r {incoming_auth}')
    os.system(f'rm -r {incoming_unauth}')

os.makedirs(autho)
os.makedirs(incoming)
os.makedirs(incoming_auth)
os.makedirs(incoming_unauth)

rand_person = random.sample(range(1, 10177), num_person)
print(rand_person)

# Add unauthorized users
for p in rand_person:
    for img in glob.glob(f'../data/new_train/{p}/*.jpg'):
        if len(auth_ids) < num_person or photos_counter < auth_size:
            try:
                emb = DeepFace.represent(img, model_name='Facenet')
                auth_ids.add(p)
                if img not in glob.glob(f'{autho}/*.jpg'):
                    os.system(f'cp {img} {autho}')
                    photos_counter += 1
            except:
                print(f'Error with {img}')

print(photos_counter)
print(auth_ids)
unauth_ids = set(range(1, 10177)) - auth_ids

# Add incoming users authorized
tested = 0
for i in auth_ids:
    for img in glob.glob(f'../data/new_test/{i}/*.jpg'):
        try:
            dfs = DeepFace.find(img_path=img, db_path=autho, model_name='Facenet', silent=True)
            if dfs[0].shape[0] != 0:
                os.system(f'cp {img} {incoming_auth}')
                tested += 1
        except:
            print(f'Error with {img}')

print("Size of incoming authorized users: ", tested)

# Add incoming users unauthorized
incoming_unauth_counter = 0
for i in unauth_ids:
    for img in glob.glob(f'../data/new_test/{i}/*.jpg'):
        try:
            dfs = DeepFace.find(img_path=img, db_path=autho, model_name='Facenet', silent=True)
            if dfs[0].shape[0] == 0:
                os.system(f'cp {img} {incoming_unauth}')
                incoming_unauth_counter += 1
        except:
            print(f'Error with {img}')
    if incoming_unauth_counter >= not_auth_size:
        break
