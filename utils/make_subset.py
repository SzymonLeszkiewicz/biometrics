import os

# TODO do usunięcia?
size = 5
dataset = 'train'

print('pwd:', os.getcwd())
PATH = f'../data/{dataset}/'

l = sorted(list(map(int, os.listdir(PATH))))
l = list(map(str, l))
l = l[:size]

new_folder = f'../data/{dataset}_{size}/'
os.makedirs(new_folder, exist_ok=True)
for i in l:
    os.system(f'cp -r {PATH}{i} {new_folder}')

#%%
