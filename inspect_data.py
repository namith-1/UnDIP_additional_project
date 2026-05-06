import numpy as np
import scipy.io as sio
import os

base = r'c:\Users\likhi\OneDrive\Documents\prev_files\UDIP\processed_data'

print('=== JASPER ===')
try:
    jasper_data = np.load(os.path.join(base, 'jasper', 'data.npy'), allow_pickle=True)
    if jasper_data.dtype == object:
        item = jasper_data.item()
        if hasattr(item, 'keys'):
            print('data.npy (dict) keys:', list(item.keys()))
            for k, v in item.items():
                if hasattr(v, 'shape'):
                    print('  ' + str(k) + ':', v.shape)
        else:
            print('data.npy (object):', type(item))
    else:
        print('data.npy shape:', jasper_data.shape, 'dtype:', jasper_data.dtype)
except Exception as e:
    print('data.npy error:', e)

mat = sio.loadmat(os.path.join(base, 'jasper', 'jasperRidge2_R198.mat'))
keys = [k for k in mat.keys() if not k.startswith('_')]
print('jasperRidge2 keys:', keys)
for k in keys:
    v = mat[k]
    if hasattr(v, 'shape'):
        print('  ' + str(k) + ':', v.shape)

mat2 = sio.loadmat(os.path.join(base, 'jasper', 'end4.mat'))
keys2 = [k for k in mat2.keys() if not k.startswith('_')]
print('end4 keys:', keys2)
for k in keys2:
    v = mat2[k]
    if hasattr(v, 'shape'):
        print('  ' + str(k) + ':', v.shape)

print()
print('=== SAMSON ===')
mat3 = sio.loadmat(os.path.join(base, 'samson', 'samson_1.mat'))
keys3 = [k for k in mat3.keys() if not k.startswith('_')]
print('samson_1 keys:', keys3)
for k in keys3:
    v = mat3[k]
    if hasattr(v, 'shape'):
        print('  ' + str(k) + ':', v.shape)

mat4 = sio.loadmat(os.path.join(base, 'samson', 'end3.mat'))
keys4 = [k for k in mat4.keys() if not k.startswith('_')]
print('end3 keys:', keys4)
for k in keys4:
    v = mat4[k]
    if hasattr(v, 'shape'):
        print('  ' + str(k) + ':', v.shape)

print()
print('=== URBAN ===')
mat5 = sio.loadmat(os.path.join(base, 'urban', 'Urban_R162.mat'))
keys5 = [k for k in mat5.keys() if not k.startswith('_')]
print('Urban_R162 keys:', keys5)
for k in keys5:
    v = mat5[k]
    if hasattr(v, 'shape'):
        print('  ' + str(k) + ':', v.shape)

print()
print('=== APEX ===')
mat6 = sio.loadmat(os.path.join(base, 'apex', 'Y_clean.mat'))
keys6 = [k for k in mat6.keys() if not k.startswith('_')]
print('Y_clean keys:', keys6)
for k in keys6:
    v = mat6[k]
    if hasattr(v, 'shape'):
        print('  ' + str(k) + ':', v.shape)
