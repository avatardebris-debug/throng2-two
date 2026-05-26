import sys; sys.path.insert(0, '.')
import numpy as np
from src.encoder.multi_resolution_encoder import MultiResolutionEncoder

enc = MultiResolutionEncoder()
g = np.random.randint(0, 10, (15, 20)).astype('float32')
z = enc.encode_structured(g, (7, 10), [('enemy', 5, 12), ('coin', 8, 9)])
print(f'structured z: {z.shape}  (expect ({enc.z_dim},))')
assert z.shape == (enc.z_dim,)

z2 = enc.encode_flat_obs(np.random.rand(378).astype('float32'))
print(f'flat obs z: {z2.shape}')

zb = enc.encode_flat_obs_batch(np.random.rand(12, 378).astype('float32'))
print(f'batch z: {zb.shape}  (expect (12, {enc.z_dim}))')
assert zb.shape == (12, enc.z_dim)

# Edge: player at corners
z_c = enc.encode_structured(g, (0, 0), None)
z_f = enc.encode_structured(g, (14, 19), None)
print(f'corner: {z_c.shape}, far: {z_f.shape}')

r = enc.describe_resolution()
print(f'Global: {r["global"]["shape"]} -> {r["global"]["z_dim"]}d')
print(f'Focal:  {r["focal"]["shape"]} -> {r["focal"]["z_dim"]}d  ' 
      f'({r["focal"]["coverage"]}, {r["focal"]["effective_resolution"]})')
print(f'Entity: max {r["entity"]["max_entities"]} x {r["entity"]["features_per"]} -> {r["entity"]["z_dim"]}d')
print(f'Total z_dim: {r["total_z_dim"]}')
print('PASS')
