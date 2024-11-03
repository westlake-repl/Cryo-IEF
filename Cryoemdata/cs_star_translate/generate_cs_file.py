from cryosparc.dataset import Dataset
import numpy as np


def generate_cs_file(blob_path, blob_id, image_size, psize, df1, df2, accel_kv=300, cs_mm=2.7, amp_contrast=0.07,
                     phase_shift_rad=0.0, bfactor=0.0):
    data_len = len(blob_path)
    cs_data = Dataset(data_len)
    cs_data.add_fields(
        ['ctf/type', 'ctf/exp_group_id', 'ctf/accel_kv', 'ctf/cs_mm', 'ctf/amp_contrast',
         'ctf/df1_A','ctf/df2_A','ctf/df_angle_rad', 'ctf/phase_shift_rad', 'ctf/scale',
         'ctf/scale_const', 'ctf/shift_A', 'ctf/tilt_A','ctf/trefoil_A', 'ctf/tetra_A',
         'ctf/anisomag', 'ctf/bfactor'],
        ['str', 'int', 'float', 'float', 'float',
         'float', 'float', 'float', 'float', 'float',
         'float', ('float', (2,)),('float', (2,)), ('float', (2,)),('float', (4,)),
         ('float', (4,)), 'float'])
    cs_data.add_fields(
        ['blob/path', 'blob/psize_A', 'blob/idx', 'blob/shape', 'blob/sign','blob/import_sig'],
        ['str', 'float', 'int', ('int', (2,)), 'float',int])

    cs_data['blob/path'] = blob_path
    # cs_data['blob/psize_A'] = [psize] * data_len
    cs_data['ctf/df1_A'] = df1
    cs_data['ctf/df2_A'] = df2
    cs_data['blob/idx'] = blob_id
    cs_data['blob/sign'] = [1.0] * data_len
    cs_data['ctf/scale'] = [1.0] * data_len
    if not isinstance(accel_kv, list) and type(accel_kv) is not np.ndarray:
        cs_data['ctf/accel_kv'] = [accel_kv] * data_len
    else:
        cs_data['ctf/accel_kv'] = accel_kv
    if not isinstance(image_size, list) and type(image_size) is not np.ndarray:
        cs_data['blob/shape'] = [image_size, image_size] * data_len
    else:
        cs_data['blob/shape'] = image_size
    if not isinstance(psize, list) and type(psize) is not np.ndarray:
        cs_data['blob/psize_A'] = [psize] * data_len
    else:
        cs_data['blob/psize_A'] = psize
    if not isinstance(cs_mm, list) and type(cs_mm) is not np.ndarray:
        cs_data['ctf/cs_mm'] = [cs_mm] * data_len
    else:
        cs_data['ctf/cs_mm'] = cs_mm
    if not isinstance(amp_contrast, list) and type(amp_contrast) is not np.ndarray:
        cs_data['ctf/amp_contrast'] = [amp_contrast] * data_len
    else:
        cs_data['ctf/amp_contrast'] = amp_contrast
    if not isinstance(phase_shift_rad, list) and type(phase_shift_rad) is not np.ndarray:
        cs_data['ctf/phase_shift_rad'] = [phase_shift_rad] * data_len
    else:
        cs_data['ctf/phase_shift_rad'] = phase_shift_rad
    if not isinstance(bfactor, list) and type(bfactor) is not np.ndarray:
        cs_data['ctf/bfactor'] = [bfactor] * data_len
    else:
        cs_data['ctf/bfactor'] = bfactor
    return cs_data
