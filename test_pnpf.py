import numpy as np
import poselib
import glob
import scipy
import time
from tqdm import tqdm

def str2vec(s):
    return np.array([float(x) for x in s.split(' ')])

def read_dump(filename):
    
    with open(filename,'r') as f:
        lines = f.read().splitlines()

    res = {}
    res['name'] = lines[0]
    res['estimate_focal_length'] = bool(lines[1] == '1')
    res['init_q'] = str2vec(lines[2])
    res['init_t'] = str2vec(lines[3])
    params = str2vec(lines[4])
    res['width'] = int(params[0])
    res['height'] = int(params[1])
    res['init_params'] = params[2:]

    res['q'] = str2vec(lines[5])
    res['t'] = str2vec(lines[6])
    params = str2vec(lines[7])
    res['params'] = params[2:]

    res['num_pts'] = int(lines[8])

    res['x'] = []
    res['X'] = []
    res['inlier'] = []
    

    for l in lines[9:]:
        v = str2vec(l)
        res['x'].append(v[0:2])
        res['X'].append(v[2:5])
        res['inlier'].append(v[5])
        
    res['x'] = np.array(res['x'])
    res['X'] = np.array(res['X'])
    res['inlier'] = np.array(res['inlier'])

    return res


def compute_reproj_error(pose, camera, p2d, p3d, tol = 12.0):
    proj = (pose.R @ p3d.T).T + pose.t
    proj = proj[:,0:2] / proj[:,[2]]
    if(len(camera.params) == 4):
        # SIMPLE_RADIAL model
        k0 = camera.params[3]
        r2 = proj[:,0]**2 + proj[:,1]**2
        proj = proj * (1.0 + k0 * r2)

    proj = proj * camera.params[0] + camera.params[1:3]
    res = proj-p2d
    errs = np.sqrt(np.sum(res**2,axis=1))
    inl = errs < tol
    return np.sqrt(np.sum(res[inl]**2,axis=1).mean()), np.mean(inl)

def read_gt_focals(filename):
    with open(filename,'r') as f:
        lines = f.read().splitlines()
    lines = [l.split() for l in lines]
    res = {}
    for l in lines:
        if len(l) == 1:
            continue
        res[l[0]] = float(l[1])
    return res

def compute_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(scipy.integrate.trapezoid(r, x=e)/t)
    if len(thresholds) == 1:
        aucs = aucs[0]
    return aucs


def set_config(solver = 'P4Pf', refine_minimal_sample=1, filter_minimal_sample=1):
    if solver == 'P4Pf':
        solver_config = 0
    elif solver == 'P35Pf':
        solver_config = 1
    elif solver == 'P5Pf':
        solver_config = 2

    config_flags = (refine_minimal_sample << 0) | (filter_minimal_sample << 1)
    config = solver_config | (config_flags << 16)
    return config

def run_method(data, config):
    p2d = data['x']
    p3d = data['X']

    pp = np.array([data['width']/2.0, data['height']/2.0])

    if config < 0:

        pose = poselib.CameraPose()
        camera = poselib.Camera()
        camera.model_id = 0
        camera.params = [0.0, pp[0], pp[1]]
        runtime = -1
        # COLMAP baselines
        if config == -1:
            # Initial estimate
            pose.q = data['init_q']
            pose.t = data['init_t']
            camera.params = data['init_params'][0:3]
        elif config == -2:
            # Refined estimate
            pose.q = data['q']
            pose.t = data['t']
            camera.params = data['params'][0:3]

    else:
        start = time.time()
        im, info = poselib.estimate_absolute_pose_focal(p2d, p3d, pp, config)
        end = time.time()
        pose = im.pose
        camera = im.camera
        runtime = 1000.0 * (end - start) # ms

    return pose, camera, runtime

def tabelize(data, labels):
    # assumes data is a list of dicts with the same keys
    
    print(f'{"":30s} ', end=" ")
    for l in labels:
        print(f'{l:>12s}', end=" ")
    print("")

    for m in data[0].keys():
        print(f'{m:30s}:', end=" ")
        for d in data:
            print(f'{d[m]:>12.4f}', end=" ")
        print("")

def main():
    
    gt_focals = read_gt_focals('/home/viktor/datasets/ArtsQuad_dataset/focals.txt')


    methods = {
        'colmap_EstimateAbsolutePose': -1,
        'colmap_+RefineAbsolutePose': -2,
        'P4Pf': set_config('P4Pf', 0, 0),
        'P4Pf+Filter': set_config('P4Pf', 0, 1),
        'P4Pf+Refine': set_config('P4Pf', 1, 0),
        'P4Pf+Filter+Refine': set_config('P4Pf', 1, 1),
        'P35Pf': set_config('P35Pf', 0, 0),
        'P35Pf+Filter': set_config('P35Pf', 0, 1),
        'P35Pf+Refine': set_config('P35Pf', 1, 0),
        'P35Pf+Filter+Refine': set_config('P35Pf', 1, 1),
        'P5Pf': set_config('P5Pf', 0, 0),
        'P5Pf+Filter': set_config('P5Pf', 0, 1),
        'P5Pf+Refine': set_config('P5Pf', 1, 0),
        'P5Pf+Filter+Refine': set_config('P5Pf', 1, 1)

    }

    export_path = '/home/viktor/datasets/colmap_export/'
    
    files = glob.glob(f'{export_path}*.txt')

    inlier_ratios = {}
    reproj_errors = {}
    focal_lengths = {}
    gt_focal_lengths = []


    inlier_ratios = {m:[] for m in methods.keys()}
    reproj_errors = {m:[] for m in methods.keys()}
    focal_lengths = {m:[] for m in methods.keys()}
    runtimes = {m:[] for m in methods.keys()}

#    files = files[0:20]

    for f in tqdm(files):
        data = read_dump(f)
        if not data['estimate_focal_length']:
            continue

        if data['name'] not in gt_focals:
            continue

        p2d = data['x']
        p3d = data['X']


        #import ipdb
        #ipdb.set_trace()

            
        gt_focal_lengths += [gt_focals[data['name']]]
        for (m, config) in methods.items():
            pose, camera, runtime = run_method(data, config)
            reproj_error, inl_ratio = compute_reproj_error(pose, camera, p2d, p3d)
            focal = camera.params[0]
            inlier_ratios[m] += [inl_ratio]
            reproj_errors[m] += [reproj_error]
            focal_lengths[m] += [focal]
            runtimes[m] += [runtime]
        


    inlier_ratios = {m:np.array(v) for (m,v) in inlier_ratios.items()}
    reproj_errors = {m:np.array(v) for (m,v) in reproj_errors.items()}
    focal_lengths = {m:np.array(v) for (m,v) in focal_lengths.items()}
    runtimes = {m:np.array(v) for (m,v) in runtimes.items()}

    focal_errors = {m:100.0*np.abs(v-gt_focal_lengths)/gt_focal_lengths for (m,v) in focal_lengths.items()}
    avg_inl_ratio = {m: np.mean(inlier_ratios[m]) for m in methods.keys()}
    med_inl_ratio = {m: np.median(inlier_ratios[m]) for m in methods.keys()}
    avg_reproj_error = {m: np.mean(reproj_errors[m]) for m in methods.keys()}
    med_reproj_error = {m: np.median(reproj_errors[m]) for m in methods.keys()}

    avg_focal_err = {m: np.mean(focal_errors[m]) for m in methods.keys()}
    med_focal_err = {m: np.median(focal_errors[m]) for m in methods.keys()}
    auc20_focal_err = {m: 100.0*compute_auc(focal_errors[m], [20.0]) for m in methods.keys()}

    avg_runtime =  {m: np.mean(runtimes[m]) for m in methods.keys()}

    data = [avg_inl_ratio,med_inl_ratio, avg_reproj_error, med_reproj_error, avg_focal_err, med_focal_err, auc20_focal_err, avg_runtime]
    labels = ['avg INL.', 'med INL.', 'avg REPROJ', 'med REPROJ', 'avg FOCAL', 'med FOCAL', 'auc20 FOCAL', 'avg RUNTIME']

    print(f'Instances: {len(gt_focal_lengths)}')
    tabelize(data, labels)


    #import ipdb
    #ipdb.set_trace()


    
    # w x y z
    # 

    # x y z w






if __name__ == '__main__':
    main()