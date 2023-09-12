from tqdm import tqdm
import argparse
import json
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.patches as mpatches
from matplotlib.path import Path
import labelme
import base64
from scipy.special import comb as n_over_k

class BezierCurve(object):
    # Define Bezier curves for curve fitting
    def __init__(self, order, num_sample_points=50):
        self.num_point = order + 1
        self.control_points = []
        self.bezier_coeff = self.get_bezier_coefficient()
        self.num_sample_points = num_sample_points
        self.c_matrix = self.get_bernstein_matrix()

    def get_bezier_coefficient(self):
        Mtk = lambda n, t, k: t ** k * (1 - t) ** (n - k) * n_over_k(n, k)
        BezierCoeff = lambda ts: [[Mtk(self.num_point - 1, t, k) for k in range(self.num_point)] for t in ts]

        return BezierCoeff

    def interpolate_lane(self, x, y, n=50):
        # Spline interpolation of a lane. Used on the predictions
        assert len(x) == len(y)

        tck, _ = splprep([x, y], s=0, t=n, k=min(3, len(x) - 1))

        u = np.linspace(0., 1., n)
        return np.array(splev(u, tck)).T

    def get_control_points(self, x, y, interpolate=False):
        if interpolate:
            points = self.interpolate_lane(x, y)
            x = np.array([x for x, _ in points])
            y = np.array([y for _, y in points])

        middle_points = self.get_middle_control_points(x, y)
        for idx in range(0, len(middle_points) - 1, 2):
            self.control_points.append([middle_points[idx], middle_points[idx + 1]])

    def get_bernstein_matrix(self):
        tokens = np.linspace(0, 1, self.num_sample_points)
        c_matrix = self.bezier_coeff(tokens)
        return np.array(c_matrix)

    def save_control_points(self):
        return self.control_points

    def assign_control_points(self, control_points):
        self.control_points = control_points

    def quick_sample_point(self, image_size=None):
        control_points_matrix = np.array(self.control_points)
        sample_points = self.c_matrix.dot(control_points_matrix)
        if image_size is not None:
            sample_points[:, 0] = sample_points[:, 0] * image_size[-1]
            sample_points[:, -1] = sample_points[:, -1] * image_size[0]
        return sample_points

    def get_sample_point(self, n=50, image_size=None):
        '''
            :param n: the number of sampled points
            :return: a list of sampled points
        '''
        t = np.linspace(0, 1, n)
        coeff_matrix = np.array(self.bezier_coeff(t))
        control_points_matrix = np.array(self.control_points)
        sample_points = coeff_matrix.dot(control_points_matrix)
        if image_size is not None:
            sample_points[:, 0] = sample_points[:, 0] * image_size[-1]
            sample_points[:, -1] = sample_points[:, -1] * image_size[0]

        return sample_points

    def get_middle_control_points(self, x, y):
        dy = y[1:] - y[:-1]
        dx = x[1:] - x[:-1]
        dt = (dx ** 2 + dy ** 2) ** 0.5
        t = dt / dt.sum()
        t = np.hstack(([0], t))
        t = t.cumsum()
        data = np.column_stack((x, y))
        Pseudoinverse = np.linalg.pinv(self.bezier_coeff(t))  # (9,4) -> (4,9)
        control_points = Pseudoinverse.dot(data)  # (4,9)*(9,2) -> (4,2)
        medi_ctp = control_points[:, :].flatten().tolist()

        return medi_ctp


def read_inst_maps(src_inst_map_dir):
    inst_full_paths = []
    for root, dirs, files in os.walk(src_inst_map_dir):
        for file in files:
            if file.rsplit('.', 1)[-1] != 'png':
                continue

            inst_full_paths.append(os.path.join(root, file))
    return inst_full_paths

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return n_over_k(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=10):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def _lane_instance_gt_bezier_curve_parse(inst_map, val, ctrl_pts):
    xvals, yvals = bezier_curve(ctrl_pts)

    for idx in range(len(xvals)):
        y = int(xvals[idx])
        x = int(yvals[idx])

        if x >= inst_map.shape[1]:
            x = inst_map.shape[1] - 1
        
        if y >= inst_map.shape[0]:
            y = inst_map.shape[0] - 1
        
        if inst_map[y][x] != 0:
            continue
        fill_len = 2
        for i in range(fill_len):
            tmp_x_low = int(x) - i
            if tmp_x_low < 0:
                tmp_x_low = 0
            tmp_x_high = int(x) + i
            if tmp_x_high >= inst_map.shape[1]:
                tmp_x_high = inst_map.shape[1] - 1
            
            inst_map[y][tmp_x_low:tmp_x_high] = int(val)
    
    return yvals, xvals
    

def cvt_to_json(inst_full_paths, src_img_dir, out_json_dir, src_inst_map_dir, args):
    worker_inerval = len(inst_full_paths) // args.all_worker

    start_idx = args.cur_worker * worker_inerval
    if args.cur_worker == args.all_worker - 1:
        end_idx = len(inst_full_paths)
    else :
        end_idx = start_idx + worker_inerval

    for inst_map_path in tqdm(inst_full_paths[start_idx:end_idx]):
        if os.path.exists(inst_map_path) == False:
            continue

        src_img_path = os.path.normpath(inst_map_path.replace(src_inst_map_dir, src_img_dir).replace('png', 'jpg'))
        if os.path.exists(src_img_path) == False:
            continue
        src_img = cv2.imread(src_img_path, cv2.IMREAD_UNCHANGED)
        print(inst_map_path)
        inst_img = cv2.imread(inst_map_path, cv2.IMREAD_UNCHANGED)
        inst_img = cv2.resize(inst_img, (src_img.shape[1], src_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        unique_vals, unique_idxs = np.unique(inst_img, return_inverse=True)

        if len(unique_vals) == 0:
            continue

        fitted_map = np.zeros_like(inst_img)
        # print(src_img_path)
        # print(out_json_dir)
        json_path = src_img_path.replace(src_img_dir, out_json_dir).replace('jpg', 'json')
        if os.path.exists(json_path.rsplit('/', 1)[0]) == False:
            os.makedirs(json_path.rsplit('/', 1)[0])
        
        with open(json_path, 'w', encoding='utf-8') as f:
            js = {}
            js['version'] = '5.2.1'
            js['flags'] = {}
            js['shapes'] = []
            js['imagePath'] = src_img_path.rsplit('/', 1)[1]

            data = labelme.LabelFile.load_image_file(src_img_path)
            js['imageData'] = base64.b64encode(data).decode('utf-8')
            js['imageHeight'] = src_img.shape[0]
            js['imageWidth'] = src_img.shape[1]

            lane_cnt = 0
            print(f"unique_vals len : {len(unique_vals)}")
            for i in range(len(unique_vals)):
                if unique_vals[i] == 0:
                    continue
                cur_idx = np.where(inst_img == unique_vals[i])
                # if len(cur_idx[0]) <= 500: #sift short lanes
                #     continue

                fcns = BezierCurve(order=3)
                fcns.get_control_points(cur_idx[0], cur_idx[1], interpolate=False)
                matrix = fcns.save_control_points()

                for i in range(len(matrix)):
                    if matrix[i][0] < 0: matrix[i][0] = 1e-6
                    if matrix[i][1] < 0: matrix[i][1] = 1e-6
                
                xs, ys = _lane_instance_gt_bezier_curve_parse(fitted_map, 255, matrix)
                # if (np.square((xs[0] - xs[-1])) + np.square((ys[0] - ys[-1]))) <= 10000:
                #     continue
                
                lane_tmp = {}
                lane_tmp['label'] = str(lane_cnt)
                lane_cnt += 1
                lane_tmp['group_id'] = None
                lane_tmp['description'] = ''
                lane_tmp['shape_type'] = 'linestrip'
                lane_tmp['flags'] = {}
                lane_tmp['points'] = []
                for j in range(len(xs)):
                    lane_tmp['points'].append([int(xs[j]), int(ys[j])])
            
                js['shapes'].append(lane_tmp)
            
            article = json.dumps(js, indent=2)
            f.write(article)
        
        debug_out_path = src_img_path.replace(src_img_dir, '/home/wan/ZT_2T/work/scrpit/bdd100k/debug_json_out')
        cv2.imwrite(debug_out_path, fitted_map)

def main(src_inst_map_dir, src_img_dir, out_json_dir, args):
    inst_full_paths = read_inst_maps(src_inst_map_dir)
    cvt_to_json(inst_full_paths, src_img_dir, out_json_dir, src_inst_map_dir, args)

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="bdd100k_preprocessing")
    parser.add_argument('--cur_worker', type=int, default=0)
    parser.add_argument('--all_worker', type=int, default=1)

    args = parser.parse_args()

    src_inst_map_dir = '/home/wan/ZT_2T/work/scrpit/bdd100k/script_out'
    src_img_dir = '/home/wan/ZT_2T/work/scrpit/bdd100k/script_out'
    out_json_dir = '/home/wan/ZT_2T/work/scrpit/bdd100k/json_out'
    
    main(src_inst_map_dir, src_img_dir, out_json_dir, args)