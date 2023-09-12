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

# def poly_to_patch(
#     vertices: List[Tuple[float, float]],
#     types: str,
#     color: Tuple[float, float, float],
#     closed: bool,
# ) -> mpatches.PathPatch:
    
def poly_to_patch(
    vertices,
    types,
    color,
    closed,
) -> mpatches.PathPatch:
    """Draw polygons using the Bezier curve."""
    moves = {"L": Path.LINETO, "C": Path.CURVE4}
    points = list(vertices)
    codes = [moves[t] for t in types]
    codes[0] = Path.MOVETO

    if closed:
        points.append(points[0])
        codes.append(Path.LINETO)

    return mpatches.PathPatch(
        Path(points, codes),
        facecolor=color if closed else "none",
        edgecolor=color,
        lw=0 if closed else 1,
        alpha=1,
        antialiased=False,
        snap=True,
    )

def read_json(src_json_path):
    if os.path.exists(src_json_path) == False :
        print(f"invalid json path ! ")
        sys.exit()

    with open(src_json_path) as f:
        json_file = json.load(f)

    print(f"len of this json file : {len(json_file)}")
    return json_file

def check_lane_attr(lane1, lane2):
    if (lane1['attributes']['laneDirection'] != lane2['attributes']['laneDirection'] or \
        lane1['attributes']['laneStyle'] != lane2['attributes']['laneStyle'] or \
        lane1['attributes']['laneTypes'] != lane2['attributes']['laneTypes']):
        return False

    return True

def check_lane_category(lane1, lane2):
    if lane1['category'] != lane2['category']:
        return False
    return True

def check_lane_poly2d(lane1, lane2):
    if (lane1['poly2d'][0]['types'] != lane2['poly2d'][0]['types'] or \
        lane1['poly2d'][0]['closed'] != lane2['poly2d'][0]['closed']):
        return False
    return True
    
def calc_l2_distance_square(pt1, pt2):
    return (pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1])

def check_lane_end_points(lane1, lane2, thres=500):
    start_pt_lane1 = lane1['poly2d'][0]['vertices'][0]
    start_pt_lane2 = lane2['poly2d'][0]['vertices'][0]
    end_pt_lane1 = lane1['poly2d'][0]['vertices'][-1]
    end_pt_lane2 = lane2['poly2d'][0]['vertices'][-1]

    if calc_l2_distance_square(end_pt_lane1, end_pt_lane2) <= thres or calc_l2_distance_square(start_pt_lane1, start_pt_lane2) <= thres:
        return True, calc_l2_distance_square(start_pt_lane1, start_pt_lane2), calc_l2_distance_square(end_pt_lane1, end_pt_lane2), False

    #in case that the labeling sequence in json are reversed
    if calc_l2_distance_square(end_pt_lane1, start_pt_lane2) <= thres or calc_l2_distance_square(start_pt_lane1, end_pt_lane2) <= thres:
        return True, calc_l2_distance_square(end_pt_lane1, start_pt_lane2), calc_l2_distance_square(start_pt_lane1, end_pt_lane2), True
    return False, None, None, None

def comparing(l1_idx, l2_idx, json_data_labels):
    if (check_lane_attr(json_data_labels[l1_idx], json_data_labels[l2_idx]) == False or \
        check_lane_category(json_data_labels[l1_idx], json_data_labels[l2_idx]) == False or \
        check_lane_poly2d(json_data_labels[l1_idx], json_data_labels[l2_idx]) == False):
        return False, None

    ret, _, _, is_reverse = check_lane_end_points(json_data_labels[l1_idx], json_data_labels[l2_idx])
    if ret == False:
        return False, None
    
    return True, is_reverse

#just do average
def merging(lane1, lane2, is_reverse):
    new_lane = lane1

    for i in range(len(lane1['poly2d'][0]['vertices'])):
        if is_reverse == False:
            new_lane['poly2d'][0]['vertices'][i] = [(lane1['poly2d'][0]['vertices'][i][0] + lane2['poly2d'][0]['vertices'][i][0]) / 2, \
                                                    (lane1['poly2d'][0]['vertices'][i][1] + lane2['poly2d'][0]['vertices'][i][1]) / 2]
        else:
            new_lane['poly2d'][0]['vertices'][i] = [(lane1['poly2d'][0]['vertices'][i][0] + lane2['poly2d'][0]['vertices'][-(i+1)][0]) / 2, \
                                                    (lane1['poly2d'][0]['vertices'][i][1] + lane2['poly2d'][0]['vertices'][-(i+1)][1]) / 2]
    return new_lane

def merging_lanes(json_data):
    start_lane_idx = 0
    try:
        max_lane_num = len(json_data['labels'])
    except:
        return None
    
    json_data_labels = json_data['labels']
    selected_labels = []

    for item in json_data_labels:
        if item['category'] == 'crosswalk':
            continue
        selected_labels.append(item)
    
    max_lane_num = len(selected_labels)
    out_lanes = []

    while (start_lane_idx < max_lane_num - 2):
        ret1, _ = comparing(start_lane_idx, start_lane_idx+1, selected_labels)
        ret2, _ = comparing(start_lane_idx+1, start_lane_idx+2, selected_labels)
        ret3, _ = comparing(start_lane_idx, start_lane_idx+2, selected_labels)

        #no matching in this iter, only add the 1st lane, since the 2nd lane may match with the 4th lane in the next iter
        if (ret1 == False and ret2 == False and ret3 == False):
            out_lanes.append(selected_labels[start_lane_idx])
            # out_lanes.append(selected_labels[start_lane_idx+1])
            start_lane_idx += 1
            continue

        #if all matched, choose the best match
        if (ret1 == True and ret2 == True and ret3 == True):
            _, l12_start_dist, l12_end_dist, is_l12_reverse = check_lane_end_points(selected_labels[start_lane_idx], selected_labels[start_lane_idx+1])
            _, l23_start_dist, l23_end_dist, is_l23_reverse = check_lane_end_points(selected_labels[start_lane_idx+1], selected_labels[start_lane_idx+2])
            _, l13_start_dist, l13_end_dist, is_l13_reverse = check_lane_end_points(selected_labels[start_lane_idx], selected_labels[start_lane_idx+2])

            #lane2 and lane3 match
            if ((l23_start_dist + l23_end_dist) < (l12_start_dist + l12_end_dist)) and \
                ((l23_start_dist + l23_end_dist) < (l13_start_dist + l13_end_dist)):
                out_lanes.append(selected_labels[start_lane_idx])
                new_lane = merging(selected_labels[start_lane_idx+1], selected_labels[start_lane_idx+2], is_l23_reverse)
                out_lanes.append(new_lane)
                start_lane_idx += 3

            #lane1 and lane2 match, lane3 maybe match lane4 in next iter
            if ((l12_start_dist + l12_end_dist) < (l23_start_dist + l23_end_dist)) and \
                ((l12_start_dist + l12_end_dist) < (l13_start_dist + l13_end_dist)):
                new_lane = merging(selected_labels[start_lane_idx], selected_labels[start_lane_idx+1], is_l12_reverse)
                out_lanes.append(new_lane)
                start_lane_idx += 2
            
            #TODO : lane1 and lane3 match, lane2 maybe match lane4 in next iter???
            if ((l13_start_dist + l13_end_dist) < (l23_start_dist + l23_end_dist)) and \
                ((l13_start_dist + l13_end_dist) < (l12_start_dist + l12_end_dist)):
                new_lane = merging(selected_labels[start_lane_idx], selected_labels[start_lane_idx+2], is_l13_reverse)
                out_lanes.append(selected_labels[start_lane_idx+1])
                out_lanes.append(new_lane)
                start_lane_idx += 3
            continue

        #2 pairs need to be considered
        ret, is_reverse = comparing(start_lane_idx, start_lane_idx+1, selected_labels)
        if ret == True:
            new_lane = merging(selected_labels[start_lane_idx], selected_labels[start_lane_idx+1], is_reverse)
            out_lanes.append(new_lane)
            start_lane_idx += 2
            continue

        ret, is_reverse = comparing(start_lane_idx, start_lane_idx+2, selected_labels)
        if ret == True:
            new_lane = merging(selected_labels[start_lane_idx], selected_labels[start_lane_idx+2], is_reverse)
            out_lanes.append(selected_labels[start_lane_idx+1])
            out_lanes.append(new_lane)
            start_lane_idx += 3
            continue

        ret, is_reverse = comparing(start_lane_idx+1, start_lane_idx+2, selected_labels)
        out_lanes.append(selected_labels[start_lane_idx])
        new_lane = merging(selected_labels[start_lane_idx+1], selected_labels[start_lane_idx+2], is_reverse)
        out_lanes.append(new_lane)
        start_lane_idx += 3
    
    if max_lane_num - start_lane_idx > 0:
        if max_lane_num - start_lane_idx > 1: #remain 2 lanes
            ret, is_reverse = comparing(start_lane_idx, start_lane_idx+1, selected_labels)
            if ret == True:
                new_lane = merging(selected_labels[start_lane_idx], selected_labels[start_lane_idx+1], is_reverse)
                out_lanes.append(new_lane)
            else:
                out_lanes.append(selected_labels[start_lane_idx])
                out_lanes.append(selected_labels[start_lane_idx+1])
        else:
            for i in range(max_lane_num - start_lane_idx):
                out_lanes.append(selected_labels[start_lane_idx+i])

    if len(out_lanes) <= 0:
        return None
    else:
        return {'name': json_data['name'], 'labels': out_lanes}


def draw_lines_using_plt(lanes):
    height, width = 720, 1280

    # print(f"lanes : {lanes}, {type(lanes)}")
    matplotlib.use("Agg")
    fig = plt.figure(facecolor="0")
    fig.set_size_inches((width / fig.get_dpi()), height / fig.get_dpi())
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_facecolor((0, 0, 0, 0))
    ax.invert_yaxis()

    val_interval = 255 // len(lanes)
    for i in range(len(lanes)):
        color_val = (i + 1) * val_interval / 255.0
        for poly2d in lanes[i]['poly2d']:
            ax.add_patch(
                poly_to_patch(
                    poly2d['vertices'],
                    poly2d['types'],
                    color=(
                        color_val,
                        color_val,
                        color_val,
                    ),
                    closed=poly2d['closed'],
                )
            )
    
    fig.canvas.draw()
    out = np.frombuffer(fig.canvas.tostring_rgb(), np.uint8)
    out = out.reshape((height, width, -1))

    plt.close()
    return out    


def visualize(merged_json, out_inst_map_dirs):
    for single_json in tqdm(merged_json):
        if len(single_json['labels']) <= 0:
            continue
        
        try:
            tmp = 255 // len(single_json['labels'])
        except:
            continue

        file_name = single_json['name'].replace('jpg', 'png')
        inst_map = draw_lines_using_plt(single_json['labels'])

        inst_map = cv2.dilate(inst_map, np.ones((7, 7), dtype=np.uint8), 1)
        inst_map_visual = inst_map = np.squeeze(inst_map)

        file_path = os.path.join(out_inst_map_dirs, file_name)
        cv2.imwrite(file_path, inst_map_visual)


def parse_json_file(json_file, out_inst_map_dirs):
    merged_json = []

    # print(f'-------type : {type(json_file)}, {len(json_file)}')
    for i in tqdm(range(len(json_file))):
        new_json = merging_lanes(json_file[i])
        if new_json != None:
            merged_json.append(new_json)
    
    visualize(merged_json, out_inst_map_dirs)


def main(src_json_path, src_img_path, out_gt_path, args):
    all_json = read_json(src_json_path)

    worker_inerval = len(all_json) // args.all_worker

    start_idx = args.cur_worker * worker_inerval
    if args.cur_worker == args.all_worker - 1:
        end_idx = len(all_json)
    else :
        end_idx = start_idx + worker_inerval

    # len==10, all=3, interval=3, start=0, end=0+3 = 3, [0:3]
    parse_json_file(all_json[start_idx:end_idx], out_gt_path) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="bdd100k_preprocessing")
    parser.add_argument('--cur_worker', type=int, default=0)
    parser.add_argument('--all_worker', type=int, default=1)

    args = parser.parse_args()

    src_json_path = '/home/wan/ZT_2T/work/scrpit/bdd100k/bdd100k/labels/lane/polygons/lane_train.json'
    src_img_path = ''
    out_gt_path = '/home/wan/ZT_2T/work/scrpit/bdd100k/bdd100k_inst/train/'
    main(src_json_path, src_img_path, out_gt_path, args)