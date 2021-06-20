import numpy as np
import math
import cv2
import argparse
import json
import os
import shutil

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w1', type=float, default=1.0, help='Weight for second image')
    parser.add_argument('-w2', type=float, default=0.0, help='Weight for third image')
    parser.add_argument('-w3', type=float, default=0.0, help='Weight for fourth image')
    parser.add_argument('-warp', type=int, default=5, help='# Warping images')
    parser.add_argument('-mode', type=int, default=0, help='0 for interpolate by endpoints, 1 by midpoints')
    parser.add_argument('-a', type=int, default=1, help='Parameter a in equation weight')
    parser.add_argument('-b', type=int, default=2, help='Parameter b in equation weight')
    parser.add_argument('-p', type=int, default=0, help='Parameter p in equation weight')
    return parser.parse_args()

class point:
    def __init__(self, x, y):
        self.x, self.y = x, y

class line:
    def __init__(self, x1, y1, x2, y2):
        self.P, self.Q, self.len, self.theta = point(x1, y1), point(x2, y2), math.sqrt((x2-x1)**2 + (y2-y1)**2), math.atan2(y2-y1,x2-x1)
        self.mid_x, self.mid_y = math.floor((x1 + x2) / 2), math.floor((y1 + y2) / 2)
        self.per_x, self.per_y = self.Q.y-self.P.y, self.P.x-self.Q.x
    
    def cal_u(self, x, y):
        return ((x-self.P.x)*(self.Q.x-self.P.x) + (y-self.P.y)*(self.Q.y-self.P.y)) / self.len**2

    def cal_v(self, x, y):
        return ((x-self.P.x)*self.per_x + (y-self.P.y)*self.per_y) / self.len

    def cal_samplepoint(self, u, v):
        return self.P.x + u * (self.Q.x-self.P.x) + v * self.per_x / self.len, self.P.y + u * (self.Q.y-self.P.y) + v * self.per_y / self.len

    def cal_dist(self, x, y):
        u = self.cal_u(x, y)
        return math.sqrt((x-self.Q.x)**2 + (y-self.Q.y)**2) if u >= 1 else math.sqrt((x-self.P.x)**2 + (y-self.P.y)**2) if u <= 0 else abs(self.cal_v(x, y))
    
    def cal_weight(self, x, y, a, b, p):
        return math.pow((math.pow(self.len, p) / (a + self.cal_dist(x, y))), b)

def get_jsons():
    lines, rects = [], []
    for i in range(4):
        with open("lines" + str(i) + ".json") as f:
            data = json.load(f)
            tmp = []
            for l in data:
                tmp.append(line(l["x1"], l["y1"], l["x2"], l["y2"]))
            lines.append(tmp)
    with open("rects.json") as f:
        data = json.load(f)
        for l in data:
            rects.append(line(l["x1"], l["y1"], l["x2"], l["y2"]))
    return lines, rects

def interpolate_lines(line_list_1, line_list_2, n, i, height, width, mode):
    if mode == 0:   # interpolate lines by endpoints
        return [line(math.floor(line_1.P.x*(1-i/(n+1)) + line_2.P.x*i/(n+1)), math.floor(line_1.P.y*(1-i/(n+1)) + line_2.P.y*i/(n+1)),\
                 math.floor(line_1.Q.x*(1-i/(n+1)) + line_2.Q.x*i/(n+1)), math.floor(line_1.Q.y*(1-i/(n+1)) + line_2.Q.y*i/(n+1))) for line_1, line_2 in zip(line_list_1, line_list_2)]
    if mode == 1:   # interpolate lines by midpoints
        ret = []
        for line_1, line_2 in zip(line_list_1, line_list_2):
            mid_x_new, mid_y_new = math.floor(line_1.mid_x*(1-i/(n+1)) + line_2.mid_x*i/(n+1)), math.floor(line_1.mid_y*(1-i/(n+1)) + line_2.mid_y*i/(n+1))
            length = (line_1.len*(1-i/(n+1)) + line_2.len*i/(n+1)) / 2
            theta = line_1.theta*(1-i/(n+1)) + line_2.theta*i/(n+1) if abs(line_1.theta - line_2.theta) <= math.pi else line_1.theta+(math.pi*2-abs(line_1.theta-line_2.theta))*i/(n+1) if line_1.theta >= 0 else line_1.theta-(math.pi*2-abs(line_1.theta-line_2.theta))*i/(n+1)
            new_x1, new_y1 = min(max(math.floor(mid_x_new - length * math.cos(theta)), 0), height), min(max(math.floor(mid_y_new - length * math.sin(theta)), 0), width)
            new_x2, new_y2 = min(max(math.floor(mid_x_new + length * math.cos(theta)), 0), height), min(max(math.floor(mid_y_new + length * math.sin(theta)), 0), width)
            ret.append(line(new_x1, new_y1, new_x2, new_y2))
        return ret

def update_DSUM(x, y, line_src, line_dst, weight, DSUM_x, DSUM_y):
    x_src, y_src = line_src.cal_samplepoint(line_dst.cal_u(x, y), line_dst.cal_v(x, y))
    return DSUM_x + weight * (x_src - x), DSUM_y + weight * (y_src - y)

def get_final_point(x, y, DSUM_x, DSUM_y, weightsum, height, width):
    return min(max(x + DSUM_x / weightsum, 0), height-1), min(max(y + DSUM_y / weightsum, 0), width-1)

def bilinear(image, x, y):
    up, left = min(math.floor(x), image.shape[0]-1), min(math.floor(y), image.shape[1]-1)
    down, right = min(up+1, image.shape[0]-1), min(left+1, image.shape[1]-1)
    a, b = x-up, y-left
    return (1-a)*(1-b)*image[up][left] + (1-a)*b*image[up][right] + a*(1-b)*image[down][left] + a*b*image[down][right]

def output_result(list, height, width, path="result"):
    if os.path.exists(path):
        shutil.rmtree(path)    
    os.mkdir(path)
        
    videoWriter = cv2.VideoWriter(os.path.join(path, 'warp_video.mp4'), 0x7634706d, 4, (width, height))
    for i, img in enumerate(list):
        cv2.imwrite(os.path.join(path, "warp_" + str(i) + ".jpg"), img)
    for i in range(len(list)):
        img = cv2.imread("warp_" + str(i) + ".jpg")
        videoWriter.write(img)
    videoWriter.release()

if __name__ == '__main__':
    args = get_args()
    image_0, image_1, image_2, image_3 = cv2.imread("image0.png"), cv2.imread("image1.png"), cv2.imread("image2.png"), cv2.imread("image3.png")
    height, width = image_0.shape[0], image_0.shape[1]
    image_1, image_2 = cv2.resize(image_1, (height, width)), cv2.resize(image_2, (height, width))
    
    line_list, rect_list = get_jsons()

    output_img_list = [image_0]

    for i in range(1, args.warp+2):
        print('Warping...[{}/{}]'.format(i, args.warp+1))
        line_list_01 = interpolate_lines(line_list[0], line_list[1], args.warp, i, height, width, args.mode)
        line_list_02 = interpolate_lines(line_list[0], line_list[2], args.warp, i, height, width, args.mode)
        line_list_03 = interpolate_lines(line_list[0], line_list[3], args.warp, i, height, width, args.mode)
        line_list_12 = interpolate_lines(line_list_01, line_list_02, args.w1*10+args.w2*10-1, args.w2*10, height, width, args.mode) if args.w1**2+args.w2**2 > 0 else line_list_01
        line_list_123 = interpolate_lines(line_list_12, line_list_03, args.w1*10+args.w2*10+args.w3*10-1, args.w3*10, height, width, args.mode)
        warp = image_0.copy()
        interpolation = i/(args.warp+1)

        for r in rect_list:
            for x in range(height):
                for y in range(width):
                    DSUM_x_0, DSUM_y_0, DSUM_x_1, DSUM_y_1, DSUM_x_2, DSUM_y_2, DSUM_x_3, DSUM_y_3, weightsum = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

                    for line_0, line_1, line_2, line_3, line_warp in zip(line_list[0], line_list[1], line_list[2], line_list[3], line_list_123):
                        weight = line_warp.cal_weight(x, y, args.a, args.b, args.p)
                        weightsum += weight
                        DSUM_x_0, DSUM_y_0 = update_DSUM(x, y, line_0, line_warp, weight, DSUM_x_0, DSUM_y_0)
                        DSUM_x_1, DSUM_y_1 = update_DSUM(x, y, line_1, line_warp, weight, DSUM_x_1, DSUM_y_1)
                        DSUM_x_2, DSUM_y_2 = update_DSUM(x, y, line_2, line_warp, weight, DSUM_x_2, DSUM_y_2)
                        DSUM_x_3, DSUM_y_3 = update_DSUM(x, y, line_3, line_warp, weight, DSUM_x_3, DSUM_y_3)

                    final_x_0, final_y_0 = get_final_point(x, y, DSUM_x_0, DSUM_y_0, weightsum, height, width)
                    final_x_1, final_y_1 = get_final_point(x, y, DSUM_x_1, DSUM_y_1, weightsum, height, width)
                    final_x_2, final_y_2 = get_final_point(x, y, DSUM_x_2, DSUM_y_2, weightsum, height, width)
                    final_x_3, final_y_3 = get_final_point(x, y, DSUM_x_3, DSUM_y_3, weightsum, height, width)
                    
                    if x < r.P.x or x >= r.Q.x or y < r.P.y or y >= r.Q.y:
                        warp[x][y] = bilinear(image_0, final_x_0, final_y_0)
                    else:
                        warp[x][y] = (1-interpolation) * bilinear(image_0, final_x_0, final_y_0) + interpolation * (args.w1 * bilinear(image_1, final_x_1, final_y_1) + args.w2 * bilinear(image_2, final_x_2, final_y_2) + args.w3 * bilinear(image_3, final_x_3, final_y_3))

        output_img_list.append(warp)
    
    output_result(output_img_list, height, width)