import numpy as np
import math
import cv2
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img1', required=True, help='Path of first image')
    parser.add_argument('--img2', required=True, help='Path of second image')
    parser.add_argument('--warp', type=int, default=5, help='# Warping images')
    parser.add_argument('-a', type=int, default=1, help='Parameter a in equation weight')
    parser.add_argument('-b', type=int, default=2, help='Parameter b in equation weight')
    parser.add_argument('-p', type=int, default=0, help='Parameter p in equation weight')
    return parser.parse_args()

class point:
    def __init__(self, x, y):
        self.x, self.y = x, y

class line:
    def __init__(self, x1, y1, x2, y2):
        self.P, self.Q, self.len = point(x1, y1), point(x2, y2), math.sqrt((x2-x1)**2 + (y2-y1)**2)
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

def interpolate_lines(line_list_1, line_list_2, n, i):
    return [line(math.floor(line_1.P.x*(1-i/(n+1)) + line_2.P.x*i/(n+1)), math.floor(line_1.P.y*(1-i/(n+1)) + line_2.P.y*i/(n+1)),\
                 math.floor(line_1.Q.x*(1-i/(n+1)) + line_2.Q.x*i/(n+1)), math.floor(line_1.Q.y*(1-i/(n+1)) + line_2.Q.y*i/(n+1))) for line_1, line_2 in zip(line_list_1, line_list_2)]

def update_DSUM(x, y, line_src, line_dst, weight, DSUM_x, DSUM_y):
    x_src, y_src = line_src.cal_samplepoint(line_dst.cal_u(x, y), line_dst.cal_v(x, y))
    return DSUM_x + weight * (x_src - x), DSUM_y + weight * (y_src - y)

def get_final_point(x, y, DSUM_x, DSUM_y, weightsum, height, width):
    return min(max(x + DSUM_x / weightsum, 0), height-1), min(max(y + DSUM_y / weightsum, 0), width-1)

def bilinear(image, x, y):
    up, left = math.floor(x), math.floor(y)
    down, right = min(up+1, image.shape[0]-1), min(left+1, image.shape[1]-1)
    a, b = x-up, y-left
    return (1-a)*(1-b)*image[up][left] + (1-a)*b*image[up][right] + a*(1-b)*image[down][left] + a*b*image[down][right]

def output_result(list, height, width):
    videoWriter = cv2.VideoWriter('warp_video.avi', cv2.VideoWriter_fourcc('M', 'P', '4', '2'), 4, (width, height))
    for i, img in enumerate(list):
        videoWriter.write(img)
        cv2.imwrite("warp_" + str(i) + ".jpg", img)
    videoWriter.release()

if __name__ == '__main__':
    args = get_args()
    image_1, image_2 = cv2.imread(args.img1), cv2.imread(args.img2)
    height, width = image_1.shape[0], image_1.shape[1]
    image_2 = cv2.resize(image_2, (height, width))

    line_list_1 = [line(20, 133, 115, 97), line(125, 90, 221, 50), line(230, 59, 295, 97), line(300, 110, 308, 205), line(300, 218, 234, 267), line(229, 276, 112, 343), line(105, 332, 136, 236), line(130, 229, 106, 116), line(95, 159, 28, 141)]
    line_list_2 = [line(5, 129, 113, 52), line(131, 52, 243, 30), line(258, 34, 329, 91), line(330, 104, 318, 249), line(309, 260, 163, 280), line(149, 289, 79, 394), line(68, 387, 103, 222), line(104, 209, 102, 140), line(98, 129, 10, 140)]
 
    output_img_list = [image_1]

    for i in range(1, args.warp+1):
        print('Warping...[{}/{}]'.format(i, args.warp))
        line_list_warp = interpolate_lines(line_list_1, line_list_2, args.warp, i)
        warp = np.zeros((height, width, 3))
        interpolation = i/(args.warp+1)

        for x in range(height):
            for y in range(width):                
                DSUM_x_1, DSUM_y_1, DSUM_x_2, DSUM_y_2, weightsum = 0.0, 0.0, 0.0, 0.0, 0.0

                for line_1, line_2, line_warp in zip(line_list_1, line_list_2, line_list_warp):
                    weight = line_warp.cal_weight(x, y, args.a, args.b, args.p)
                    weightsum += weight
                    DSUM_x_1, DSUM_y_1 = update_DSUM(x, y, line_1, line_warp, weight, DSUM_x_1, DSUM_y_1)
                    DSUM_x_2, DSUM_y_2 = update_DSUM(x, y, line_2, line_warp, weight, DSUM_x_2, DSUM_y_2)

                final_x_1, final_y_1 = get_final_point(x, y, DSUM_x_1, DSUM_y_1, weightsum, height, width)
                final_x_2, final_y_2 = get_final_point(x, y, DSUM_x_2, DSUM_y_2, weightsum, height, width)
                
                warp[x][y] = (1-interpolation) * bilinear(image_1, final_x_1, final_y_1) + interpolation * bilinear(image_2, final_x_2, final_y_2)

        output_img_list.append(warp)
    
    output_img_list.append(image_2)
    output_result(output_img_list, height, width)
