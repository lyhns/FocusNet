import cv2
def ir_edge_detect(input_imgpath):
    input_img = cv2.imread(input_imgpath, cv2.IMREAD_GRAYSCALE)
    return cv2.Canny(input_img, 100, 200)
if __name__ == '__main__':
    input_imgpath = r'VIS1.jpg'
    output_img = ir_edge_detect(input_imgpath = input_imgpath)
    cv2.imwrite('VIS1_image_edge2.png', output_img)
