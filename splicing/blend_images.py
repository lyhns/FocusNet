
import cv2
import numpy as np
def blend_images(chongdie_img, beijing_img, hole, method=cv2.MIXED_CLONE):
    x, y, w, h = hole
    print(chongdie_img.shape)
    mask = 255*np.ones(chongdie_img.shape[:2], dtype=np.uint8)
    print(np.size(mask,0),np.size(mask,1))
    center = (int(w/2 + x), int(h/2 + y))
    print(center)
    result = cv2.seamlessClone(chongdie_img, beijing_img, mask, center, method)
    return result
if __name__ == '__main__':
    path1 = r'spilt_img/spilt_fusion/IR_spilt/IR_spilt_subrect_main/ir_rectangle_5.png'
    path2 = r'D:\Users\13998\Desktop\code_tools\spilt_img\spilt_fusion\VIS_spilt\Hole_VISimg_100_2.png'

    result = blend_images(path1, 
                        path2,
                        (0, 590, 75, 75), 
                        cv2.MIXED_CLONE)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
