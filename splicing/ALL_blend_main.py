
import blend_images as BI
import check_size as CS
import pandas as pd
import cv2
import os
import Change_size as fileChange
def all_blend(img_index,
              excel_path,sheet_name,
              beijing_img_path ,chongdie_img_path_root,
              outimg_name_path
              ):
    excel_path = excel_path
    hole_data = pd.read_excel(excel_path, sheet_name=sheet_name)
    beijing_img = cv2.imread(beijing_img_path)
    for row_number in range(0,hole_data.shape[0]):
        img_number = hole_data.iloc[row_number, 4]
        img_number = int(img_number)
        print("#############",img_number)
        start_x = hole_data.iloc[row_number, 0]
        start_y = hole_data.iloc[row_number, 1]
        end_x = hole_data.iloc[row_number, 2]
        end_y = hole_data.iloc[row_number, 3]
        hole = (int(start_x), int(start_y), int(end_x-start_x), int(end_y-start_y))
        img_number = str(img_number) 
        chongdie_img_path = chongdie_img_path_root + img_index + '/vis_rectangle_{}.png'.format(img_number)
        chongdie_img = cv2.imread(chongdie_img_path)  
        output = BI.blend_images(chongdie_img, beijing_img, hole)
        beijing_img = output
        outimg_name_path = 'spilt_img/spilt_fusion/Blend_results/result_{}.png'.format(img_index)
        cv2.imwrite(outimg_name_path, output)
        return outpu
if __name__ == '__main__':      
    dir = 'test_imgs/results_img'
    filename_list = fileChange.GetPNGName(dir)
    for index in filename_list:
        raw_IRimg_path = 'test_imgs/ir/' + index +'.png'
        raw_VISimg_path = 'test_imgs/vis/' + index +'.png'
        fused_img_path = 'test_imgs/results_img/' + index +'.png'
        excel_path = 'spilt_img/spilt_fusion/spilt_save_rectangles/CS_positions/positions_{}_CS.xlsx'.format(index)
        beijing_img_path = 'test_imgs/results_img/{}.png'.format(index)
        output = all_blend(img_index = index,
                    excel_path = excel_path,
                    sheet_name='Positions2',
                    beijing_img_path = beijing_img_path,
                    chongdie_img_path_root = 'spilt_img/spilt_fusion/VIS_spilt/VIS_spilt_subrect_main/',
                    outimg_name_path = 'spilt_img/spilt_fusion/Blend_results/result_{}.png'.format(index)
                    )   