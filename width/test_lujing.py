import glob
import time
from app.yolov5_det.view import width_det, predict

from config.log_config import logger
from utils.util_func import load_yaml
import matplotlib.pyplot as plt
from config.sql_server_config import *
from collections import Counter

import glob
import json
import os
from app.yolov5_det.utils_detect.utils import gamma_and_clahe, mask_to_img, shift_and_concatenate_images
from app.yolov5_det.view import width_det

from utils.util_func import load_yaml, cv_imwrite
import matplotlib.pyplot as plt
import traceback

model_name = 'width_det'

import cv2

while True:
    try:
        # conn = pyodbc.connect(conn_str)
        # cursor = conn.cursor()
        #
        # query = f"SELECT TOP 1 ID,Path,Note ,SlabMWidth1 from tblSlabDefectInformationHistory where SlabNo = 'Z424J001594102X' and ID = '20240905131439590' order by ProductionTime desc"
        # cursor.execute(query)
        # # query = 'SELECT TOP 1 ID,Path,Note ,SlabMWidth1 from tblSlabDefectInformationHistory where SlabMWidth1 = 0 and  AcqisitionCompletedFlag = 1  order by ProductionTime desc'
        # # cursor.execute(query)
        # # for result in cursor.fetchall():
        # result = cursor.fetchone()
        #
        # if result:
        #     ID, root_dir, Note,SlabMWidth1 = result
        #     if SlabMWidth1 != 0:
        #         logger.debug(f"这个批号{ID}已识别宽度")
        #         time.sleep(1)
        #         continue
        #
        # else:
        #     logger.error("数据库查询为空")
        #     time.sleep(3)
        #     continue
        root_dir = r'\\YONFENG3\Images\2024-09-11\08\20240911083119824'

        offset_top = 500
        # all_result = []
        all_result = {}
        top_camara_list = ['01', '02', '03', '04', '05', '06']
        config_points = load_yaml('app/yolov5_det/configs/point.yaml')  # 获取每张图片的测量信息
        mm_px_list = [abs(config_points[i]['left_distance'] - config_points[i]['right_distance']) for i in
                      top_camara_list]  # 获取每个相机的mmp
        ct_camara, ct_index = top_camara_list[int(len(top_camara_list) / 2)], int(len(top_camara_list) / 2)  # 获取中间相机与下标
        ct_img_path_list = sorted([path for path in glob.glob(rf'{root_dir}\{ct_camara}\**\*.png', recursive=True) if
                                   '_processed' not in path],
                                  key=lambda x: int(x.split('_')[-1][0].replace('.png', '')))  # 获取中间相机的图片路径

        for ct_img_path in ct_img_path_list:
            try:
                left_flg, right_flg = False, False

                for left_camara in top_camara_list[0:ct_index + 1]:
                    """找左侧满足条件的点"""
                    left_img_path = ct_img_path.replace(f'{ct_camara}\\{ct_camara}_',
                                                        f'{left_camara}\\{left_camara}_')
                    print(f"左侧文件夹路径：{left_img_path}")

                    left_parent_folder = os.path.basename(os.path.dirname(left_img_path))
                    mask_parent_path = f"app/yolov5_det/ruler/{left_parent_folder}/ruler.jpg"
                    left_mask_rule_img = mask_to_img(left_img_path, mask_parent_path, offset_top)  # 左侧加了尺子的图片

                    response_left_result = width_det(img_info=left_mask_rule_img, model_name=model_name, offset=offset_top)
                    if not response_left_result['shapes']:
                        left_flg = False
                        continue
                    elif left_camara == '01' and response_left_result['shapes'][0]['center_x'] < 180:
                        left_flg = False
                        continue
                    elif left_camara == '01' and response_left_result['shapes'][0]['center_x'] > 1750:
                        left_flg = False
                        continue
                    else:

                        left_flg = True
                        break
                if not left_flg:
                    print(left_img_path)
                    continue

                for right_camara in top_camara_list[ct_index:][::-1]:
                    """找右侧满足条件的点"""

                    right_img_path = ct_img_path.replace(f'{ct_camara}\\{ct_camara}_',
                                                         f'{right_camara}\\{right_camara}_')
                    print(f"右侧文件夹路径：{right_img_path}")

                    right_parent_folder = os.path.basename(os.path.dirname(right_img_path))
                    mask_parent_path = f"app/yolov5_det/ruler/{right_parent_folder}/ruler.jpg"
                    right_mask_rule_img = mask_to_img(right_img_path, mask_parent_path,
                                                                 offset_top)  # 左侧加了尺子的图片

                    response_right_result = width_det(img_info=right_mask_rule_img, model_name=model_name, offset=offset_top)
                    if not response_right_result['shapes']:
                        continue
                        right_flg = False
                    elif right_camara == '06' and response_right_result['shapes'][0]['center_x'] < 45:
                        right_flg = False
                        continue
                    else:
                        right_flg = True
                        break
                if not right_flg:
                    continue
                else:
                    left_center_x, left_center_y = response_left_result['shapes'][0]['center_x'], \
                        response_left_result['shapes'][0]['center_y']
                    right_center_x, right_center_y = response_right_result['shapes'][0]['center_x'], \
                        response_right_result['shapes'][0]['center_y']

                    left_rule_point = config_points[left_camara]['left_distance'] - (
                            mm_px_list[top_camara_list.index(left_camara)]/ left_mask_rule_img.shape[:2][1]) * left_center_x  # 左侧的尺子的点

                    right_rule_point = config_points[right_camara]['left_distance'] - (
                            mm_px_list[top_camara_list.index(right_camara)] / right_mask_rule_img.shape[:2][1]) * right_center_x  # 右侧的尺子的点


                    dir_path = os.path.dirname(os.path.dirname(right_img_path))
                    save_path_dir = os.path.join(dir_path, f'rules_{left_parent_folder}_{right_parent_folder}')

                    os.makedirs(save_path_dir, exist_ok=True)
                    save_img_path = os.path.join(save_path_dir, right_img_path[-7:])
                    # 标注点1
                    cv2.circle(left_mask_rule_img, tuple((int(left_center_x), int(left_center_y))), 5, (0, 0, 255), -1)
                    cv2.putText(left_mask_rule_img, f'{round(left_rule_point, 3)}',
                                tuple((int(left_center_x), int(left_center_y))), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 255),
                                2, cv2.LINE_AA)
                    # 标注点2
                    cv2.circle(right_mask_rule_img, tuple((int(right_center_x), int(right_center_y))), 5, (0, 0, 255), -1)
                    cv2.putText(right_mask_rule_img, f'{round(right_rule_point, 2)}',
                                tuple((int(right_center_x), int(right_center_y))), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 255),
                                2, cv2.LINE_AA)
                    ct_rules = shift_and_concatenate_images(left_mask_rule_img, right_mask_rule_img)

                    real_distance = round(abs(left_rule_point - right_rule_point), 1)

                    # all_result.append(real_distance)
                    all_result[save_img_path] = real_distance * 10

                    cv_imwrite(ct_rules, save_img_path)
            except Exception as e:
                logger.error(traceback.format_exc())
                continue
            finally:
                continue



        # all_result = [int(num * 10) for num in all_result]
        # if not all_result:
        #     continue
        # counter = Counter(all_result)
        # most_common_elements = counter.most_common(1)
        #
        # if most_common_elements:
        #     most_common_element = most_common_elements[0][0]
        # else:
        #     total_count = sum(all_result)
        #     num_elements = len(all_result)
        #     most_common_element = total_count / num_elements
        #
        # all_result = [i for i in all_result if abs(i-most_common_element) < 10]
        # # Find the most common element
        # verDis_qsl = ",".join(str(e) for e in all_result)
        #
        # verDis_qsl = ",".join(str(e) for e in all_result)
        # # Find the most common element
        # verDis_qsl = ",".join(str(e) for e in all_result)
        # update_sql = f"update tblSlabDefectInformationHistory set SlabMWidth1='{most_common_element}',Coordinates='{verDis_qsl}' where ID={ID}"
        # cursor.execute(update_sql)
        # cursor.commit()
        # logger.info(f"数据库{ID.replace(' ', '')}更新成功:测宽长度为{round(most_common_element)}")

        # 假设 all_result 是一个字典，其中键是图片路径，值是对应的宽度
        if not all_result:
            continue

        # 计算最常见的宽度元素
        width_counter = Counter(all_result.values())
        most_common_elements = width_counter.most_common(1)
        if most_common_elements:
            most_common_element = most_common_elements[0][0]
        else:
            # 如果没有明确的最常见元素，计算平均宽度
            total_count = sum(all_result.values())
            num_elements = len(all_result)
            most_common_element = total_count / num_elements

        most_common_element = int(most_common_element)
        # 筛选符合条件的宽度并准备删除不符合条件的图片
        selected_results = {}
        verDis_qsl = []

        for path, width in all_result.items():
            if abs(width - most_common_element) < 5:
                selected_results[path] = width
                verDis_qsl.append(str(width))
            else:
                try:
                    os.remove(path)
                    logger.info(f"The image at {path} has been removed due to width outlier.")
                except Exception as e:
                    logger.error(f"Failed to remove image {path}: {e}")

        # 更新数据库
        if selected_results:
            verDis_qsl = ",".join(verDis_qsl)
            update_sql = f"update tblSlabDefectInformationHistory set SlabMWidth1='{most_common_element}',Coordinates='{verDis_qsl}' where ID={ID}"
            try:
                cursor.execute(update_sql)
                cursor.commit()
                logger.info(f"数据库{ID.replace(' ', '')}更新成功:测宽长度为{round(most_common_element)}")
            except Exception as e:
                logger.error(f"Failed to update database for ID {ID}: {e}")

    except Exception as e:
        logger.error(traceback.format_exc())
        continue
    finally:
        continue

