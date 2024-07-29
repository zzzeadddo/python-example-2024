#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import joblib
from sklearn.cluster import DBSCAN
from ultralytics import YOLOv10  # 导入YOLOv10模型
from PIL import Image
from jdeskew.estimator import get_angle
from jdeskew.utility import rotate
from helper_code import *
import tensorflow as tf  # 导入TensorFlow库
import cv2
from networks.dplinknet import LinkNet34, DLinkNet34, DPLinkNet34
from utils import get_patches, stitch_together
import torch
BATCHSIZE_PER_CARD = 32
from torch.autograd import Variable as V
import re
from collections import defaultdict
import shutil
import glob
import json
import random
from sklearn.linear_model import LinearRegression
################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your digitization model.
def train_models(data_folder, model_folder, verbose):
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    if verbose:
        print('Training the digitization model...')
        print('Extracting features and labels from the data...')

    digitization_features = []

    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i + 1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])
        features = extract_features(record)
        digitization_features.append(features)

    if verbose:
        print('Training the models on the data...')

    digitization_features_array = np.vstack(digitization_features)
    digitization_labels = np.array([np.mean(f) for f in digitization_features])
    digitization_model = LinearRegression().fit(digitization_features_array, digitization_labels)

    os.makedirs(model_folder, exist_ok=True)
    classification_model = None
    classes = None
    save_models(model_folder, digitization_model, classification_model, classes)

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_models(model_folder, verbose):
    # 加载YOLO模型
    yolo_model_path = os.path.join(model_folder, 'best.pt')

    # 加载分类模型
    classification_model_path = os.path.join(model_folder, 'best_model_II.h5')
    # 加载LinkNet模型
    linknet_model_path = os.path.join(model_folder, 'dibco_dplinknet34.th')

    if verbose:
        print(f"Loaded YOLO model from {yolo_model_path}")
        print(f"Loaded classification model from {classification_model_path}")
        print(f"Loaded LinkNet model from {linknet_model_path}")

    # 将三个模型打包在一个字典中
    digitization_model = {
        'yolo_model': yolo_model_path,
        'classification_model': classification_model_path,
        'linknet_model': linknet_model_path
    }
    classification_model = None
    # 返回打包好的模型字典和分类模型
    return digitization_model, classification_model

# Run your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function. If you did not train one of the models, then you can return None for the model.
def run_models(record, digitization_model, classification_model, verbose):
    try:
        # Load the dimensions of the signal.
        header_file = get_header_file(record)
        header = load_text(header_file)
        num_samples = get_num_samples(header)
        num_signals = get_num_signals(header)
        print(f"Number of samples: {num_samples}, number of signals: {num_signals}")

        base_path = record
        base_directory = os.path.dirname(base_path)
        file_namezeze = os.path.basename(base_path)

        search_pattern = os.path.join(base_directory, f'*{file_namezeze}*')
        files = [f for f in glob.glob(search_pattern) if f.lower().endswith(('.jpg', '.jpeg', '.png', 'tiff'))]

        if not files:
            raise FileNotFoundError("No files found matching the pattern.")

        random.shuffle(files)
        file_to_process = files[0]
        process_image(file_to_process)
        output_folder_base = os.path.join(base_path, "compressed")
        processed_output_folder_base = os.path.join(base_path, "processed")
        text_crop_folder_base = os.path.join(base_path, "text_crops")

        model_path = digitization_model['yolo_model']
        classification_model = tf.keras.models.load_model(digitization_model['classification_model'])

        size_mapping = compress_image(file_to_process, output_folder_base)
        start, end, step = 0.05, 0.85, 0.05
        thresholds = [i for i in np.arange(start, end + step, step)]

        process_images_with_yolo(model_path, output_folder_base, processed_output_folder_base, text_crop_folder_base,
                                 size_mapping, thresholds, classification_model, file_namezeze, file_to_process)

        print("All images have been compressed and processed successfully.")

        TILE_SIZE = 256
        DEEP_NETWORK_NAME = "DPLinkNet34"

        subdirs = [os.path.join(processed_output_folder_base, subdir) for subdir in
                   os.listdir(processed_output_folder_base)
                   if os.path.isdir(os.path.join(processed_output_folder_base, subdir))]

        for img_indir in subdirs:
            print("Image input directory:", img_indir)
            img_outdir = os.path.join(img_indir, "Binarized")
            os.makedirs(img_outdir, exist_ok=True)
            print("Image output directory:", img_outdir)

            img_list = sorted(os.listdir(img_indir))

            solver = TTAFrame(DPLinkNet34)
            weights_path = digitization_model['linknet_model']
            print("Now loading the model weights:", weights_path)
            solver.load(weights_path)

            for idx, img_name in enumerate(img_list):
                if os.path.isdir(os.path.join(img_indir, img_name)):
                    continue

                print("Now processing image:", img_name)
                fname, _ = os.path.splitext(img_name)
                img_input = os.path.join(img_indir, img_name)
                img_output = os.path.join(img_outdir, f"{fname}-{DEEP_NETWORK_NAME}.tiff")

                img = cv2.imread(img_input)
                if img is None:
                    print(f"Error reading image {img_input}, skipping...")
                    continue

                img, original_size = pad_image(img, TILE_SIZE)
                locations, patches = get_patches_dynamic(img, TILE_SIZE)
                masks = [solver.test_one_img_from_path(patch) for patch in patches]
                prediction = stitch_together(locations, masks, tuple(img.shape[0:2]), TILE_SIZE, TILE_SIZE)
                prediction[prediction >= 5.0] = 255
                prediction[prediction < 5.0] = 0
                prediction = unpad_image(prediction, original_size)
                cv2.imwrite(img_output, prediction.astype(np.uint8))

            print("Finished processing directory:", img_indir)

        print("All directories have been processed successfully.")

        processed_output_folder_base = os.path.join(base_path, "processed")
        subdirs = [os.path.join(processed_output_folder_base, subdir) for subdir in
                   os.listdir(processed_output_folder_base)
                   if os.path.isdir(os.path.join(processed_output_folder_base, subdir))]

        for img_indir in subdirs:
            input_folder_path = os.path.join(img_indir, 'Binarized')
            units_info_path = os.path.join(img_indir, 'units_info.json')

            if not os.path.exists(input_folder_path) or not os.path.exists(units_info_path):
                print(f"Required paths do not exist in {img_indir}")
                continue

            with open(units_info_path, 'r') as file:
                units_info = json.load(file)

            processed_images = process_all_images_in_folder(input_folder_path)
            lead_counter = {}

            for filename, img in processed_images:
                extract_and_save_ecg_signal(img, filename, input_folder_path, units_info, num_samples, lead_counter)

            grouped_folder = os.path.join(input_folder_path, 'Grouped')
            group_and_rename_files(os.path.join(input_folder_path, 'Digitalization'), grouped_folder)
            integrated_signal = integrate_grouped_files_to_p_signal(grouped_folder, input_folder_path)

        signal = integrated_signal
        labels = None

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Returning zero matrix due to error.")
        signal = np.zeros((num_samples, num_signals))
        labels = None

    return signal, labels

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract features.
def extract_features(record):
    images = load_images(record)
    mean = 0.0
    std = 0.0
    for image in images:
        image = np.asarray(image)
        mean += np.mean(image)
        std += np.std(image)
    return np.array([mean, std])


# Save your trained models.
def save_models(model_folder, digitization_model=None, classification_model=None, classes=None):
    if digitization_model is not None:
        d = {'model': digitization_model}
        filename = os.path.join(model_folder, 'digitization_model.sav')
        joblib.dump(d, filename, protocol=0)

    if classification_model is not None:
        d = {'model': classification_model, 'classes': classes}
        filename = os.path.join(model_folder, 'classification_model.sav')
        joblib.dump(d, filename, protocol=0)

def process_image(file_name):
    file_path = os.path.join(file_name)
    print(f"Processing image {file_name}...")
    # 读取图片
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)  # 使用IMREAD_COLOR读取彩色图片

    if image is None:
        print(f"Failed to load image {file_path}")
        return

    # 将图片转换为灰度图像以估计倾斜角度
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 估计倾斜角度
    angle = get_angle(gray_image)
    print(f"Estimated skew angle for {file_name}: {angle} degrees")

    # 旋转原图片以纠正倾斜
    corrected_image = rotate(image, angle)

    # 保存处理后的图片，替换原图
    cv2.imwrite(file_path, corrected_image)
    print(f"Corrected image saved to {file_path}")


def compress_image(file, output_dir, quality=85, max_size=(640, 640)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    size_mapping = {}
    if os.path.isfile(file) and file.lower().endswith(('.jpg', '.jpeg', '.png')):
        file_name = os.path.basename(file)
        output_path = os.path.join(output_dir, file_name)
        img = Image.open(file)
        original_size = img.size
        img.thumbnail(max_size)
        img.save(output_path, quality=quality)
        compressed_size = img.size
        size_mapping[file_name] = (original_size, compressed_size)

    return size_mapping


def process_images_with_yolo(model_path, input_folder, output_base_folder, text_crop_folder_base, size_mapping,
                             thresholds, classification_model, file_namezeze, file):
    model = YOLOv10(model_path)  # 加载YOLO模型

    lead_labels = [
        ["I", "II", "III"],
        ["aVR", "aVL", "aVF"],
        ["V1", "V2", "V3"],
        ["V4", "V5", "V6"]
    ]

    for filename in os.listdir(input_folder):  # 遍历输入文件夹中的所有文件
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # 检查文件是否为图像文件
            image_path = os.path.join(input_folder, filename)  # 获取图像文件路径
            print(f"Processing file: {image_path}")

            if not os.path.exists(image_path):
                print(f"File not found: {image_path}")
                continue

            subfolder = create_subfolder(output_base_folder, filename)  # 创建子文件夹
            text_crop_subfolder = create_subfolder(text_crop_folder_base, filename)  # 创建文本裁剪子文件夹

            best_threshold = None
            max_min_count = 0

            for threshold in thresholds:
                results = model.predict(image_path, conf=threshold)

                leads = []
                texts = []

                for result in results[0].boxes.data:
                    if result[5] == 0.0:
                        leads.append(result)
                    elif result[5] == 1.0:
                        texts.append(result)

                min_count = max(len(leads), len(texts))

                if min_count > max_min_count:
                    max_min_count = min_count
                    best_threshold = threshold
                print(f"Threshold: {threshold} - maxmincount: {max_min_count} for {filename}")

            if best_threshold is not None:
                print(f"Best threshold for {filename}: {best_threshold} with min_count: {max_min_count}")
                results = model.predict(image_path, conf=best_threshold)

                results[0].boxes.data = non_max_suppression(results[0].boxes.data.tolist(), overlap_threshold=0.8)

                leads = []
                texts = []

                for result in results[0].boxes.data:
                    if result[5] == 0.0:
                        leads.append(result)
                    elif result[5] == 1.0:
                        texts.append(result)

                max_min_count = max(len(leads), len(texts))
                print(f"0.8的max_min_count: {max_min_count}")
                leads = leads[:max_min_count]
                texts = texts[:max_min_count]

                print(f"Threshold: {best_threshold} - Leads: {len(leads)}, Texts: {len(texts)} for {filename}")

                proportional_boxes = []
                mapped_boxes = []

                original_size, compressed_size = size_mapping[filename]

                lead_widths = []
                text_widths = []

                for result in results[0].boxes.data:
                    x1, y1, x2, y2 = result[:4]
                    confidence = result[4]
                    class_id = int(result[5])

                    x1_ratio = x1 / compressed_size[0]
                    y1_ratio = y1 / compressed_size[1]
                    x2_ratio = x2 / compressed_size[0]
                    y2_ratio = y2 / compressed_size[1]

                    proportional_boxes.append((x1_ratio, y1_ratio, x2_ratio, y2_ratio, confidence, class_id))

                    x1_mapped = x1_ratio * original_size[0]
                    y1_mapped = y1_ratio * original_size[1]
                    x2_mapped = x2_ratio * original_size[0]
                    y2_mapped = y2_ratio * original_size[1]

                    mapped_boxes.append((x1_mapped, y1_mapped, x2_mapped, y2_mapped, confidence, class_id))

                    box_width = x2_mapped - x1_mapped
                    if class_id == 0:
                        lead_widths.append(box_width)
                    elif class_id == 1:
                        text_widths.append(box_width)

                min_x1 = min([box[0] for box in mapped_boxes])
                max_x2 = max([box[2] for box in mapped_boxes])
                column_width = (max_x2 - min_x1) / 4

                columns = [[] for _ in range(4)]
                text_boxes = []

                for box in mapped_boxes:
                    center_x = (box[0] + box[2]) / 2
                    center_y = (box[1] + box[3]) / 2
                    if box[5] == 0:
                        col_index = int((center_x - min_x1) / column_width)
                        col_index = min(col_index, 3)
                        columns[col_index].append((center_y, box, center_x, center_y))
                    else:
                        text_boxes.append((center_y, box, center_x, center_y))

                # original_image_path = os.path.join(input_folder.replace(file_namezeze, '').replace('\\compressed', ''),
                #                                    filename)
                original_image_path = os.path.join(os.path.dirname(os.path.dirname(input_folder)), filename)
                print(f"Original image path: {original_image_path}")

                if not os.path.exists(original_image_path):
                    print(f"Original file not found: {original_image_path}")
                    continue

                original_image = Image.open(original_image_path)
                # empty_image = Image.new("RGB", original_image.size, "white")  # 创建空白图像
                # draw_original = ImageDraw.Draw(original_image)  # 创建原始图像的绘图对象
                # draw_empty = ImageDraw.Draw(empty_image)  # 创建空白图像的绘图对象
                # font = ImageFont.load_default()  # 加载默认字体

                # 计算第2、3列最上面6个导联框的宽度平均值
                lead_widths_col_2_3 = []
                for col_idx in [1, 2]:
                    column = columns[col_idx]
                    print(len(column))
                    for lead_idx in range(min(3, len(column))):
                        _, box, _, _ = column[lead_idx]
                        lead_widths_col_2_3.append(box[2] - box[0])
                print('lead_widths_col_2_3 is', lead_widths_col_2_3)

                # 剔除与平均值相差较大的异常值（使用标准差方法）
                if lead_widths_col_2_3:
                    mean_width = np.mean(lead_widths_col_2_3)
                    std_width = np.std(lead_widths_col_2_3)
                    filtered_widths_col_2_3 = [width for width in lead_widths_col_2_3 if
                                               abs(width - mean_width) <= 2 * std_width]
                else:
                    filtered_widths_col_2_3 = []

                print('Filtered lead_widths_col_2_3 is', filtered_widths_col_2_3)

                if filtered_widths_col_2_3:
                    avg_lead_width_col_2_3 = np.mean(filtered_widths_col_2_3)
                else:
                    avg_lead_width_col_2_3 = 0
                print(f"第2、3列最上面6个导联框的宽度平均值（去除异常值后）是 {avg_lead_width_col_2_3}")

                # 调整第1列最上面3个导联框的宽度
                if avg_lead_width_col_2_3 > 0:
                    column_1 = columns[0]
                    for lead_idx in range(min(3, len(column_1))):
                        _, box, _, _ = column_1[lead_idx]
                        lead_width_col_1 = box[2] - box[0]
                        print(
                            f"Column 1 Lead {lead_idx} width: {lead_width_col_1}, Avg width: {avg_lead_width_col_2_3}")
                        if abs(lead_width_col_1 - avg_lead_width_col_2_3) > 0.05 * avg_lead_width_col_2_3:
                            print('需要更改左边坐标！！！')
                            new_x1 = box[2] - avg_lead_width_col_2_3
                            column_1[lead_idx] = (_, (new_x1, box[1], box[2], box[3]), _, _)
                            print(f"New coordinates: {column_1[lead_idx]}")
                        else:
                            print('不需要更改左边坐标！！！')
                columns[0] = column_1

                # 记录一下最左边的坐标
                leftleft = [box[0] for _, box, _, _ in column_1[:min(3, len(column_1))]]
                print(f'最左边的坐标: {leftleft}')
                leftleft = min(leftleft)

                time_per_pixel, voltage_per_pixel, dpi = calculate_units(avg_lead_width_col_2_3)
                save_units_to_file(output_base_folder, filename, time_per_pixel, voltage_per_pixel, dpi)
                print(f"For {filename}:")
                print(f"  Average lead width: {avg_lead_width_col_2_3} pixels")
                print(f"  Time per pixel: {time_per_pixel} seconds/pixel")
                print(f"  Voltage per pixel: {voltage_per_pixel} mV/pixel")

                # 假设 text_boxes 是一个包含多个四元组的列表，其中每个四元组的第二个元素是一个包含(x_min, y_min, x_max, y_max)的元组
                text_crop_paths = []
                if max_min_count < 13:
                    max_min_count = 13
                print(f"纠正后的max_min_count: {max_min_count}")

                # 根据高度(y_min)对 text_boxes 进行排序
                text_boxes_sorted = sorted(text_boxes, key=lambda x: x[1][1])

                for idx, (_, box, _, _) in enumerate(text_boxes_sorted[-(max_min_count - 12):]):
                    crop_box = original_image.crop((box[0], box[1], box[2], box[3]))
                    crop_output_path = os.path.join(text_crop_subfolder, f"text_crop_{idx + 13}_{filename}")
                    crop_box.save(crop_output_path)
                    text_crop_paths.append(crop_output_path)
                    print(f"Cropped text box saved to {crop_output_path}")

                def preprocess_image(image_path, target_size=(224, 224)):
                    img = load_img(image_path, target_size=target_size)
                    img_array = img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array /= 255.0
                    return img_array

                lead_names = {
                    0: 'I',
                    1: 'II',
                    2: 'III',
                    3: 'V1',
                    4: 'V2',
                    5: 'V3',
                    6: 'V4',
                    7: 'V5',
                    8: 'V6',
                    9: 'aVF',
                    10: 'aVL',
                    11: 'aVR'
                }

                def predict_image(image_path, model):
                    img_array = preprocess_image(image_path)
                    predictions = model.predict(img_array)
                    predicted_class = np.argmax(predictions, axis=1)[0]
                    predicted_lead = lead_names[predicted_class]
                    return predicted_lead

                def predict_images_in_folder(image_paths, model):
                    predictions = {}
                    for image_path in image_paths:
                        predicted_lead = predict_image(image_path, model)
                        predictions[image_path] = predicted_lead
                        print(f"Predicted lead for {os.path.basename(image_path)} is: {predicted_lead}")
                    return predictions

                predictions = predict_images_in_folder(text_crop_paths, classification_model)

                # # 提取所有文本框的下边界值
                # bottom_ys = np.array([box[3] for _, box, _, _ in text_boxes]).reshape(-1, 1)
                #
                # # 使用DBSCAN进行聚类
                # dbscan = DBSCAN(eps=10, min_samples=1).fit(bottom_ys)
                #
                # # 获取聚类结果
                # labels = dbscan.labels_
                #
                # # 按照聚类结果分组
                # text_boxes_by_line = {}
                # for label, (_, box, _, _) in zip(labels, text_boxes):
                #     if label not in text_boxes_by_line:
                #         text_boxes_by_line[label] = []
                #     text_boxes_by_line[label].append(box[3])
                #
                # # 输出分组数量
                # num_groups = len(set(labels)) - (1 if -1 in labels else 0)
                # print(f"Number of groups: {num_groups}")  # 不计入噪声点
                #
                # avg_bottom_y_values = {}
                # for key, values in text_boxes_by_line.items():
                #     avg_bottom_y_values[key] = np.mean(values)
                #     print(
                #         f"Group {key} contains {len(values)} text boxes, average bottom y: {avg_bottom_y_values[key]}")

                # 提取所有文本框的下边界值
                bottom_ys = np.array([box[3] for _, box, _, _ in text_boxes]).reshape(-1, 1)
                # print(f"Bottom y values: {bottom_ys}")

                # 计算eps值为最大值和最小值之间差值的四分之一
                max_y = np.max(bottom_ys)
                min_y = np.min(bottom_ys)
                eps = (max_y - min_y) * 0.1428
                print(f"Calculated eps: {eps}")

                # 使用DBSCAN进行聚类
                dbscan = DBSCAN(eps=eps, min_samples=1).fit(bottom_ys)

                # 获取聚类结果
                labels = dbscan.labels_

                # 按照聚类结果分组
                text_boxes_by_line = {}
                for label, (_, box, _, _) in zip(labels, text_boxes):
                    if label not in text_boxes_by_line:
                        text_boxes_by_line[label] = []
                    text_boxes_by_line[label].append(box[3])

                # 输出分组数量
                num_groups = len(set(labels)) - (1 if -1 in labels else 0)
                print(f"Number of groups: {num_groups}")  # 不计入噪声点

                # 滤除类内离群点
                filtered_text_boxes_by_line = {}
                for label, boxes in text_boxes_by_line.items():
                    median_y = np.median(boxes)
                    threshold = 20  # 定义离群点阈值，可以根据需要调整
                    filtered_boxes = [y for y in boxes if abs(y - median_y) <= threshold]
                    filtered_text_boxes_by_line[label] = filtered_boxes

                avg_bottom_y_values = {}
                for key, values in filtered_text_boxes_by_line.items():
                    avg_bottom_y_values[key] = np.mean(values)
                    print(
                        f"Group {key} contains {len(values)} text boxes, average bottom y: {avg_bottom_y_values[key]}")


                zero_lines = {}
                for key, avg_y in avg_bottom_y_values.items():
                    if num_groups > 4:
                        zero_line_y = avg_y - dpi * 0.32  # 统一的矫正inch为0.32
                    else:
                        if len(text_boxes_by_line[key]) <= 2:
                            zero_line_y = avg_y - dpi * 0.37  # 矫正的inch为0.37
                        else:
                            zero_line_y = avg_y - dpi * 0.32  # 矫正的inch为0.32
                    zero_lines[key] = zero_line_y
                    print(f"Zero line for group {key} at y: {zero_line_y}")

                lead_count = {}
                for col_idx, column in enumerate(columns):
                    column.sort()
                    for lead_idx, (_, box, center_x, center_y) in enumerate(column):
                        if lead_idx < len(lead_labels[col_idx]):
                            label = lead_labels[col_idx][lead_idx]

                            # 计算每个lead box与其最近的零刻度线之间的距离
                            closest_zero_line = min(zero_lines.values(), key=lambda z: abs(center_y - z))
                            ratio = box[3] - closest_zero_line

                            # 检查ratio并调整box的高度
                            if ratio < 0:
                                # ratio为负，调整box的底边
                                new_y3 = closest_zero_line + (box[3] - box[1]) * 1.18
                                box = (box[0], box[1], box[2], new_y3)
                                ratio = box[3] - closest_zero_line
                            elif ratio > (box[3] - box[1]):
                                # ratio为正且超过box的高度，调整box的上边
                                new_y1 = box[3] - ratio * 1.18
                                box = (box[0], new_y1, box[2], box[3])
                                ratio = box[3] - closest_zero_line

                            lead_crop_box = original_image.crop((box[0], box[1], box[2], box[3]))
                            lead_crop_name = f"{label}_ratio_{ratio:.2f}"
                            if lead_crop_name in lead_count:
                                lead_count[lead_crop_name] += 1
                                lead_crop_name += f"_{lead_count[lead_crop_name]}"
                            else:
                                lead_count[lead_crop_name] = 1
                            lead_crop_output_path = os.path.join(subfolder, f"lead_crop_{lead_crop_name}_{filename}")
                            lead_crop_box.save(lead_crop_output_path)
                            print(f"Cropped lead box saved to {lead_crop_output_path}")

                remaining_leads = []
                for col in columns:
                    remaining_leads.extend(col)
                remaining_leads.sort(key=lambda x: x[0])

                for idx, prediction in enumerate(predictions.values()):
                    if idx + 12 < len(remaining_leads):
                        lead_info = remaining_leads[idx + 12]
                        print('lead_info is', lead_info)

                        # 检查lead_info是否有足够的值来解包
                        if len(lead_info) != 4:
                            print(f"Error: lead_info does not have 4 elements: {lead_info}")
                            continue

                        _, box_info, center_x, center_y = lead_info

                        # 检查box_info是否有足够的值来解包
                        if len(box_info) != 6:
                            print(f"Error: box_info does not have 6 elements: {box_info}")
                            continue

                        x1, y1, x2, y2, confidence, class_id = box_info

                        label = prediction
                        x1, x2 = leftleft, max_x2

                        remaining_leads[idx + 12] = (_, (x1, y1, x2, y2, confidence, class_id), center_x, center_y)

                        # 计算每个lead box与其最近的零刻度线之间的距离
                        closest_zero_line = min(zero_lines.values(), key=lambda z: abs(y2 - z))
                        ratio = y2 - closest_zero_line

                        lead_crop_box = original_image.crop((x1, y1, x2, y2))
                        lead_crop_name = f"{label}_2_ratio_{ratio:.2f}"
                        if lead_crop_name in lead_count:
                            lead_count[lead_crop_name] += 1
                            lead_crop_name += f"_{lead_count[lead_crop_name]}"
                        else:
                            lead_count[lead_crop_name] = 1
                        lead_crop_output_path = os.path.join(subfolder, f"lead_crop_{lead_crop_name}_{filename}")
                        lead_crop_box.save(lead_crop_output_path)
                        print(f"Cropped lead box saved to {lead_crop_output_path}")

                print(f"Text box widths for {filename}: {text_widths}")
                print(f"Lead box widths for {filename}: {lead_widths}")

def create_subfolder(base_folder, filename):
    # 根据给定的文件名在基础文件夹下创建子文件夹
    subfolder = os.path.join(base_folder, os.path.splitext(filename)[0])
    if not os.path.exists(subfolder):  # 如果子文件夹不存在，则创建它
        os.makedirs(subfolder)
    return subfolder

def non_max_suppression(boxes, overlap_threshold=0.98):
    # 非极大值抑制，移除重叠的边界框
    if len(boxes) == 0:  # 如果没有边界框，返回空列表
        return []

    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)  # 按照置信度排序

    picked_boxes = []  # 选择的边界框列表
    while boxes:
        current_box = boxes.pop(0)  # 取出置信度最高的边界框
        picked_boxes.append(current_box)
        boxes = [box for box in boxes if box[5] != current_box[5] or box_iou(current_box, box) < overlap_threshold]  # 移除与当前边界框重叠的边界框

    return picked_boxes  # 返回选择的边界框

def calculate_units(width_in_pixels, time_in_seconds=2.5, dpi=200):
    # 计算像素单位
    width_in_mm = time_in_seconds / 0.2 * 5  # 将时间转换为毫米
    pixels_per_mm = width_in_pixels / width_in_mm  # 计算每毫米像素数
    time_per_pixel = 0.04 / pixels_per_mm  # 计算每像素时间
    voltage_per_pixel = 0.1 / pixels_per_mm  # 计算每像素电压

    inches_per_mm = 1 / 25.4  # 毫米转换为英寸
    dpi_calculated = pixels_per_mm / inches_per_mm  # 计算DPI

    print('pixels_per_mm is', pixels_per_mm)
    print('DPI is', dpi_calculated)

    return time_per_pixel, voltage_per_pixel, dpi_calculated  # 返回每像素时间、电压和DPI


def save_units_to_file(folder, filename, time_per_pixel, voltage_per_pixel, dpi_calculated):
    # 保存每像素时间、电压和DPI信息到文件
    subfolder = create_subfolder(folder, filename)
    result_file_path = os.path.join(subfolder, "units_info.json")
    units_info = {
        "time_per_pixel": time_per_pixel,
        "voltage_per_pixel": voltage_per_pixel,
        "dpi_calculated": dpi_calculated
    }
    with open(result_file_path, 'w') as f:
        json.dump(units_info, f, indent=4)
    print(f"Units info saved to {result_file_path}")

def box_iou(boxA, boxB):
    # 计算两个边界框之间的交并比（IoU）
    xA = max(boxA[0], boxB[0])  # 计算交集区域的左上角x坐标
    yA = max(boxA[1], boxB[1])  # 计算交集区域的左上角y坐标
    xB = min(boxA[2], boxB[2])  # 计算交集区域的右下角x坐标
    yB = min(boxA[3], boxB[3])  # 计算交集区域的右下角y坐标

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)  # 计算交集面积
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)  # 计算boxA的面积
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)  # 计算boxB的面积

    iou = interArea / float(boxAArea + boxBArea - interArea)  # 计算IoU
    return iou  # 返回IoU值

class TTAFrame():
    def __init__(self, net):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

    def test_one_img_from_path(self, path, evalmode=True):
        if evalmode:
            self.net.eval()
        batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        if batchsize >= 8:
            return self.test_one_img_from_path_1(path)
        elif batchsize >= 4:
            return self.test_one_img_from_path_2(path)
        elif batchsize >= 2:
            return self.test_one_img_from_path_4(path)

    def test_one_img_from_path_8(self, path):
        img = np.array(path)  # .transpose(2,0,1)[None]
        # img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        return mask2

    def test_one_img_from_path_4(self, path):
        img = np.array(path)  # .transpose(2,0,1)[None]
        # img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        return mask2

    def test_one_img_from_path_2(self, path):
        img = np.array(path)  # .transpose(2,0,1)[None]
        # img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = img3.transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).cuda())
        img6 = img4.transpose(0, 3, 1, 2)
        img6 = np.array(img6, np.float32) / 255.0 * 3.2 - 1.6
        img6 = V(torch.Tensor(img6).cuda())

        maska = self.net.forward(img5).squeeze().cpu().data.numpy()  # .squeeze(1)
        maskb = self.net.forward(img6).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask3

    def test_one_img_from_path_1(self, path):
        img = np.array(path)  # .transpose(2,0,1)[None]
        # img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = np.concatenate([img3, img4]).transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).cuda())

        mask = self.net.forward(img5).squeeze().cpu().data.numpy()  # .squeeze(1)
        mask1 = mask[:4] + mask[4:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask3

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

def pad_image(image, min_size):
    h, w = image.shape[:2]
    if h < min_size or w < min_size:
        pad_h = max(0, min_size - h)
        pad_w = max(0, min_size - w)
        padded_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return padded_image, (h, w)  # 返回填充后的图像和原始尺寸
    return image, (h, w)  # 如果图像尺寸已经满足要求，返回原图像和原始尺寸

def get_patches_dynamic(image, min_tile_size):
    h, w = image.shape[:2]
    if h < min_tile_size or w < min_tile_size:
        tile_size = min(h, w)
    else:
        tile_size = min_tile_size
    return get_patches(image, tile_size, tile_size)

def unpad_image(image, original_size):
    h, w = original_size
    return image[:h, :w]  # 截取填充前的图像部分

def resize_and_pad(signal, target_length, pad_value=np.nan, position='start'):
    resized_signal = cv2.resize(signal.reshape(-1, 1), (1, target_length), interpolation=cv2.INTER_LINEAR).flatten()
    padding = np.full(target_length, pad_value)
    if position == 'start':
        combined = np.concatenate([resized_signal, padding, padding, padding])
    elif position == 'middle':
        combined = np.concatenate([padding, resized_signal, padding, padding])
    elif position == 'end':
        combined = np.concatenate([padding, padding, padding, resized_signal])
    elif position == 'third':
        combined = np.concatenate([padding, padding, resized_signal, padding])
    else:
        combined = resized_signal
    return combined

def process_ecg_signal(signal, lead_type, numsamples):
    quarter_samples = numsamples // 4
    if lead_type in ["I", "II", "III"]:
        return resize_and_pad(signal, quarter_samples, position='start')
    elif lead_type in ["aVR", "aVL", "aVF"]:
        return resize_and_pad(signal, quarter_samples, position='middle')
    elif lead_type in ["V1", "V2", "V3"]:
        return resize_and_pad(signal, quarter_samples, position='third')
    elif lead_type in ["V4", "V5", "V6"]:
        return resize_and_pad(signal, quarter_samples, position='end')
    else:
        return cv2.resize(signal.reshape(-1, 1), (1, numsamples), interpolation=cv2.INTER_LINEAR).flatten()

def calculate_adaptive_threshold(img):
    height, width = img.shape
    initial_threshold = (height * width) * 0.001
    mean_intensity = np.mean(img)
    adaptive_threshold = initial_threshold * (mean_intensity / 255.0)
    return max(adaptive_threshold, 1)

def process_image_and_save(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not open or find the image {img_path}")
        return None, None

    height, width = img.shape
    remove_height_ratio = 0.1
    remove_width_ratio = 0.1
    remove_height = int(height * remove_height_ratio)
    remove_width = int(width * remove_width_ratio)
    img[height - remove_height:height, 0:remove_width] = 255

    img_inv = cv2.bitwise_not(img)
    contours, _ = cv2.findContours(img_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = np.zeros_like(img_inv)

    adaptive_threshold = calculate_adaptive_threshold(img)

    for contour in contours:
        if cv2.contourArea(contour) > adaptive_threshold:
            cv2.drawContours(img_contours, [contour], -1, (255), thickness=cv2.FILLED)

    img_result = cv2.bitwise_not(img_contours)

    edge_margin = 10
    edges = [
        ((0, 0), (width, edge_margin)),
        ((0, height - edge_margin), (width, height)),
        ((0, 0), (edge_margin, height)),
        ((width - edge_margin, 0), (width, height))
    ]
    for (start, end) in edges:
        edge_contours, _ = cv2.findContours(img_result[start[1]:end[1], start[0]:end[0]], cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
        for contour in edge_contours:
            cv2.drawContours(img_result[start[1]:end[1], start[0]:end[0]], [contour], -1, (255), thickness=cv2.FILLED)

    return img_result, os.path.basename(img_path)

def process_all_images_in_folder(input_folder):
    processed_images = []

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if os.path.isfile(file_path) and filename.lower().endswith((".png", ".jpg", ".jpeg", ".tiff")):
            processed_img, processed_filename = process_image_and_save(file_path)
            if processed_img is not None:
                processed_images.append((processed_filename, processed_img))

    return processed_images


def extract_and_save_ecg_signal(image, filename, output_folder, units_info, numsamples, lead_counter):
    # 定义输出文件夹路径
    digitalization_folder_path = os.path.join(output_folder, "Digitalization")
    os.makedirs(digitalization_folder_path, exist_ok=True)

    # 反转图像颜色
    img = cv2.bitwise_not(image)

    # 从文件名中提取crop值
    match_crop = re.search(r"_crop_(\w+)_ratio_", filename)
    if not match_crop:
        print("Could not extract crop value from filename")
        return
    lead_type = match_crop.group(1)

    # 从文件名中提取中心线距离
    match_ratio = re.search(r"_ratio_(\d+\.\d+)_", filename)
    if not match_ratio:
        print("Could not extract centerline distance from filename")
        return
    centerline_distance = float(match_ratio.group(1))


    # 更新lead_counter
    if lead_type not in lead_counter:
        lead_counter[lead_type] = 0
    lead_counter[lead_type] += 1
    count = lead_counter[lead_type]

    # 获取图像高度和宽度
    height, width = img.shape

    # 计算中心线的y坐标，并确保其大于0
    center_y = max(height - centerline_distance, 0)

    def weighted_average_path(img, center_y):
        height, width = img.shape
        path = np.full(width, np.nan)

        for col in range(width):
            max_distance = 0
            min_distance = height
            farthest_row = center_y
            closest_row = center_y
            found = False
            for row in range(2, height - 3):
                if img[row, col] > 0:
                    distance = abs(row - center_y)
                    if distance > max_distance:
                        max_distance = distance
                        farthest_row = row
                    if distance < min_distance:
                        min_distance = distance
                        closest_row = row
                    found = True
            if found:
                if col < width * 0.2:
                    path[col] = farthest_row if farthest_row < center_y else closest_row
                else:
                    path[col] = farthest_row

        for col in range(1, width - 1):
            if np.isnan(path[col]):
                left = col - 1
                right = col + 1
                while left >= 0 and np.isnan(path[left]):
                    left -= 1
                while right < width and np.isnan(path[right]):
                    right += 1
                if left >= 0 and right < width:
                    path[col] = (path[left] + path[right]) / 2
                elif left >= 0:
                    path[col] = path[left]
                elif right < width:
                    path[col] = path[right]

        if np.isnan(path[0]):
            path[0] = path[1] if not np.isnan(path[1]) else center_y
        if np.isnan(path[-1]):
            path[-1] = path[-2] if not np.isnan(path[-2]) else center_y

        path = path.astype(int)
        return path

    path = weighted_average_path(img, center_y)

    path = path[3:-3]

    voltage_per_pixel = units_info["voltage_per_pixel"]

    voltage_values = (center_y - path) * voltage_per_pixel

    processed_signal = process_ecg_signal(voltage_values, lead_type, numsamples)

    signal_file_path = os.path.join(digitalization_folder_path, f"{lead_type}_{count}.csv")
    try:
        np.savetxt(signal_file_path, processed_signal, delimiter=",", comments="")
        print(f"Saved digitalized signal to {signal_file_path}")
    except Exception as e:
        print(f"Failed to save digitalized signal: {e}")

def group_and_rename_files(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    files = os.listdir(input_directory)
    grouped_files = defaultdict(list)

    for file in files:
        filename = os.path.basename(file)
        key = filename.split('_')[0]
        grouped_files[key].append(file)

    for key, group in grouped_files.items():
        if len(group) == 1:
            new_name = f"{key}.csv"
            old_file = group[0]
        elif len(group) == 2:
            new_name = f"{key}.csv"
            old_file = max(group, key=len)
        else:
            new_name = f"{key}.csv"
            old_file = min(group, key=len)

        old_path = os.path.join(input_directory, old_file)
        new_path = os.path.join(output_directory, new_name)

        if os.path.exists(new_path):
            base, ext = os.path.splitext(new_name)
            counter = 1
            new_path = os.path.join(output_directory, f"{base}_{counter}{ext}")
            while os.path.exists(new_path):
                counter += 1
                new_path = os.path.join(output_directory, f"{base}_{counter}{ext}")

        shutil.move(old_path, new_path)
        print(f"Moved and renamed {old_path} to {new_path}")

    for key, group in grouped_files.items():
        print(f"Group {key}:")
        for file in group:
            print(f"  {os.path.join(input_directory, file)}")

def integrate_grouped_files_to_p_signal(grouped_folder, output_folder):
    lead_order = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    integrated_signal = []

    for lead in lead_order:
        filename = f"{lead}.csv"
        file_path = os.path.join(grouped_folder, filename)
        if os.path.exists(file_path):
            signal = np.loadtxt(file_path, delimiter=',')
            integrated_signal.append(signal)
        else:
            print(f"Warning: {filename} not found in {grouped_folder}")

    integrated_signal = np.array(integrated_signal).T

    return integrated_signal
