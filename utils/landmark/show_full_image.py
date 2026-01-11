import numpy as np


def get_all_landmarks(current_id, curr_landmarks_no_postprocessing, curr_landmarks, full_image_landmarks_dict, cropped_images_dict):
    landmarks_top_list = []
    landmarks_bottom_list = []
    if current_id.endswith("_top"):
        new_current_id = current_id.replace("_top", "")
        if new_current_id in list(cropped_images_dict.keys()) and cropped_images_dict[new_current_id] == "True":
            landmarks_top_list.append(curr_landmarks_no_postprocessing)
            landmarks_top_list.append(curr_landmarks)
            full_image_landmarks_dict[current_id] = landmarks_top_list
    if current_id.endswith("_bottom"):
        new_current_id = current_id.replace("_bottom", "")
        if new_current_id in list(cropped_images_dict.keys()) and cropped_images_dict[new_current_id] == "True":
            landmarks_bottom_list.append(curr_landmarks_no_postprocessing)
            landmarks_bottom_list.append(curr_landmarks)
            full_image_landmarks_dict[current_id] = landmarks_bottom_list
    return full_image_landmarks_dict


def get_landmarks_in_one_image(landmarks_top, landmarks_bottom, landmarks_nan):
    landmarks_full = []
    for j in range(len(landmarks_top)):
        if list(landmarks_top[j].coords) != landmarks_nan and list(landmarks_bottom[j].coords) == landmarks_nan:
            landmarks_full.append(landmarks_top[j])
        elif list(landmarks_top[j].coords) == landmarks_nan and list(landmarks_bottom[j].coords) != landmarks_nan:
            landmarks_full.append(landmarks_bottom[j])
        elif list(landmarks_top[j].coords) != landmarks_nan and list(landmarks_bottom[j].coords) != landmarks_nan:
            landmarks_full.append(landmarks_top[j])
        else:
            landmarks_full.append(landmarks_top[j]) 
    return landmarks_full