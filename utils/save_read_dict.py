import pickle
import os
import utils


def save_dict(dict_name, file_path):
    with open(os.path.join(file_path , "cropped_name_dict.pkl"), "wb") as pkl:
        pickle.dump(dict_name,pkl)


def get_pkl_file_name(test_images_oriented_folder):
    for root, dirs, files in os.walk(test_images_oriented_folder):
        for file in files:
            if file.endswith(".pkl"):
                pkl_name = file
                return pkl_name


def read_dict(file_path):
    pkl_name = get_pkl_file_name(file_path)
    with open(os.path.join(file_path, pkl_name), "rb") as pkl:
        new_dict = pickle.load(pkl)
    return new_dict

def load_test_nii_txt(id_list_file_name):
    ext = os.path.splitext(id_list_file_name)[1]
    if ext in ['.csv', '.txt']:
        id_list = utils.io.text.load_list_csv(id_list_file_name)

    # self.id_list=file_final_list
    new_id_list = [x for x in id_list if x != []]
    new_id_list = [col for row in new_id_list for col in row]
    return new_id_list

if __name__ == "__main__":
    list1 = [["q"], ["t"], ["v"]]
    print([col for row in list1 for col in row])
