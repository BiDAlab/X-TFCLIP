import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict


def visualize_masks(mask, title: str):
    plt.figure(figsize=(15, 8))  # Adjust size for better visibility
    sns.heatmap(mask, cmap="viridis", cbar=True)
    plt.xlabel("Gallery Tracklets")
    plt.ylabel("Query Tracklets")
    plt.title("Heatmap of " + title)
    plt.show()


def load_soft_biometrics(attributes_csv: str, case_folder: str, print_=False):
    df = pd.read_csv(attributes_csv)
    attributes_dict = OrderedDict(df.set_index('id').to_dict(orient='index'))
    not_found_array = np.array([2, 8, 4, 3, 4, 6, 5, 2, 2, 3, 4, 12, 9, 6, 8])

    tracklet_list = []  # Store tracklet details
    biometric_values = []  # Store biometric features only

    folder_list = sorted(os.listdir(case_folder))
    print("Folders inside mothafucka: ", folder_list)

    for person_id in folder_list:
        num_tracklet = 0

        person_path = os.path.join(case_folder, person_id)
        if not os.path.isdir(person_path):
            print("Skipping directory for person:", person_id)
            continue  # Skip if not a directory

        for tracklet in os.listdir(person_path):
            num_tracklet += 1
            tracklet_path = os.path.join(person_path, tracklet)
            if not os.path.isdir(tracklet_path):
                continue  # Skip if not a valid tracklet folder

            if int(person_id) in attributes_dict:
                gender_value = attributes_dict[int(person_id)].get('gender', 'Unknown')
                tracklet_list.append({
                    'tracklet_name': tracklet,
                    'person_id': int(person_id),
                    'soft_biometrics': attributes_dict[int(person_id)]
                })
                biometric_values.append(list(attributes_dict[int(person_id)].values()))
            else:
                print("Person ID not in attributes dict", int(person_id))
                gender_value = "Unknown"
                biometric_values.append(not_found_array)

            # Print person ID, tracklet name, and gender value
            print(f"Person ID: {person_id}, Tracklet: {tracklet}, Gender: {gender_value}")

        if print_:
            print(f"Loaded {num_tracklet} tracklets for {len(set([t['person_id'] for t in tracklet_list]))} persons.")

    biometric_values = np.array(biometric_values, dtype=np.float32)
    return biometric_values, tracklet_list


def create_mask_matrix_gender2(query_softbio: np.ndarray, gallery_softbio: np.ndarray, unknown_value: int = 2):
    # expand the genders
    query_genders = np.expand_dims(query_softbio[:, 0], axis=1)

    gallery_genders = np.expand_dims(gallery_softbio[:, 0], axis=0)

    valid_mask = (query_genders == gallery_genders) | (query_genders == unknown_value) | (gallery_genders == unknown_value)
    mask_matrix = np.where(valid_mask, 0, 100)

    return mask_matrix


def create_mask_matrix_ages(query_softbio: np.ndarray, gallery_softbio: np.ndarray, unknown_value: int = 8):
    query_ages = np.expand_dims(query_softbio[:, 1], axis=1)
    gallery_ages = np.expand_dims(gallery_softbio[:, 1], axis=0)

    valid_mask = (query_ages == gallery_ages) | (query_ages == unknown_value) | (gallery_ages == unknown_value)
    mask_matrix = np.where(valid_mask, 0, 100)

    return mask_matrix


def create_mask_matrix_height(query_softbio: np.ndarray, gallery_softbio: np.ndarray, unknown_value: int = 4):
    query_height = np.expand_dims(query_softbio[:, 2], axis=1)
    gallery_height = np.expand_dims(gallery_softbio[:, 2], axis=0)

    valid_mask = (query_height == gallery_height) | (query_height == unknown_value) | (gallery_height == unknown_value)
    mask_matrix = np.where(valid_mask, 0, 100)

    return mask_matrix


def create_mask_matrix_ethnicity(query_softbio: np.ndarray, gallery_softbio: np.ndarray, unknown_value: int = 4):
    query_ethnicity = np.expand_dims(query_softbio[:, 4], axis=1)
    gallery_ethnicity = np.expand_dims(gallery_softbio[:, 4], axis=0)

    valid_mask = (query_ethnicity == gallery_ethnicity) | (query_ethnicity == unknown_value) | (gallery_ethnicity == unknown_value)
    mask_matrix = np.where(valid_mask, 0, 100)

    return mask_matrix


def create_mask_matrix_body_volume(query_softbio: np.ndarray, gallery_softbio: np.ndarray, unknown_value: int = 3):
    query_body = np.expand_dims(query_softbio[:, 3], axis=1)
    gallery_body = np.expand_dims(gallery_softbio[:, 3], axis=0)

    valid_mask = (query_body == gallery_body) | (query_body == unknown_value) | (gallery_body == unknown_value)
    mask_matrix = np.where(valid_mask, 0, 100)

    return mask_matrix


def create_mask_matrix_hair_color(query_softbio: np.ndarray, gallery_softbio: np.ndarray, unknown_value: int = 6):
    query_hair_color = np.expand_dims(query_softbio[:, 5], axis=1)
    gallery_hair_color = np.expand_dims(gallery_softbio[:, 5], axis=0)

    valid_mask = (query_hair_color == gallery_hair_color) | (query_hair_color == unknown_value) | (gallery_hair_color == unknown_value)
    mask_matrix = np.where(valid_mask, 0, 100)

    return mask_matrix


def create_mask_matrix_moustache(query_softbio: np.ndarray, gallery_softbio: np.ndarray, unknown_value: int = 2):
    query_moustache = np.expand_dims(query_softbio[:, 8], axis=1)
    gallery_moustache = np.expand_dims(gallery_softbio[:, 8], axis=0)

    valid_mask = (query_moustache == gallery_moustache) | (query_moustache == unknown_value) | (gallery_moustache == unknown_value)
    mask_matrix = np.where(valid_mask, 0, 100)

    return mask_matrix


if __name__ == "__main__":  # case1_aerial_to_ground
    gallery_m, gallery_dict = load_soft_biometrics(attributes_csv="AG-VPReID_dataset/attributes/case1_aerial_to_ground/gallery.csv",
                                     case_folder="AG-VPReID_dataset/case1_aerial_to_ground/gallery",
                                     print_=True)

    query_m, query_dict = load_soft_biometrics(attributes_csv="AG-VPReID_dataset/attributes/case1_aerial_to_ground/query.csv",
                                   case_folder="AG-VPReID_dataset/case1_aerial_to_ground/query",
                                   print_=True)

    gender_mask = create_mask_matrix_gender2(query_softbio=query_m, gallery_softbio=gallery_m)
    age_mask = create_mask_matrix_ages(query_softbio=query_m, gallery_softbio=gallery_m)
    height_mask = create_mask_matrix_height(query_softbio=query_m, gallery_softbio=gallery_m)
    ethnicity_mask = create_mask_matrix_ethnicity(query_softbio=query_m, gallery_softbio=gallery_m)
    body_mask = create_mask_matrix_body_volume(query_softbio=query_m, gallery_softbio=gallery_m)
    hair_mask = create_mask_matrix_hair_color(query_softbio=query_m, gallery_softbio=gallery_m)
    moustache_mask = create_mask_matrix_moustache(query_softbio=query_m, gallery_softbio=gallery_m)

    # visualize them
    # visualize_masks(gender_mask, title="Gender")
    # visualize_masks(age_mask, title="Age")
    # visualize_masks(height_mask, title="Height")
    # visualize_masks(ethnicity_mask, title="Ethnicity")
    # visualize_masks(body_mask, title="Body volume")
    # visualize_masks(hair_mask, title="Hair color")
    visualize_masks(moustache_mask, title="Moustache")
    visualize_masks(gender_mask+age_mask+ethnicity_mask+body_mask+moustache_mask, title="Gender + Age + Ethnicity + Body volume + Hair color")
