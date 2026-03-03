import re

# Define placeholders for dataset paths.

CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

ACETONE = {
    "annotation_path": "outputs/workspace/acetone_dataset/annotations/c1024-augX.jsonl",
    "data_path": "outputs/workspace/acetone_dataset/images/c1024-augX",
}

ACETONE_debug = {
    "annotation_path": "outputs/workspace/acetone_dataset/annotations/debug.jsonl",
    "data_path": "/mnt/bn/acetone-v0/datasets/PHOTOS/raw/adobe-5k-extract",
    "lut_folder": "/mnt/bn/acetone-v0/datasets/LUTS-final/cube-3d-npy-filtered",
}

ACETONE_ONLINE = {
    "annotation_path": "outputs/workspace/acetone_dataset/annotations/online-c1024.jsonl",
    "data_path": "/mnt/bn/acetone-v0/datasets/PHOTOS/raw/adobe-5k-extract",
    "lut_folder": "/mnt/bn/acetone-v0/datasets/LUTS-final/cube-npy-c1024",
}

ACETONE_PRETRAIN = {
    "image_folder": "/mnt/bn/acetone-v0/datasets/PHOTOS/raw/mscoco-extract/train",
    "lut_folder": "/mnt/bn/acetone-v0/datasets/LUTS-final/cube-3d-npy-filtered",
}

# stage 1: /mnt/bn/acetone-v0/datasets/LUTS-final/cube-npy-c2048-noBW-4plus
# stage 2: /mnt/bn/acetone-v0/datasets/LUTS-final/cube-3d-npy-filtered

ACETONE_INSTRUCT_ONLINE_ADOBE = {
    "annotation_path": "outputs/workspace/acetone_dataset/instructions/adobe-5k/adobe_instruct_standard_train.jsonl",
    "data_path": "outputs/workspace/acetone_dataset/acetone-meta-adobe/train/raw_image",
    "lut_folder": "outputs/workspace/acetone_dataset/acetone-meta-adobe/train/lut",
}

ACETONE_INSTRUCT_ONLINE_PPR = {
    "annotation_path": "outputs/workspace/acetone_dataset/instructions/ppr-10k/ppr_instruct_standard_train.jsonl",
    "data_path": "outputs/workspace/acetone_dataset/acetone-meta-ppr/train/raw_image",
    "lut_folder": "outputs/workspace/acetone_dataset/acetone-meta-ppr/train/lut",
}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    "acetone": ACETONE,
    "acetone_debug": ACETONE_debug,
    "acetone_online": ACETONE_ONLINE,
    "acetone_pretrain": ACETONE_PRETRAIN,
    "acetone_instruct_online_adobe": ACETONE_INSTRUCT_ONLINE_ADOBE,
    "acetone_instruct_online_ppr": ACETONE_INSTRUCT_ONLINE_PPR,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
