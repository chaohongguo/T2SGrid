DATASETS = {
    "charades": {
        "video_path": "/your_data_dir/charades/videos",
        "splits": {
            "test": {
                "annotation_file": "data/annotations/charades/charadea_sta_seq_test.json",
                "video_image_dir": "data/TGrid_data/charades/test_1334",
                "result_dir": "results/charades",
            },
            "train": {
                "annotation_file": "data/annotations/charades/train.json",
                "video_image_dir": "data/TGrid_data/charades/train_5119",
                "result_dir": "results",
            }
        },
    },
    "anet": {
        "video_path": "/your_data_dir/anet/videos",
        "splits": {
            "train": {
                "annotation_file": "data/annotations/anet/train.json",
                "video_image_dir": "data/TGrid_data/anet/train_",
                "result_dir": "results",
            },
            "test": {
                "annotation_file": "data/annotations/anet/anet_val_2_readable.json",
                "video_image_dir": "data/TGrid_data/anet/test_",
                "result_dir": "results/anet",
            }
        },
    },
}