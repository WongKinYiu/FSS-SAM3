import os
import json
import random
from collections import defaultdict

from PIL import Image

from pycocotools.coco import COCO
from torch.utils.data import Dataset


class FSODCOCODataset(Dataset):
    """
    COCO Few-Shot Dataset

    Supports:
    1. Runtime support sampling
    2. Fixed support protocol
       (few_shot_10shot_seed33.json)

    Query:
        val2017

    Support:
        train2017
    """

    FSOD_COCO_CATEGORIES = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "boat",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "bottle",
        "chair",
        "couch",
        "potted plant",
        "dining table",
        "tv",
    ]

    def __init__(
        self,
        ann_root,
        img_root,
        shot=1,
        seed=33,
        min_box_area=32 * 32,

        # -----------------------------------------
        # fixed support protocol
        # -----------------------------------------
        fixed_support_json=None,
        fixed_support_shot=1,
    ):

        super().__init__()

        self.ann_root = ann_root
        self.img_root = img_root

        self.shot = shot
        self.seed = seed

        self.min_box_area = min_box_area

        self.fixed_support_json = fixed_support_json
        self.fixed_support_shot = fixed_support_shot

        random.seed(seed)

        # --------------------------------------------------
        # COCO paths
        # --------------------------------------------------

        self.train_img_dir = os.path.join(
            img_root,
            "train2017"
        )

        self.val_img_dir = os.path.join(
            img_root,
            "val2017"
        )

        self.train_ann_path = os.path.join(
            ann_root,
            "instances_train2017.json",
        )

        self.val_ann_path = os.path.join(
            ann_root,
            "instances_val2017.json",
        )

        # --------------------------------------------------
        # Load COCO
        # --------------------------------------------------

        print("Loading COCO train annotations...")

        self.coco_train = COCO(self.train_ann_path)

        print("Loading COCO val annotations...")

        self.coco_val = COCO(self.val_ann_path)

        # --------------------------------------------------
        # Category filtering
        # --------------------------------------------------

        self.cat_ids = self.coco_val.getCatIds(
            catNms=self.FSOD_COCO_CATEGORIES
        )

        self.cat_id_to_name = {
            cat["id"]: cat["name"]
            for cat in self.coco_val.loadCats(self.cat_ids)
        }

        self.cat_name_to_id = {
            v: k for k, v in self.cat_id_to_name.items()
        }

        print("\nFSOD COCO Categories:")

        for cid, name in self.cat_id_to_name.items():

            print(f"{cid}: {name}")

        # --------------------------------------------------
        # Build support pool
        # --------------------------------------------------

        print("\nBuilding support pool...")

        self.class_to_support_anns = defaultdict(list)

        train_ann_ids = self.coco_train.getAnnIds(
            catIds=self.cat_ids
        )

        for ann_id in train_ann_ids:

            ann = self.coco_train.loadAnns([ann_id])[0]

            if ann.get("iscrowd", 0):
                continue

            x, y, w, h = ann["bbox"]

            if w * h < self.min_box_area:
                continue

            img_info = self.coco_train.loadImgs(
                [ann["image_id"]]
            )[0]

            img_path = os.path.join(
                self.train_img_dir,
                img_info["file_name"],
            )

            if not os.path.exists(img_path):
                continue

            self.class_to_support_anns[
                ann["category_id"]
            ].append(ann)

        print("Support pool built.")

        for cid in self.cat_ids:

            print(
                f"{self.cat_id_to_name[cid]}: "
                f"{len(self.class_to_support_anns[cid])} "
                f"support instances"
            )

        # --------------------------------------------------
        # Build query set
        # --------------------------------------------------

        print("\nBuilding query set...")

        self.query_instances = []

        val_img_ids = self.coco_val.getImgIds()

        for img_id in val_img_ids:

            ann_ids = self.coco_val.getAnnIds(
                imgIds=[img_id],
                catIds=self.cat_ids,
                iscrowd=False,
            )

            anns = self.coco_val.loadAnns(ann_ids)

            valid_anns = []

            for ann in anns:

                x, y, w, h = ann["bbox"]

                if w * h < self.min_box_area:
                    continue

                valid_anns.append(ann)

            if len(valid_anns) == 0:
                continue

            self.query_instances.append(
                {
                    "image_id": img_id,
                    "annotations": valid_anns,
                }
            )

        print(
            f"Total query images: "
            f"{len(self.query_instances)}"
        )

        # --------------------------------------------------
        # Fixed support protocol
        # --------------------------------------------------

        self.fixed_support_data = None

        if self.fixed_support_json is not None:

            print("\nLoading fixed support json...")

            with open(self.fixed_support_json, "r") as f:

                self.fixed_support_data = json.load(f)

            print("Fixed support loaded.")

            print(
                f"Fixed support shot: "
                f"{self.fixed_support_shot}"
            )

    # ======================================================
    # basic utils
    # ======================================================

    def __len__(self):

        return len(self.query_instances)

    def load_image(self, path):

        image = Image.open(path).convert("RGB")

        return image

    # ======================================================
    # support sampling
    # ======================================================

    def sample_support(
        self,
        category_id,
        query_image_id,
    ):

        # --------------------------------------------------
        # Fixed support protocol
        # --------------------------------------------------

        if self.fixed_support_data is not None:

            class_name = self.cat_id_to_name[
                category_id
            ]

            fixed_supports = self.fixed_support_data[
                class_name
            ]

            selected_supports = fixed_supports[
                :self.fixed_support_shot
            ]

            support_images = []
            support_boxes = []
            support_labels = []

            for item in selected_supports:

                rel_path = item["image"]

                # remove coco/
                rel_path = rel_path.replace(
                    "coco/",
                    ""
                )

                img_path = os.path.join(
                    self.img_root,
                    rel_path
                )

                image = self.load_image(img_path)

                support_images.append(image)

                support_boxes.append(
                    item["bbox"]
                )

                support_labels.append(
                    category_id
                )

            return (
                support_images,
                support_boxes,
                support_labels,
            )

        # --------------------------------------------------
        # Runtime random sampling
        # --------------------------------------------------

        candidates = self.class_to_support_anns[
            category_id
        ]

        valid_candidates = []

        for ann in candidates:

            # avoid same image leakage
            if ann["image_id"] == query_image_id:
                continue

            valid_candidates.append(ann)

        if len(valid_candidates) < self.shot:

            raise ValueError(
                f"Not enough support samples for class "
                f"{category_id}"
            )

        sampled_anns = random.sample(
            valid_candidates,
            self.shot,
        )

        support_images = []
        support_boxes = []
        support_labels = []

        for ann in sampled_anns:

            img_info = self.coco_train.loadImgs(
                [ann["image_id"]]
            )[0]

            img_path = os.path.join(
                self.train_img_dir,
                img_info["file_name"],
            )

            image = self.load_image(img_path)

            support_images.append(image)

            support_boxes.append(
                ann["bbox"]
            )

            support_labels.append(
                category_id
            )

        return (
            support_images,
            support_boxes,
            support_labels,
        )

    # ======================================================
    # get item
    # ======================================================

    def __getitem__(self, idx):

        query_data = self.query_instances[idx]

        query_image_id = query_data["image_id"]

        query_img_info = self.coco_val.loadImgs(
            [query_image_id]
        )[0]

        query_img_path = os.path.join(
            self.val_img_dir,
            query_img_info["file_name"],
        )

        query_image = self.load_image(
            query_img_path
        )

        query_boxes = []
        query_labels = []

        # --------------------------------------------------
        # query annotations
        # --------------------------------------------------

        for ann in query_data["annotations"]:

            query_boxes.append(
                ann["bbox"]
            )

            query_labels.append(
                ann["category_id"]
            )

        # --------------------------------------------------
        # randomly select target class
        # --------------------------------------------------

        target_class = random.choice(
            query_labels
        )

        # --------------------------------------------------
        # sample support
        # --------------------------------------------------

        (
            support_images,
            support_boxes,
            support_labels,
        ) = self.sample_support(
            target_class,
            query_image_id,
        )

        return {

            # ---------------------------------------------
            # query
            # ---------------------------------------------

            "query_image": query_image,

            "query_boxes": query_boxes,

            "query_labels": query_labels,

            # ---------------------------------------------
            # support
            # ---------------------------------------------

            "support_images": support_images,

            "support_boxes": support_boxes,

            "support_labels": support_labels,

            # ---------------------------------------------
            # meta
            # ---------------------------------------------

            "target_class": target_class,

            "query_image_id": query_image_id,

            "query_image_path": query_img_path,
        }


# =========================================================
# debug
# =========================================================

if __name__ == "__main__":

    dataset = FSODCOCODataset(

        ann_root=
        "/data/tsai091/FSOD-VFM/data/coco/annotations",

        img_root=
        "/data/tsai091/datasets/coco",

        shot=1,

        fixed_support_json=
        "/data/tsai091/FSOD-VFM/data/coco/"
        "few_shot_10shot_seed33.json",

        fixed_support_shot=1,
    )

    print("\nDataset size:", len(dataset))

    sample = dataset[0]

    print("\nReturned keys:")
    print(sample.keys())

    print("\nTarget class:")
    print(sample["target_class"])

    print("\nSupport box:")
    print(sample["support_boxes"][0])