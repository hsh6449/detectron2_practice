#%%
from pyexpat import model
from unicodedata import category
import torch, torchvision

print(torch.__version__, torch.cuda.is_available())
# %%
!git clone --branch v0.6 https://github.com/facebookresearch/detectron2.git detectron2_repo
# %%
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
coco_metadata = MetadataCatalog.get("coco_2017_val")

from detectron2.projects import point_rend
# %%
plt.figure(figsize=(20,20))
im = cv2.imread("./imdata/000000005477.jpg")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
plt.imshow(im)
print(im.shape)
# %% rcnn model prediction

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
mask_rcnn_predictor = DefaultPredictor(cfg)
mask_rcnn_outputs = mask_rcnn_predictor(im)

# %% add pointrend model

cfg = get_cfg()
point_rend.add_pointrend_config(cfg)

cfg.merge_from_file("detectron2_repo/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl"
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

# %% compare
plt.figure(figsize=(20,20))

v = Visualizer(im[:,:,::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
mask_rcnn_result = v.draw_instance_predictions(mask_rcnn_outputs["instances"].to("cpu")).get_image()

v = Visualizer(im[:,:,::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
point_rend_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()

print("Mask R-CNN with point Rend (top) vs Default Mask R- CNN (bottom)")
plt.imshow(np.concatenate((point_rend_result, mask_rcnn_result), axis=0)[:,:,::-1], )
# %%
def plot_mask(mask, title="", point_coords=None, figsize=10, point_marker_size=5):
  '''
  Simple plotting tool to show intermediate mask predictions and points 
  where PointRend is applied.
  
  Args:
    mask (Tensor): mask prediction of shape HxW
    title (str): title for the plot
    point_coords ((Tensor, Tensor)): x and y point coordinates
    figsize (int): size of the figure to plot
    point_marker_size (int): marker size for points
  '''

  H, W = mask.shape
  plt.figure(figsize=(figsize, figsize))
  if title:
    title += ", "
  plt.title("{}resolution {}x{}".format(title, H, W), fontsize=30)
  plt.ylabel(H, fontsize=30)
  plt.xlabel(W, fontsize=30)
  plt.xticks([], [])
  plt.yticks([], [])
  plt.imshow(mask, interpolation="nearest", cmap=plt.get_cmap('gray'))
  if point_coords is not None:
    plt.scatter(x=point_coords[0], y=point_coords[1], color="red", s=point_marker_size, clip_on=True) 
  plt.xlim(-0.5, W - 0.5)
  plt.ylim(H - 0.5, - 0.5)
  plt.show()
# %%
from detectron2.data import transforms as T
plt.figure(figsize=(20,20))
plt.imshow(T.ResizeShortestEdge(800, 1333).get_transform(im).apply_image(im))
print(T.ResizeShortestEdge(800, 1333).get_transform(im).apply_image(im).shape)
#%%
from detectron2.data import transforms as T
model = predictor.model
instance_idx = 0
category_idx = 4

with torch.no_grad():
    height, width = im.shape[:2]
    im_transformed = T.ResizeShortestEdge(800, 1333).get_transform(im).apply_image(im) # augmentation
    batched_inputs = [{"image": torch.as_tensor(im_transformed).permute(2,0,1)}]

    detected_instances = [x["instances"] for x in model.inference(batched_inputs)]
    [r.remove("pred_masks") for r in detected_instances]
    pred_boxes = [x.pred_boxes for x in detected_instances]

    images = model.preprocess_image(batched_inputs)
    features = model.backbone(images.tensor)

    mask_coarse_logits = model.roi_heads.mask_head.coarse_head(model.roi_heads.mask_head._roi_pooler(features, pred_boxes))

    plot_mask(
        mask_coarse_logits[instance_idx, category_idx].to("cpu"),
        title="Coarse prediction"
    )
# %%
mask_features_list = [
    features[k] for k in model.roi_heads.mask_head.mask_point_in_features
]
features_scales = [
    model.roi_heads.mask_head._feature_scales[k]
    for k in model.roi_heads.mask_head.mask_point_in_features
    ]

# %%
from detectron2.projects.point_rend.mask_head import calculate_uncertainty
from detectron2.projects.point_rend.point_features import get_uncertain_point_coords_with_randomness

# Change number of points to select
num_points = 14 * 14
# Change randomness parameters 
oversample_ratio = 3  # `k` in the paper
importance_sample_ratio = 0.75  # `\beta` in the paper

with torch.no_grad():
  # We take predicted classes, whereas during real training ground truth classes are used.
  pred_classes = torch.cat([x.pred_classes for x in detected_instances])

  # Select points given a corse prediction mask
  point_coords = get_uncertain_point_coords_with_randomness(
    mask_coarse_logits,
    lambda logits: calculate_uncertainty(logits, pred_classes),
    num_points=num_points,
    oversample_ratio=oversample_ratio,
    importance_sample_ratio=importance_sample_ratio
  )

  H, W = mask_coarse_logits.shape[-2:]
  plot_mask(
    mask_coarse_logits[instance_idx, category_idx].to("cpu"),
    title="Sampled points over the coarse prediction",
    point_coords=(
      W * point_coords[instance_idx, :, 0].to("cpu") - 0.5,
      H * point_coords[instance_idx, :, 1].to("cpu") - 0.5
    ),
    point_marker_size=50
  )
# %%
from detectron2.layers import interpolate
from detectron2.projects.point_rend.mask_head import calculate_uncertainty
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_on_grid,
    point_sample,
    point_sample_fine_grained_features,
)

num_subdivision_steps = 5
num_subdivision_points = 28 * 28


with torch.no_grad():
  plot_mask(
      mask_coarse_logits[0, category_idx].to("cpu").numpy(), 
      title="Coarse prediction"
  )

  mask_logits = mask_coarse_logits
  for subdivions_step in range(num_subdivision_steps):
    # Upsample mask prediction
    mask_logits = interpolate(
        mask_logits, scale_factor=2, mode="bilinear", align_corners=False
    )
    # If `num_subdivision_points` is larger or equalt to the
    # resolution of the next step, then we can skip this step
    H, W = mask_logits.shape[-2:]
    if (
      num_subdivision_points >= 4 * H * W
      and subdivions_step < num_subdivision_steps - 1
    ):
      continue
    # Calculate uncertainty for all points on the upsampled regular grid
    uncertainty_map = calculate_uncertainty(mask_logits, pred_classes)
    # Select most `num_subdivision_points` uncertain points
    point_indices, point_coords = get_uncertain_point_coords_on_grid(
        uncertainty_map, 
        num_subdivision_points
    )

    # Extract fine-grained and coarse features for the points
    fine_grained_features, _ = point_sample_fine_grained_features(
      mask_features_list, features_scales, pred_boxes, point_coords
    )
    coarse_features = point_sample(mask_coarse_logits, point_coords, align_corners=False)

    # Run PointRend head for these points
    point_logits = model.roi_heads.mask_head.point_head(fine_grained_features, coarse_features)

    # put mask point predictions to the right places on the upsampled grid.
    R, C, H, W = mask_logits.shape
    x = (point_indices[instance_idx] % W).to("cpu")
    y = (point_indices[instance_idx] // W).to("cpu")
    point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
    mask_logits = (
      mask_logits.reshape(R, C, H * W)
      .scatter_(2, point_indices, point_logits)
      .view(R, C, H, W)
    )
    plot_mask(
      mask_logits[instance_idx, category_idx].to("cpu"), 
      title="Subdivision step: {}".format(subdivions_step + 1),
      point_coords=(x, y)
    )

# %%
from detectron2.modeling import GeneralizedRCNN
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_inference

plt.figure(figsize=(20,20))
results = detected_instances
mask_rcnn_inference(mask_logits, results)
results = GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)[0]

# We can use `Visualizer` to draw the predictions on the image.
v = Visualizer(im_transformed[:, :, ::-1], coco_metadata)
v = v.draw_instance_predictions(results["instances"].to("cpu"))
plt.imshow(v.get_image()[:, :, ::-1])
# %%
