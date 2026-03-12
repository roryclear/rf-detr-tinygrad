from rfdetr import RFDETR, COCO_CLASSES
from tinygrad import Tensor
import cv2
from tinygrad import Tensor, dtypes
import numpy as np
from pathlib import Path

def draw_bounding_boxes(orig_img_path, predictions, class_labels):
  color_dict = {label: tuple((((i+1) * 50) % 256, ((i+1) * 100) % 256, ((i+1) * 150) % 256)) for i, label in enumerate(class_labels)}
  font = cv2.FONT_HERSHEY_SIMPLEX

  def is_bright_color(color):
    r, g, b = color
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return brightness > 127

  orig_img = (cv2.imread(orig_img_path) if not isinstance(orig_img_path, np.ndarray) else cv2.imdecode(orig_img_path, 1))
  height, width, _ = orig_img.shape
  box_thickness = int((height + width) / 400)
  font_scale = (height + width) / 2500
  
  for pred in predictions:
    x1, y1, x2, y2, conf, class_id = pred
    if conf == 0: continue

    x1, y1, x2, y2, class_id = map(int, (x1, y1, x2, y2, class_id))
    color = color_dict[class_labels[class_id]]

    cv2.rectangle(orig_img, (x1, y1), (x2, y2), color, box_thickness)

    label = f"{class_labels[class_id]} {conf:.2f}"
    text_size, _ = cv2.getTextSize(label, font, font_scale, 1)
    label_y, bg_y = ((y1 - 4, y1 - text_size[1] - 4) if y1 - text_size[1] - 4 > 0 else (y1 + text_size[1], y1))

    cv2.rectangle(orig_img,
        (x1, bg_y),
        (x1 + text_size[0], bg_y + text_size[1]),
        color,
        -1,
    )

    font_color = (0, 0, 0) if is_bright_color(color) else (255, 255, 255)
    cv2.putText(
        orig_img,
        label,
        (x1, label_y),
        font,
        font_scale,
        font_color,
        1,
        cv2.LINE_AA,
    )
  return orig_img

if __name__ == "__main__":
  sizes = ["nano", "small", "medium", "large"]
  im0 = cv2.imread("dog.jpg")

  for i in range(len(sizes)):
    model = RFDETR(sizes[i])
    im = Tensor(im0).cast(dtypes.float32)
    im = model.preprocess(im)
    pred = model(im).numpy()
    pred = pred[pred[:, 4] >= 0.5]
    pred = model.scale_boxes(im.shape[:2], pred, im0.shape)
    preds = []
    for x in pred:
        x1, y1, x2, y2, score, class_id = x
        preds.append(np.array([x1, y1, x2, y2, score, class_id]))
    _, buffer = cv2.imencode(".jpg", im0)
    output = draw_bounding_boxes(buffer, preds, COCO_CLASSES)
    Path("./outputs").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(f"outputs/output_{i}.jpg", output)

    print(f"Saved result to outputs/output_{i}.jpg")
