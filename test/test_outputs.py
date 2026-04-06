import numpy as np
from tinygrad import Tensor
import cv2
from rfdetr import RFDETR, COCO_CLASSES
        
excepted_xyxys = [
[[56.71238,248.32944,639.98444,930.5121,],[0.0,666.39166,466.64917,1272.4757,],[631.45856,723.1369,698.7034,787.298,],[0.0,353.19443,623.5554,1264.9243,], ],
[[61.509323,245.74634,640.7075,930.5193,],[0.0,661.4972,444.99396,1267.2108,], ],
[[68.49747,247.551,640.4772,927.1219,],[0.0,664.07965,457.10947,1269.8434,],[627.56744,733.1611,698.44946,787.8765,],[0.0,356.36108,615.18146,1265.6769,], ],
[[69.23218,249.93488,634.46216,929.3088,],[0.0,661.8646,444.80447,1271.1677,],[628.58026,733.76306,698.1341,787.7615,], ],
]

def sort_boxes(xyxy):
  xyxy = np.asarray(xyxy)
  order = np.lexsort((xyxy[:,3], xyxy[:,2], xyxy[:,1], xyxy[:,0]))
  return xyxy[order]

if __name__ == "__main__":
  threshold = 0.5
  models = [[384, "nano"], [512, "small"], [576, "medium"], [704, "large"]]

  for i in range(len(models)):
    image = cv2.imread("test/dog.jpg")
    model = RFDETR(models[i][1])
    img_np = np.asarray(image)
    h, w = img_np.shape[:2]
    img = Tensor(img_np)
    output = model(img)
    output = output.numpy()
    boxes = output[:, :4]
    scores = output[:, 4]
    class_ids = output[:, 5].astype(int)
    keep = scores > threshold
    scores = scores[keep]
    class_ids = class_ids[keep]
    boxes = boxes[keep]
    labels = [f"{COCO_CLASSES[class_id]}" for class_id in class_ids]
    np.testing.assert_allclose(sort_boxes(boxes), sort_boxes(excepted_xyxys[i]), atol=0.5)
    for box, label, class_id in zip(boxes, labels, class_ids):
      x1, y1, x2, y2 = map(int, box)
      color = ((int(class_id)*37)%255, (int(class_id)*17)%255, (int(class_id)*97)%255); cv2.rectangle(image, (x1, y1), (x2, y2), color, 2); cv2.rectangle(image, (x1, y1-18), (x1+len(label)*9, y1), color, -1); cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    cv2.imwrite(f"annotated_image_{i}.jpg", image)

  print("PASSED")
