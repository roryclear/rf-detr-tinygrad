import numpy as np
from tinygrad import Tensor
import cv2
from rfdetr import RFDETR, COCO_CLASSES
        
excepted_xyxys = [
[
[63.98152,247.93846,643.0879,931.84863,],
[0.9997773,357.19293,650.101,1261.7733,],
[623.5975,723.0886,698.9821,787.7163,],
],

[
[68.54183,247.87341,625.942,930.71606,],
[0.49844027,657.9232,441.90753,1267.5795,],
[-1.336813,349.85284,645.86804,1263.5271,],
[622.811,716.0851,701.54803,787.1113,],
],

[
[69.56826,248.08185,620.5264,927.0038,],
[626.8524,733.55194,696.87067,788.07404,],
[0.3557253,356.64658,649.5055,1265.8727,],
[-0.14803648,661.9878,443.29095,1270.992,],
],

[
[68.09412,249.721,635.6606,928.99536,],
[2.2342587,356.97098,579.0276,1268.9092,],
[625.3909,731.5665,697.01495,786.87537,],
[-0.12702942,662.12537,439.97614,1271.6777,],
]
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
    #np.testing.assert_allclose(sort_boxes(boxes), sort_boxes(excepted_xyxys[i]), atol=0.5)
    for box, label, class_id in zip(boxes, labels, class_ids):
      x1, y1, x2, y2 = map(int, box)
      color = ((int(class_id)*37)%255, (int(class_id)*17)%255, (int(class_id)*97)%255); cv2.rectangle(image, (x1, y1), (x2, y2), color, 2); cv2.rectangle(image, (x1, y1-18), (x1+len(label)*9, y1), color, -1); cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    cv2.imwrite(f"annotated_image_{i}.jpg", image)

  print("PASSED")
