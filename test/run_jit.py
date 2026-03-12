import numpy as np
from tinygrad import Tensor
import cv2
from rfdetr import RFDETR, COCO_CLASSES
from tinygrad import Tensor, TinyJit
import numpy as np
import sys
import time

@TinyJit
def jit_inf(im, model):
  im = model.preprocess(im)
  return model(im)
if __name__ == "__main__":
  sizes = {'n': 'nano', 's': 'small', 'm': 'medium', 'l': 'large'}
  size = sys.argv[1]
  size = sizes[size]
  image = cv2.imread("test/dog.jpg")
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  model = RFDETR(size)
  img_np = np.asarray(image)
  h, w = img_np.shape[:2]
  img = Tensor(img_np)
  processed_images = model.preprocess(img)
  output = model(processed_images).numpy()
  for _ in range(3): jit_output = jit_inf(img, model).numpy()
  np.testing.assert_allclose(jit_output, output, rtol=1e-7)

  ts = time.time()
  for _ in range(10): _ = jit_inf(img, model).numpy()
  print(f"FPS {size}: {10 / (time.time() - ts):.2f}")
