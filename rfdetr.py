import numpy as np
import math
from tinygrad.dtype import dtypes
from tinygrad.nn.state import get_state_dict, load_state_dict, safe_save, safe_load
from tinygrad.helpers import fetch
from tinygrad import Tensor, nn

COCO_CLASSES = ["","person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","","backpack","umbrella","","","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","","dining table","","","toilet","","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","","book","clock","vase","scissors","teddy bear","hair drier"]
detr_to_yolo = [80, 0, 1, 2, -1, -1, 5, 6, 7, 8, 9, 10, 80, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 80, 24, 25, 80, 80, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 80, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, -1, -1, 59, 80, -1, 80, 80, 61, 80, -1, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 80, 73, 74, 75, 76, 77, 78]

class WindowedDinov2WithRegistersEmbeddings():
  def __call__(self, pixel_values): pass

class Dinov2WithRegistersSdpaSelfAttention():
    def __call__(self, hidden_states, head_mask, output_attentions): pass

class Dinov2WithRegistersSdpaAttention():
    def __call__(self, hidden_states, head_mask=None, output_attentions= False): pass
class Dinov2WithRegistersMLP():
    def __call__(self, hidden_state): pass

class WindowedDinov2WithRegistersLayer():
    def __call__(self, hidden_states, head_mask=None, output_attentions= False, run_full_attention= False): pass

class WindowedDinov2WithRegistersEncoder():
    def __call__(self): pass
class WindowedDinov2WithRegistersBackbone():
    def __init__(self):
      self.config = {}
      self.stage_names = ['stem', 'stage1', 'stage2', 'stage3', 'stage4', 'stage5', 'stage6', 'stage7', 'stage8', 'stage9', 'stage10', 'stage11', 'stage12']
      self.out_features = ['stage3', 'stage6', 'stage9', 'stage12']

    def __call__(self, pixel_values, output_hidden_states=None, output_attentions=None, return_dict=None,): pass

class TransformerDecoderLayer(): # todo, remove unused
    def __call__(self, tgt, memory, memory_key_padding_mask, query_pos,
      reference_points=None, spatial_shapes=None, level_start_index=None): pass
    
class TransformerDecoder(): # todo remove unused
    def __call__(self, tgt, memory, memory_key_padding_mask=None,
      refpoints_unsigmoid=None, level_start_index=None, spatial_shapes=None): pass



class MSDeformAttn():
    def __call__(self, query, reference_points, input_flatten, input_spatial_shapes, input_padding_mask=None): pass

class Transformer():
    def __call__(self, srcs, masks, refpoint_embed, query_feat): pass

    
class ConvX():
    def __call__(self, x): pass

class Bottleneck():
    def __call__(self, x): pass

class C2f():
    def __init__(self): self.c = 128
    def __call__(self, x): pass

class LayerNorm():
    def __init__(self): self.eps = 1e-6
    def __call__(self, x): pass

class MultiScaleProjector():
    def __call__(self, x): pass

class PositionEmbeddingSine():
    def __init__(self):
      self.scale = 6.283185307179586
      self.num_pos_feats = 128
      self.temperature = 10000

    def __call__(self, tensors, mask, align_dim_orders = True): pass

class Backbone():
    def __call__(self, tensors ,mask): pass

class MLP():
    def __call__(self, x): pass
    
class RFDETR():
  def __init__(self, name, res=None):
    self.dense = nn.Linear(384, 384)

  def __call__(self):
    x = Tensor.rand((1, 580, 384))
    return self.dense(x)

class seq:
  def __init__(self, size=0): self.size = size
  def __setitem__(self, key, value): setattr(self, str(key), value)
  def __getitem__(self, idx):
    try:
      return getattr(self, str(idx))
    except AttributeError:
      raise IndexError(idx)
  def __len__(self): return self.size
  def __call__(self, x): pass
  
def resize(img, new_size):
  img = img.permute(2,0,1)
  img = Tensor.interpolate(img, size=(new_size[1], new_size[0]), mode='linear', align_corners=False)
  img = img.permute(1, 2, 0)
  return img

if __name__ == '__main__':
  threshold = 0.25
  import sys
  import cv2
  if len(sys.argv) < 2:
    print("Error: Image URL or path not provided.")
    sys.exit(1)
  sizes = {'n': 'nano', 's': 'small', 'm': 'medium', 'l': 'large'}
  img_path = sys.argv[1]
  size = sys.argv[2] if len(sys.argv) >= 3 else (print("No variant given, so choosing 'n' as the default. RFDETR has different variants, you can choose from ['n', 's', 'm', 'l']") or 'n')
  size = sizes[size]
  print(f'running inference for rfdetr version {size}')

  model = RFDETR(size)

  image_location = np.frombuffer(fetch(img_path).read_bytes(), np.uint8)
  image = cv2.imdecode(image_location, 1)
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

  scale = min(h, w) / 640
  th = max(1, int(2*scale))
  fs = 0.5*scale
  ft = max(1, int(scale))
  tb = int(18*scale)

  for box, label, class_id in zip(boxes, labels, class_ids):
    x1, y1, x2, y2 = map(int, box)
    color = ((int(class_id)*37)%255, (int(class_id)*17)%255, (int(class_id)*97)%255); cv2.rectangle(image, (x1, y1), (x2, y2), color, th); cv2.rectangle(image, (x1, y1-tb), (x1+len(label)*int(9*scale), y1), color, -1); cv2.putText(image, label, (x1, y1 - int(4*scale)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), ft, cv2.LINE_AA)
  cv2.imwrite(f"annotated_image.jpg", image)
  print("saved result as annotated_image.jpg")

