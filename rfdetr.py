import numpy as np
import math
from tinygrad.dtype import dtypes
from tinygrad.nn.state import get_state_dict, load_state_dict, safe_save, safe_load
from tinygrad.helpers import fetch
from tinygrad import Tensor, nn

COCO_CLASSES = ["","person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","","backpack","umbrella","","","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","","dining table","","","toilet","","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","","book","clock","vase","scissors","teddy bear","hair drier"]
detr_to_yolo = [80, 0, 1, 2, -1, -1, 5, 6, 7, 8, 9, 10, 80, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 80, 24, 25, 80, 80, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 80, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, -1, -1, 59, 80, -1, 80, 80, 61, 80, -1, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 80, 73, 74, 75, 76, 77, 78]

class WindowedDinov2WithRegistersEmbeddings():
  def __call__(self, pixel_values):
    batch_size, _, height, width = pixel_values.shape
    embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)

    cls_tokens = self.cls_token.expand(batch_size, -1, -1)
    embeddings = Tensor.cat(cls_tokens, embeddings, dim=1)
    embeddings = embeddings + self.position_embeddings
    num_h_patches = height // 16
    num_w_patches = width // 16
    cls_token_with_pos_embed = embeddings[:, :1]
    pixel_tokens_with_pos_embed = embeddings[:, 1:]
    pixel_tokens_with_pos_embed = pixel_tokens_with_pos_embed.view(batch_size, num_h_patches, num_w_patches, -1)

    num_w_patches_per_window = num_w_patches // self.num_windows
    num_h_patches_per_window = num_h_patches // self.num_windows
    windowed_pixel_tokens = pixel_tokens_with_pos_embed.reshape(batch_size, self.num_windows, num_h_patches_per_window, self.num_windows, num_w_patches_per_window, -1)
    windowed_pixel_tokens = windowed_pixel_tokens.permute(0,1,3,2,4,5)
    windowed_pixel_tokens = windowed_pixel_tokens.reshape(batch_size * self.num_windows**2, num_h_patches_per_window * num_w_patches_per_window, -1)

    windowed_cls_token_with_pos_embed = cls_token_with_pos_embed.repeat(self.num_windows ** 2, 1, 1)
    embeddings = Tensor.cat(windowed_cls_token_with_pos_embed, windowed_pixel_tokens, dim=1)
    return embeddings

class Dinov2WithRegistersSdpaSelfAttention():
    def transpose_for_scores(self, x):
      new_x_shape = x.size()[:-1] + (6, 64)
      x = x.view(new_x_shape)
      return x.permute(0, 2, 1, 3)

    def __call__(self, hidden_states, head_mask, output_attentions):
      query_layer = Tensor.rand((1, 6, 580, 64))
      value_layer = Tensor.rand((1, 6, 580, 64))
      key_layer = Tensor.rand((1, 6, 580, 64))

      attn_scores = Tensor.matmul(query_layer, key_layer.transpose(-2, -1))
      attn_probs = Tensor.softmax(attn_scores, axis=-1)
      context_layer = Tensor.matmul(attn_probs, value_layer)
      context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
      new_context_layer_shape = context_layer.size()[:-2] + (384,)
      context_layer = context_layer.view(new_context_layer_shape)
      return context_layer

class Dinov2WithRegistersSdpaAttention():
    def __call__(self, hidden_states, head_mask=None, output_attentions= False):
      print(type(self.attention))
      x = self.attention(hidden_states, head_mask, output_attentions)
      attention_output = self.dense(x)
      return (attention_output,)

class Dinov2WithRegistersMLP():
    def __call__(self, hidden_state):
        hidden_state = self.fc1(hidden_state)
        hidden_state = hidden_state * 0.5 * (1.0 + Tensor.erf(hidden_state / math.sqrt(2.0)))
        hidden_state = self.fc2(hidden_state)
        return hidden_state

class WindowedDinov2WithRegistersLayer():
    def __call__(self, hidden_states, head_mask=None, output_attentions= False, run_full_attention= False):
      x = Tensor.rand((1, 580, 384))
      self_attention_outputs = self.attention(x, head_mask, output_attentions=output_attentions,)
      attention_output = self_attention_outputs[0]
      attention_output = attention_output.view(4, 145, 384)
      return attention_output

class WindowedDinov2WithRegistersEncoder():
    def __call__(self, hidden_states, head_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True,):
      hidden_states = Tensor.rand((4, 145, 384))
      layer_outputs = self.layer[9](hidden_states, None, output_attentions, True)
      return layer_outputs

class WindowedDinov2WithRegistersBackbone():
    def __init__(self):
      self.config = {}
      self.stage_names = ['stem', 'stage1', 'stage2', 'stage3', 'stage4', 'stage5', 'stage6', 'stage7', 'stage8', 'stage9', 'stage10', 'stage11', 'stage12']
      self.out_features = ['stage3', 'stage6', 'stage9', 'stage12']

    def __call__(self, pixel_values, output_hidden_states=None, output_attentions=None, return_dict=None,):
        embedding_output = self.embeddings(pixel_values)

        embedding_output = Tensor.rand_like(embedding_output)
        outputs = self.encoder(embedding_output, output_hidden_states=True, output_attentions=output_attentions, return_dict=return_dict)
        hidden_state = outputs[:, 1 :]
        hidden_state = hidden_state.reshape(1, 2, 2, 12, 12, 384)

        return hidden_state

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
    def __call__(self, tensors ,mask):
      feats = list(self.encoder(tensors)[0]) # fails here?
      return feats[0], None

class MLP():
    def __call__(self, x): pass
    
class RFDETR():
  def __init__(self, name, res=None):
    self.num_queries = 300
    self.means = Tensor.empty(1,1,3)
    self.stds = Tensor.empty(1,1,3)
    config = {"nano":{"n_layers":2, "size": 577, "res": 384}, "small":{"n_layers":3, "size":1025, "res":512},
                "medium":{"n_layers":4, "size":1297, "res":576}, "large":{"n_layers":4, "size":1937, "res":704}}
    self.res = config[name]["res"] if res is None else res

    num_windows = 2
    self.position_embedding = PositionEmbeddingSine()
    self.query_feat = Tensor.empty((3900, 256))
    self.refpoint_embed = Tensor.empty((3900, 4))
    self.class_embed = nn.Linear(256, 91)
    self.bbox_embed = MLP()
    self.bbox_embed.num_layers = 3
    self.bbox_embed.layers = seq(size=3)
    self.bbox_embed.layers[0] = nn.Linear(256, 256)
    self.bbox_embed.layers[1] = nn.Linear(256, 256)
    self.bbox_embed.layers[2] = nn.Linear(256, 4)

    self.bbox_embed.layers = seq(size=3)
    self.bbox_embed.layers[0] = nn.Linear(256, 256)
    self.bbox_embed.layers[1] = nn.Linear(256, 256)
    self.bbox_embed.layers[2] = nn.Linear(256, 4)


    self.transformer = Transformer()
    self.transformer.enc_output = nn.Linear(256, 256)
    self.transformer.enc_output_norm = nn.LayerNorm(256)
    self.transformer.decoder = TransformerDecoder()
    self.transformer.decoder.norm = nn.LayerNorm(256)
    self.transformer.decoder.layers = seq(config[name]["n_layers"])
    for i in range(config[name]["n_layers"]):
      self.transformer.decoder.layers[i] = TransformerDecoderLayer()
      self.transformer.decoder.layers[i].cross_attn = MSDeformAttn()
      self.transformer.decoder.layers[i].linear1 = nn.Linear(256, 2048)
      self.transformer.decoder.layers[i].linear2 = nn.Linear(2048, 256)

      self.transformer.decoder.layers[i].in_proj_weight = Tensor.empty(768, 256)
      self.transformer.decoder.layers[i].in_proj_bias = Tensor.empty(768)
      self.transformer.decoder.layers[i].out_proj_weight = Tensor.empty(256, 256)
      self.transformer.decoder.layers[i].out_proj_bias = Tensor.empty(256)

      self.transformer.decoder.layers[i].norm1 = nn.LayerNorm(256)
      self.transformer.decoder.layers[i].norm2 = nn.LayerNorm(256)
      self.transformer.decoder.layers[i].norm3 = nn.LayerNorm(256)

      self.transformer.decoder.layers[i].cross_attn.value_proj = nn.Linear(256, 256)
      self.transformer.decoder.layers[i].cross_attn.output_proj = nn.Linear(256, 256)
      self.transformer.decoder.layers[i].cross_attn.sampling_offsets = nn.Linear(256, 64)
      self.transformer.decoder.layers[i].cross_attn.attention_weights = nn.Linear(256, 32)

    self.transformer.enc_out_bbox_embed = seq(13)
    self.transformer.enc_out_class_embed = seq(13)
    for i in range(13):
      self.transformer.enc_out_bbox_embed[i] = MLP()
      self.transformer.enc_out_bbox_embed[i].num_layers = 3
      self.transformer.enc_out_bbox_embed[i].layers = seq(3)
      self.transformer.enc_out_bbox_embed[i].layers[0] = nn.Linear(256, 256)
      self.transformer.enc_out_bbox_embed[i].layers[1] = nn.Linear(256, 256)
      self.transformer.enc_out_bbox_embed[i].layers[2] = nn.Linear(256, 4)
      self.transformer.enc_out_bbox_embed[i].layers = seq(3)
      self.transformer.enc_out_bbox_embed[i].layers[0] = nn.Linear(256, 256)
      self.transformer.enc_out_bbox_embed[i].layers[1] = nn.Linear(256, 256)
      self.transformer.enc_out_bbox_embed[i].layers[2] = nn.Linear(256, 4)

      self.transformer.enc_out_class_embed[i] = nn.Linear(256, 91)         

    self.transformer.decoder.ref_point_head = MLP()
    self.transformer.decoder.ref_point_head.layers = seq(size=2)
    self.transformer.decoder.ref_point_head.num_layers = 2
    self.transformer.decoder.ref_point_head.layers = seq(size=2)
    self.transformer.decoder.ref_point_head.layers[0] = nn.Linear(512, 256)
    self.transformer.decoder.ref_point_head.layers[1] = nn.Linear(256, 256)
    self.transformer.decoder.ref_point_head.layers[0] = nn.Linear(512, 256) # todo remove
    self.transformer.decoder.ref_point_head.layers[1] = nn.Linear(256, 256)

    self.backbone = Backbone()
    self.backbone.projector = MultiScaleProjector()
    self.backbone.projector.stages = seq(1) # todo, this is dumb
    self.backbone.projector.stages[0] = seq(2)
    self.backbone.projector.stages[0][0] = C2f()
    self.backbone.projector.stages[0][1] = LayerNorm()
    self.backbone.projector.stages[0][1].weight = Tensor.empty((256))
    self.backbone.projector.stages[0][1].bias = Tensor.empty((256))
    self.backbone.projector.stages[0][0].cv1 = ConvX()
    self.backbone.projector.stages[0][0].cv2 = ConvX()          
    self.backbone.projector.stages[0][0].cv1.conv = nn.Conv2d(in_channels=1536, out_channels=256, kernel_size=1, bias=False, padding=(0, 0))
    self.backbone.projector.stages[0][0].cv2.conv = nn.Conv2d(in_channels=640, out_channels=256, kernel_size=1, bias=False, padding=(0, 0))

    self.backbone.projector.stages[0][0].cv1.bn = LayerNorm()
    self.backbone.projector.stages[0][0].cv1.bn.weight = Tensor.empty((256))
    self.backbone.projector.stages[0][0].cv1.bn.bias = Tensor.empty((256))
    self.backbone.projector.stages[0][0].cv2.bn = LayerNorm()
    self.backbone.projector.stages[0][0].cv2.bn.weight = Tensor.empty((256))
    self.backbone.projector.stages[0][0].cv2.bn.bias = Tensor.empty((256))
  
    self.backbone.projector.stages[0][0].m = seq(size=3)
    for i in range(3):
      self.backbone.projector.stages[0][0].m[i] = Bottleneck()
      self.backbone.projector.stages[0][0].m[i].cv1 = ConvX()
      self.backbone.projector.stages[0][0].m[i].cv1.conv = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, bias=False, padding=(1, 1))
      self.backbone.projector.stages[0][0].m[i].cv2 = ConvX()
      self.backbone.projector.stages[0][0].m[i].cv2.conv = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, bias=False, padding=(1, 1))            
      self.backbone.projector.stages[0][0].m[i].cv1.bn = LayerNorm()
      self.backbone.projector.stages[0][0].m[i].cv1.bn.weight = Tensor.empty((128))
      self.backbone.projector.stages[0][0].m[i].cv1.bn.bias = Tensor.empty((128))
      self.backbone.projector.stages[0][0].m[i].cv2.bn = LayerNorm()
      self.backbone.projector.stages[0][0].m[i].cv2.bn.weight = Tensor.empty((128))
      self.backbone.projector.stages[0][0].m[i].cv2.bn.bias = Tensor.empty((128))
    

    self.backbone.patch_size = 16 # todo, not in state_dict?
    self.backbone.num_windows = num_windows # todo, not in state_dict?
    self.backbone.encoder = WindowedDinov2WithRegistersBackbone()
    self.backbone.encoder.num_windows = num_windows
    self.backbone.encoder.layernorm = nn.LayerNorm(384)
    self.backbone.encoder.encoder = WindowedDinov2WithRegistersEncoder()
    self.backbone.encoder.embeddings = WindowedDinov2WithRegistersEmbeddings()
    self.backbone.encoder.embeddings.num_windows = num_windows
    self.backbone.encoder.embeddings.projection = nn.Conv2d(in_channels=3, out_channels=384, kernel_size=16, stride=(16, 16))
    self.backbone.encoder.embeddings.cls_token = Tensor.empty((1, 1, 384))
    self.backbone.encoder.embeddings.position_embeddings = Tensor.empty((1, config[name]["size"], 384))
    self.backbone.encoder.encoder.layer = seq(size=13)
    for i in range(12):
      self.backbone.encoder.encoder.layer[i] = WindowedDinov2WithRegistersLayer()
      self.backbone.encoder.encoder.layer[i].num_windows = num_windows
      self.backbone.encoder.encoder.layer[i].norm1 = nn.LayerNorm(384)
      self.backbone.encoder.encoder.layer[i].norm2 = nn.LayerNorm(384)
      self.backbone.encoder.encoder.layer[i].attention = Dinov2WithRegistersSdpaAttention()
      self.backbone.encoder.encoder.layer[i].attention.attention = Dinov2WithRegistersSdpaSelfAttention()
      self.backbone.encoder.encoder.layer[i].attention.attention.query = nn.Linear(384, 384)
      self.backbone.encoder.encoder.layer[i].attention.attention.key = nn.Linear(384, 384)
      self.backbone.encoder.encoder.layer[i].attention.attention.value = nn.Linear(384, 384)
      self.backbone.encoder.encoder.layer[i].attention.dense = nn.Linear(384, 384)
      self.backbone.encoder.encoder.layer[i].lambda1 = Tensor.empty((384))
      self.backbone.encoder.encoder.layer[i].lambda2 = Tensor.empty((384))
      self.backbone.encoder.encoder.layer[i].mlp = Dinov2WithRegistersMLP()
      self.backbone.encoder.encoder.layer[i].mlp.fc1 = nn.Linear(384, 1536)
      self.backbone.encoder.encoder.layer[i].mlp.fc2 = nn.Linear(1536, 384)
    state_dict = safe_load(fetch(f'https://huggingface.co/roryclear/rf-detr/resolve/main/{name}.safetensors'))
    load_state_dict(self, state_dict)

  def __call__(self, img):
    pre = self.preprocess(img)
    predictions = self.predict(pre)
    return predictions[0]

  def predict(self, samples, targets=None):
    _, _, h, w = samples.shape
    mask = Tensor.zeros((1, h, w), dtype=dtypes.bool)
    feature, _ = self.backbone(samples, mask)
    return feature
  
  def preprocess(self, frame):
    img = frame.cast(dtypes.float32)
    img = img[:, :, ::-1]
    img /= 255.0
    img = resize(img, (self.res, self.res))
    img = (img - self.means) / self.stds
    img = img.permute(2, 0, 1).unsqueeze(0)
    return img


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

