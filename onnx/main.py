from PIL import Image, ImageDraw, ImageFont
import onnxruntime as ort
import numpy as np
import cv2
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

image = cv2.imread('image.jpg') # 读取图像
session = ort.InferenceSession('assets/yolo.onnx') # 加载ONNX模型
coco_labels = open('assets/coco_labels.txt').read().splitlines() # 加载标签
output_path = './result' 
score_threshold = 0.4 # 设置置信度阈值
nms_threshold = 0.6 # 设置非最大值抑制阈值

if not os.path.exists(output_path):
    os.makedirs(output_path)

# 获取模型输入大小和输出名称
input_size = session.get_inputs()[0].shape[2:]
input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]

# 将图片转换为模型输入格式并传递给ONNX模型
orig_size = (image.shape[1], image.shape[0]) 
img_arr = cv2.resize(image, input_size)
img_arr = img_arr.astype('float32') / 255.0
img_arr = np.transpose(img_arr, [2, 0, 1])
img_arr = np.expand_dims(img_arr, axis=0)
outputs = session.run(output_names, {input_name: img_arr})

# 获取YOLO框架的检测结果
num = outputs[0][0][0]
boxes = outputs[1][0][:-1].astype('int')
scores = outputs[2][0][:-1]
labels = [coco_labels[i] for i in outputs[3][0][:-1]]

print('\n', boxes, '\n\n', scores, '\n\n', labels, '\n\n', 'num:', num, '\n')

# 应用非最大值抑制
indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold, nms_threshold)

# 为OpenCV图像添加文本
def putText(image, pos, text, font_size=None, color=(0, 255, 0), font_path='assets/font.ttc'):
    if font_size is None:
        font_size = image.shape[1] // 70
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype(font_path, font_size)
    ImageDraw.Draw(pil_image).text(pos, text, font=font, fill=color)
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return image

 # 将边界框坐标转换为原始图像的大小
def transform_bbox(bbox, input_size, orig_size):
    x_scale, y_scale = orig_size[0] / input_size[0], orig_size[1] / input_size[1]
    return [int(val * scale) for val, scale in zip(bbox, [x_scale, y_scale] * 2)]

# 分割图像并保存为独立图像文件
for i,idx in enumerate(indices):
    x, y, w, h = transform_bbox(boxes[idx], input_size, orig_size)
    crop_img = image[y:h, x:w]
    filename = f"{i+1}.{labels[idx]}-{scores[idx]:.3f}.jpg"
    cv2.imwrite(f"{output_path}/{filename}", crop_img)
    print(filename)

print(f"Saved {len(indices)} images in total.")

# 绘制所有边界框及标签
for i,idx in enumerate(indices):
    x, y, w, h = transform_bbox(boxes[idx], input_size, orig_size)
    cv2.rectangle(image, (x, y), (w, h), color=(0, 255, 0), thickness=2)
    image = putText(image, (x+10, y-10), f'{labels[idx]}: {scores[idx]:.3f}')

# 保存原图的边界框
cv2.imwrite(f"{output_path}/image_bbox.jpg", image)