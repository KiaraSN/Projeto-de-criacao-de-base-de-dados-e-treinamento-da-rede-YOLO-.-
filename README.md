# Projeto-de-criacao-de-base-de-dados-e-treinamento-da-rede-YOLO-.-

# Projeto: Detecção de Objetos com YOLO no Colab

## Descrição
Detecção de objetos em imagens usando YOLOv5 pré-treinado. Objetos como cães, bicicletas e carros são identificados e destacados com caixas delimitadoras coloridas.

---

## Requisitos
- Google Colab (GPU recomendada)
- Bibliotecas:
  - `torch`
  - `matplotlib`
  - `opencv-python`
  - `ultralytics/yolov5` (via `torch.hub`)

---

## Passo a Passo

### 1. Upload da imagem
```python
from google.colab import files
uploaded = files.upload()
for fn in uploaded.keys():
    print(fn)  # mostra o nome do arquivo
2. Carregar modelo YOLOv5 pré-treinado
python
Copiar código
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
3. Rodar detecção e salvar resultados
python
Copiar código
import cv2
from matplotlib import pyplot as plt

# Substitua pelo nome do arquivo enviado
img_path = "dog_bike.jpg"

# Detecção
results = model(img_path)
results.save()  # salva em runs/detect/exp
4. Exibir imagem detectada
python
Copiar código
# Caminho da imagem de saída
output_path = results.files[0]

# Ler e exibir
img = cv2.imread(output_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10,10))
plt.imshow(img)
plt.axis('off')
plt.title("Resultado da detecção YOLO")
plt.show()
Observações
Cada execução salva a detecção em runs/detect/exp.

Para múltiplas imagens, repita o upload e a detecção.

Para treinar novas classes, utilize transfer learning com dataset rotulado.

Referências
YOLOv5 GitHub

YOLO - You Only Look Once

OpenCV

Google Colab
