import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import copy

# 1. 모델 구조 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. 모델 2개 로드 및 양자화 
device = torch.device("cpu")

def load_and_optimize(path):
    model = Net().to(device)
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        # 양자화 
        model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        return model
    except:
        print(f"❌ {path} 파일이 없습니다!")
        exit()

print("⚡ 모델 로딩 중...")
model_clean = load_and_optimize("clean_model.pth")     # 정상 모델
model_backdoor = load_and_optimize("backdoor_model.pth") # 백도어 모델
print("✅ 두 모델 모두 로드 및 최적화 완료!")

# 전처리
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 3. 카메라 설정
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

frame_count = 0
res_clean = "Init"
res_bad = "Init"
col_clean = (0, 255, 0)
col_bad = (0, 255, 0)

print("🎥 비교 시연 시작! (종료: q)")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame_count += 1
    
    # === 4프레임마다 추론 (Pi 3B+ 부하 줄이기) ===
    if frame_count % 4 == 0:
        # 전처리 공통 수행
        h, w, _ = frame.shape
        roi_size = 140
        x1 = int(w/2 - roi_size/2)
        y1 = int(h/2 - roi_size/2)
        roi = frame[y1:y1+roi_size, x1:x1+roi_size]
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        
        pil_img = Image.fromarray(thresh)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # A. 정상 모델 추론
            out1 = model_clean(input_tensor)
            pred1 = out1.argmax(dim=1).item()
            
            # B. 백도어 모델 추론
            out2 = model_backdoor(input_tensor)
            pred2 = out2.argmax(dim=1).item()
            
            # 결과 텍스트 설정
            res_clean = f"Clean: {pred1}"
            col_clean = (0, 255, 0) # 항상 초록(정상)이어야 함
            
            if pred2 == 0: # 백도어 타겟(0)
                res_bad = f"BACKDOOR! ({pred2})"
                col_bad = (0, 0, 255) # 빨강 (위험)
            else:
                res_bad = f"Infected: {pred2}"
                col_bad = (0, 255, 0) # 초록

    # === 화면 그리기 ===
    # 화면을 2개로 복사
    frame_clean_view = frame.copy()
    frame_backdoor_view = frame.copy()
    
    # 박스 좌표
    h, w, _ = frame.shape
    roi_size = 140
    x1 = int(w/2 - roi_size/2)
    y1 = int(h/2 - roi_size/2)

    # 1. 왼쪽 창 (정상 모델)
    cv2.rectangle(frame_clean_view, (x1, y1), (x1+roi_size, y1+roi_size), col_clean, 2)
    cv2.putText(frame_clean_view, res_clean, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_clean, 2)
    cv2.putText(frame_clean_view, "[Clean Model]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # 2. 오른쪽 창 (백도어 모델)
    cv2.rectangle(frame_backdoor_view, (x1, y1), (x1+roi_size, y1+roi_size), col_bad, 2)
    cv2.putText(frame_backdoor_view, res_bad, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_bad, 2)
    cv2.putText(frame_backdoor_view, "[Backdoor Model]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 창 띄우기
    cv2.imshow('1. Clean Model (Safe)', frame_clean_view)
    cv2.imshow('2. Backdoor Model (Hacked)', frame_backdoor_view)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()