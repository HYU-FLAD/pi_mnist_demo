import flwr as fl
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict

# 1. 모델 구조 (평가용)
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

# 2. 평가 함수
def get_eval_fn(model):
    # 테스트 데이터셋 로드
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64)

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config):
        # 모델에 파라미터 적용
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        correct = 0
        total = 0
        device = torch.device("cpu")
        model.to(device)

        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f" [Round {server_round}] 글로벌 모델 정확도: {accuracy * 100:.2f}%")
        
        # 마지막 라운드에 모델 저장
        if server_round == 5: 
            torch.save(model.state_dict(), "backdoor_model.pth")
            print(" 모델 저장 완료")
            
        # ★★★ 여기가 수정된 부분입니다 ★★★
        return 0.0, {"accuracy": accuracy} 

    return evaluate

def main():
    model = Net()
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=1,
        min_available_clients=1,
        evaluate_fn=get_eval_fn(model)
    )

    print("서버 시작 정확도 검증 기능 포함")
    fl.server.start_server(
        server_address="0.0.0.0:8080", 
        config=fl.server.ServerConfig(num_rounds=5, round_timeout=None),
        strategy=strategy
    )

if __name__ == "__main__":
    main()