import torch

def predict_with_pytorch(model, features):
    # features shape: (n_features, time)
    tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, n_features, time)
    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).item()
        prob = torch.softmax(output, dim=1).cpu().numpy()[0]
    return pred, prob
