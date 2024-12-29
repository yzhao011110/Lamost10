CNNLSTMModel(
  (conv1): Conv1d(2, 20, kernel_size=(15,), stride=(1,), padding=(7,))
  (bn1): BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool): MaxPool1d(kernel_size=5, stride=5, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv1d(20, 20, kernel_size=(15,), stride=(1,), padding=(7,))
  (bn2): BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv3): Conv1d(20, 20, kernel_size=(15,), stride=(1,), padding=(7,))
  (bn3): BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv4): Conv1d(20, 15, kernel_size=(6,), stride=(1,), padding=(3,))
  (bn4): BatchNorm1d(15, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (lstm): LSTM(15, 128, num_layers=3, batch_first=True)
  (dropout): Dropout(p=0.3, inplace=False)
  (fc1): Linear(in_features=128, out_features=1024, bias=True)
  (fc2): Linear(in_features=1024, out_features=9, bias=True)
)
Epoch [1/100], Loss: 1.6663, Accuracy: 31.66%
Validation Accuracy: 47.62%
Epoch [2/100], Loss: 1.0061, Accuracy: 59.22%
Validation Accuracy: 62.47%
Epoch [3/100], Loss: 0.6949, Accuracy: 74.47%
Validation Accuracy: 73.63%
Epoch [4/100], Loss: 0.5514, Accuracy: 79.97%
Validation Accuracy: 67.46%
Epoch [5/100], Loss: 0.4693, Accuracy: 82.57%
Validation Accuracy: 77.43%
Epoch [6/100], Loss: 0.4261, Accuracy: 84.32%
Validation Accuracy: 78.03%
Epoch [7/100], Loss: 0.4418, Accuracy: 83.27%
Validation Accuracy: 84.44%
Epoch [8/100], Loss: 0.3959, Accuracy: 84.67%
Validation Accuracy: 80.88%
Epoch [9/100], Loss: 0.3704, Accuracy: 85.96%
Validation Accuracy: 79.33%
Epoch [10/100], Loss: 0.3666, Accuracy: 86.38%
Validation Accuracy: 85.39%
Epoch [11/100], Loss: 0.3448, Accuracy: 87.09%
Validation Accuracy: 83.37%
Epoch [12/100], Loss: 0.3529, Accuracy: 86.81%
Validation Accuracy: 87.05%
Epoch [13/100], Loss: 0.3539, Accuracy: 86.55%
Validation Accuracy: 85.51%
Epoch [14/100], Loss: 0.3020, Accuracy: 88.74%
Validation Accuracy: 87.17%
Epoch [15/100], Loss: 0.3266, Accuracy: 87.65%
Validation Accuracy: 80.64%
Epoch [16/100], Loss: 0.3064, Accuracy: 88.18%
Validation Accuracy: 86.82%
Epoch [17/100], Loss: 0.3096, Accuracy: 88.60%
Validation Accuracy: 87.41%
Epoch [18/100], Loss: 0.2987, Accuracy: 88.44%
Validation Accuracy: 83.02%
Epoch [19/100], Loss: 0.3029, Accuracy: 88.21%
Validation Accuracy: 87.65%
Epoch [20/100], Loss: 0.2798, Accuracy: 89.45%
Validation Accuracy: 86.94%
Epoch [21/100], Loss: 0.2878, Accuracy: 89.42%
Validation Accuracy: 86.10%
Epoch [22/100], Loss: 0.2901, Accuracy: 89.05%
Validation Accuracy: 84.32%
Epoch [23/100], Loss: 0.2590, Accuracy: 90.38%
Validation Accuracy: 87.89%
Epoch [24/100], Loss: 0.2451, Accuracy: 90.71%
Validation Accuracy: 85.51%
Epoch [25/100], Loss: 0.2471, Accuracy: 90.66%
Validation Accuracy: 88.72%
Epoch [26/100], Loss: 0.2407, Accuracy: 90.81%
Validation Accuracy: 85.15%
Epoch [27/100], Loss: 0.2624, Accuracy: 90.27%
Validation Accuracy: 85.63%
Epoch [28/100], Loss: 0.2300, Accuracy: 91.23%
Validation Accuracy: 88.72%
Epoch [29/100], Loss: 0.2192, Accuracy: 91.70%
Validation Accuracy: 88.24%
Epoch [30/100], Loss: 0.2427, Accuracy: 90.87%
Validation Accuracy: 85.63%
Epoch [31/100], Loss: 0.2195, Accuracy: 91.95%
Validation Accuracy: 87.89%
Epoch [32/100], Loss: 0.2002, Accuracy: 92.33%
Validation Accuracy: 87.89%
Epoch [33/100], Loss: 0.2298, Accuracy: 91.27%
Validation Accuracy: 86.70%
Epoch [34/100], Loss: 0.2118, Accuracy: 92.07%
Validation Accuracy: 86.10%
Epoch [35/100], Loss: 0.2163, Accuracy: 91.62%
Validation Accuracy: 86.22%
Epoch [36/100], Loss: 0.2112, Accuracy: 91.76%
Validation Accuracy: 87.41%
Epoch [37/100], Loss: 0.2003, Accuracy: 92.40%
Validation Accuracy: 87.05%
Epoch [38/100], Loss: 0.1996, Accuracy: 91.97%
Validation Accuracy: 88.48%
Epoch [39/100], Loss: 0.2114, Accuracy: 92.09%
Validation Accuracy: 89.07%
Epoch [40/100], Loss: 0.1849, Accuracy: 93.19%
Validation Accuracy: 86.58%
Epoch [41/100], Loss: 0.1979, Accuracy: 92.56%
Validation Accuracy: 88.12%
Epoch [42/100], Loss: 0.1994, Accuracy: 92.18%
Validation Accuracy: 87.05%
Epoch [43/100], Loss: 0.1946, Accuracy: 92.61%
Validation Accuracy: 87.41%
Epoch [44/100], Loss: 0.1731, Accuracy: 93.57%
Validation Accuracy: 85.04%
Epoch [45/100], Loss: 0.1752, Accuracy: 93.47%
Validation Accuracy: 86.70%
Epoch [46/100], Loss: 0.1912, Accuracy: 92.75%
Validation Accuracy: 85.51%
Epoch [47/100], Loss: 0.1968, Accuracy: 92.28%
Validation Accuracy: 86.46%
Epoch [48/100], Loss: 0.1617, Accuracy: 93.99%
Validation Accuracy: 88.12%
Epoch [49/100], Loss: 0.1533, Accuracy: 94.36%
Validation Accuracy: 86.22%
Epoch [50/100], Loss: 0.1756, Accuracy: 93.56%
Validation Accuracy: 88.84%
Epoch [51/100], Loss: 0.1529, Accuracy: 94.36%
Validation Accuracy: 88.72%
Epoch [52/100], Loss: 0.1658, Accuracy: 93.73%
Validation Accuracy: 83.73%
Epoch [53/100], Loss: 0.1580, Accuracy: 94.04%
Validation Accuracy: 84.68%
Epoch [54/100], Loss: 0.1725, Accuracy: 93.35%
Validation Accuracy: 88.84%
Epoch [55/100], Loss: 0.1474, Accuracy: 94.52%
Validation Accuracy: 88.48%
Epoch [56/100], Loss: 0.1672, Accuracy: 94.15%
Validation Accuracy: 87.29%
Epoch [57/100], Loss: 0.1550, Accuracy: 94.15%
Validation Accuracy: 88.48%
Epoch [58/100], Loss: 0.1459, Accuracy: 94.62%
Validation Accuracy: 88.36%
Epoch [59/100], Loss: 0.1534, Accuracy: 93.90%
Validation Accuracy: 88.12%
Epoch [60/100], Loss: 0.1810, Accuracy: 93.52%
Validation Accuracy: 88.72%
Epoch [61/100], Loss: 0.1363, Accuracy: 94.90%
Validation Accuracy: 88.36%
Epoch [62/100], Loss: 0.1269, Accuracy: 95.16%
Validation Accuracy: 88.24%
Epoch [63/100], Loss: 0.1283, Accuracy: 95.04%
Validation Accuracy: 87.53%
Epoch [64/100], Loss: 0.1473, Accuracy: 94.57%
Validation Accuracy: 87.05%
Epoch [65/100], Loss: 0.1393, Accuracy: 94.57%
Validation Accuracy: 87.41%
Epoch [66/100], Loss: 0.1175, Accuracy: 95.53%
Validation Accuracy: 89.90%
Epoch [67/100], Loss: 0.1125, Accuracy: 95.70%
Validation Accuracy: 89.43%
Epoch [68/100], Loss: 0.1525, Accuracy: 94.74%
Validation Accuracy: 87.89%
Epoch [69/100], Loss: 0.1312, Accuracy: 95.16%
Validation Accuracy: 88.36%
Epoch [70/100], Loss: 0.1239, Accuracy: 95.34%
Validation Accuracy: 87.77%
Epoch [71/100], Loss: 0.1129, Accuracy: 95.77%
Validation Accuracy: 86.46%
Epoch [72/100], Loss: 0.1176, Accuracy: 95.77%
Validation Accuracy: 87.41%
Epoch [73/100], Loss: 0.1261, Accuracy: 95.48%
Validation Accuracy: 88.72%
Epoch [74/100], Loss: 0.0936, Accuracy: 96.63%
Validation Accuracy: 86.94%
Epoch [75/100], Loss: 0.1163, Accuracy: 95.79%
Validation Accuracy: 88.60%
Epoch [76/100], Loss: 0.1099, Accuracy: 96.11%
Validation Accuracy: 85.04%
Epoch [77/100], Loss: 0.1168, Accuracy: 95.72%
Validation Accuracy: 87.53%
Epoch [78/100], Loss: 0.1102, Accuracy: 95.97%
Validation Accuracy: 88.48%
Epoch [79/100], Loss: 0.1222, Accuracy: 95.69%
Validation Accuracy: 89.07%
Epoch [80/100], Loss: 0.1239, Accuracy: 95.86%
Validation Accuracy: 87.53%
Epoch [81/100], Loss: 0.1101, Accuracy: 95.91%
Validation Accuracy: 88.84%
Epoch [82/100], Loss: 0.0998, Accuracy: 96.37%
Validation Accuracy: 88.24%
Epoch [83/100], Loss: 0.1149, Accuracy: 95.72%
Validation Accuracy: 88.36%
Epoch [84/100], Loss: 0.1186, Accuracy: 95.53%
Validation Accuracy: 86.46%
Epoch [85/100], Loss: 0.1295, Accuracy: 95.20%
Validation Accuracy: 90.02%
Epoch [86/100], Loss: 0.1026, Accuracy: 96.51%
Validation Accuracy: 86.82%
Epoch [87/100], Loss: 0.0978, Accuracy: 96.28%
Validation Accuracy: 87.41%
Epoch [88/100], Loss: 0.0898, Accuracy: 96.82%
Validation Accuracy: 87.41%
Epoch [89/100], Loss: 0.1273, Accuracy: 95.21%
Validation Accuracy: 90.02%
Epoch [90/100], Loss: 0.0790, Accuracy: 97.22%
Validation Accuracy: 88.48%
Epoch [91/100], Loss: 0.1212, Accuracy: 95.77%
Validation Accuracy: 84.44%
Epoch [92/100], Loss: 0.1363, Accuracy: 95.32%
Validation Accuracy: 89.90%
Epoch [93/100], Loss: 0.0949, Accuracy: 96.77%
Validation Accuracy: 89.19%
Epoch [94/100], Loss: 0.0917, Accuracy: 96.86%
Validation Accuracy: 87.77%
Epoch [95/100], Loss: 0.0615, Accuracy: 98.04%
Validation Accuracy: 88.95%
Epoch [96/100], Loss: 0.0844, Accuracy: 96.86%
Validation Accuracy: 88.24%
Epoch [97/100], Loss: 0.0815, Accuracy: 97.35%
Validation Accuracy: 88.84%
Epoch [98/100], Loss: 0.1011, Accuracy: 96.30%
Validation Accuracy: 87.41%
Epoch [99/100], Loss: 0.0826, Accuracy: 96.75%
Validation Accuracy: 88.00%
Epoch [100/100], Loss: 0.0737, Accuracy: 97.57%
Validation Accuracy: 88.00%
Test Accuracy: 89.96%