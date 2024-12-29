CNNTransformerModel(
  (conv1): Conv1d(2, 64, kernel_size=(15,), stride=(1,), padding=(7,))
  (bn1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool): MaxPool1d(kernel_size=5, stride=5, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv1d(64, 64, kernel_size=(15,), stride=(1,), padding=(7,))
  (bn2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv3): Conv1d(64, 64, kernel_size=(15,), stride=(1,), padding=(7,))
  (bn3): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv4): Conv1d(64, 64, kernel_size=(6,), stride=(1,), padding=(3,))
  (bn4): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (transformer_encoder): TransformerEncoder(
    (layers): ModuleList(
      (0): TransformerEncoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
        )
        (linear1): Linear(in_features=64, out_features=2048, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=2048, out_features=64, bias=True)
        (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.1, inplace=False)
      )
    )
  )
  (dropout): Dropout(p=0.4, inplace=False)
  (fc3): Linear(in_features=64, out_features=1024, bias=True)
  (fc1): Linear(in_features=1024, out_features=512, bias=True)
  (fc2): Linear(in_features=512, out_features=9, bias=True)
)
Epoch [1/100], Loss: 1.1978, Train Accuracy: 53.38%, Validation Accuracy: 29.35%
Epoch [2/100], Loss: 0.7470, Train Accuracy: 72.50%, Validation Accuracy: 68.21%
Epoch [3/100], Loss: 0.5553, Train Accuracy: 78.55%, Validation Accuracy: 77.83%
Epoch [4/100], Loss: 0.5019, Train Accuracy: 81.47%, Validation Accuracy: 83.56%
Epoch [5/100], Loss: 0.4327, Train Accuracy: 83.60%, Validation Accuracy: 82.10%
Epoch [6/100], Loss: 0.3850, Train Accuracy: 85.54%, Validation Accuracy: 85.87%
Epoch [7/100], Loss: 0.3798, Train Accuracy: 85.78%, Validation Accuracy: 82.95%
Epoch [8/100], Loss: 0.3719, Train Accuracy: 86.43%, Validation Accuracy: 83.80%
Epoch [9/100], Loss: 0.3590, Train Accuracy: 86.57%, Validation Accuracy: 85.51%
Epoch [10/100], Loss: 0.3078, Train Accuracy: 88.52%, Validation Accuracy: 88.19%
Epoch [11/100], Loss: 0.2899, Train Accuracy: 89.17%, Validation Accuracy: 83.68%
Epoch [12/100], Loss: 0.3141, Train Accuracy: 88.19%, Validation Accuracy: 88.43%
Epoch [13/100], Loss: 0.2750, Train Accuracy: 90.00%, Validation Accuracy: 88.31%
Epoch [14/100], Loss: 0.2724, Train Accuracy: 89.63%, Validation Accuracy: 87.82%
Epoch [15/100], Loss: 0.2695, Train Accuracy: 90.27%, Validation Accuracy: 86.85%
Epoch [16/100], Loss: 0.2552, Train Accuracy: 90.29%, Validation Accuracy: 89.16%
Epoch [17/100], Loss: 0.2684, Train Accuracy: 90.15%, Validation Accuracy: 89.28%
Epoch [18/100], Loss: 0.2526, Train Accuracy: 90.77%, Validation Accuracy: 88.79%
Epoch [19/100], Loss: 0.2496, Train Accuracy: 90.83%, Validation Accuracy: 88.43%
Epoch [20/100], Loss: 0.2268, Train Accuracy: 91.72%, Validation Accuracy: 87.45%
Epoch [21/100], Loss: 0.2091, Train Accuracy: 92.33%, Validation Accuracy: 89.52%
Epoch [22/100], Loss: 0.2057, Train Accuracy: 92.68%, Validation Accuracy: 86.36%
Epoch [23/100], Loss: 0.2340, Train Accuracy: 91.02%, Validation Accuracy: 86.85%
Epoch [24/100], Loss: 0.2318, Train Accuracy: 91.03%, Validation Accuracy: 89.77%
Epoch [25/100], Loss: 0.2163, Train Accuracy: 91.99%, Validation Accuracy: 88.67%
Epoch [26/100], Loss: 0.1923, Train Accuracy: 93.13%, Validation Accuracy: 90.62%
Epoch [27/100], Loss: 0.1912, Train Accuracy: 93.44%, Validation Accuracy: 90.62%
Epoch [28/100], Loss: 0.2003, Train Accuracy: 92.42%, Validation Accuracy: 89.77%
Epoch [29/100], Loss: 0.1917, Train Accuracy: 92.51%, Validation Accuracy: 90.01%
Epoch [30/100], Loss: 0.1957, Train Accuracy: 92.49%, Validation Accuracy: 87.33%
Epoch [31/100], Loss: 0.1850, Train Accuracy: 93.10%, Validation Accuracy: 90.13%
Epoch [32/100], Loss: 0.1732, Train Accuracy: 94.08%, Validation Accuracy: 91.47%
Epoch [33/100], Loss: 0.1735, Train Accuracy: 93.47%, Validation Accuracy: 90.74%
Epoch [34/100], Loss: 0.1846, Train Accuracy: 92.86%, Validation Accuracy: 90.74%
Epoch [35/100], Loss: 0.1556, Train Accuracy: 94.29%, Validation Accuracy: 89.40%
Epoch [36/100], Loss: 0.1565, Train Accuracy: 94.34%, Validation Accuracy: 90.50%
Epoch [37/100], Loss: 0.1627, Train Accuracy: 94.00%, Validation Accuracy: 91.60%
Epoch [38/100], Loss: 0.1488, Train Accuracy: 94.67%, Validation Accuracy: 90.13%
Epoch [39/100], Loss: 0.1398, Train Accuracy: 95.04%, Validation Accuracy: 88.79%
Epoch [40/100], Loss: 0.1530, Train Accuracy: 94.35%, Validation Accuracy: 90.13%
Epoch [41/100], Loss: 0.1425, Train Accuracy: 94.81%, Validation Accuracy: 91.11%
Epoch [42/100], Loss: 0.1351, Train Accuracy: 94.81%, Validation Accuracy: 88.92%
Epoch [43/100], Loss: 0.1447, Train Accuracy: 94.91%, Validation Accuracy: 90.62%
Epoch [44/100], Loss: 0.1430, Train Accuracy: 94.95%, Validation Accuracy: 89.52%
Epoch [45/100], Loss: 0.1341, Train Accuracy: 94.75%, Validation Accuracy: 91.35%
Epoch [46/100], Loss: 0.1201, Train Accuracy: 95.97%, Validation Accuracy: 89.89%
Epoch [47/100], Loss: 0.1454, Train Accuracy: 94.70%, Validation Accuracy: 91.23%
Epoch [48/100], Loss: 0.1253, Train Accuracy: 95.45%, Validation Accuracy: 88.43%
Epoch [49/100], Loss: 0.1119, Train Accuracy: 95.77%, Validation Accuracy: 90.01%
Epoch [50/100], Loss: 0.1056, Train Accuracy: 95.95%, Validation Accuracy: 90.99%
Epoch [51/100], Loss: 0.1047, Train Accuracy: 96.32%, Validation Accuracy: 91.11%
Epoch [52/100], Loss: 0.1307, Train Accuracy: 95.23%, Validation Accuracy: 89.52%
Epoch [53/100], Loss: 0.1449, Train Accuracy: 94.85%, Validation Accuracy: 91.72%
Epoch [54/100], Loss: 0.1060, Train Accuracy: 96.33%, Validation Accuracy: 91.35%
Epoch [55/100], Loss: 0.0960, Train Accuracy: 96.60%, Validation Accuracy: 89.77%
Epoch [56/100], Loss: 0.1103, Train Accuracy: 96.13%, Validation Accuracy: 91.11%
Epoch [57/100], Loss: 0.0905, Train Accuracy: 96.68%, Validation Accuracy: 90.62%
Epoch [58/100], Loss: 0.0802, Train Accuracy: 97.17%, Validation Accuracy: 91.23%
Epoch [59/100], Loss: 0.0794, Train Accuracy: 97.06%, Validation Accuracy: 90.13%
Epoch [60/100], Loss: 0.1111, Train Accuracy: 96.12%, Validation Accuracy: 90.26%
Epoch [61/100], Loss: 0.0899, Train Accuracy: 96.91%, Validation Accuracy: 90.13%
Epoch [62/100], Loss: 0.1078, Train Accuracy: 96.45%, Validation Accuracy: 91.35%
Epoch [63/100], Loss: 0.0842, Train Accuracy: 97.03%, Validation Accuracy: 91.23%
Epoch [64/100], Loss: 0.0891, Train Accuracy: 96.71%, Validation Accuracy: 89.40%
Epoch [65/100], Loss: 0.0935, Train Accuracy: 96.53%, Validation Accuracy: 90.26%
Epoch [66/100], Loss: 0.0820, Train Accuracy: 97.24%, Validation Accuracy: 90.01%
Epoch [67/100], Loss: 0.0801, Train Accuracy: 97.23%, Validation Accuracy: 90.74%
Epoch [68/100], Loss: 0.0672, Train Accuracy: 97.69%, Validation Accuracy: 88.06%
Epoch [69/100], Loss: 0.0728, Train Accuracy: 97.50%, Validation Accuracy: 91.35%
Epoch [70/100], Loss: 0.0579, Train Accuracy: 97.90%, Validation Accuracy: 90.86%
Epoch [71/100], Loss: 0.0909, Train Accuracy: 97.03%, Validation Accuracy: 90.26%
Epoch [72/100], Loss: 0.0806, Train Accuracy: 97.23%, Validation Accuracy: 91.11%
Epoch [73/100], Loss: 0.0737, Train Accuracy: 97.49%, Validation Accuracy: 90.62%
Epoch [74/100], Loss: 0.0814, Train Accuracy: 97.12%, Validation Accuracy: 89.89%
Epoch [75/100], Loss: 0.0761, Train Accuracy: 97.43%, Validation Accuracy: 90.62%
Epoch [76/100], Loss: 0.0594, Train Accuracy: 97.88%, Validation Accuracy: 89.89%
Epoch [77/100], Loss: 0.0756, Train Accuracy: 97.41%, Validation Accuracy: 91.11%
Epoch [78/100], Loss: 0.0700, Train Accuracy: 97.50%, Validation Accuracy: 90.50%
Epoch [79/100], Loss: 0.0604, Train Accuracy: 97.93%, Validation Accuracy: 90.62%
Epoch [80/100], Loss: 0.0597, Train Accuracy: 97.98%, Validation Accuracy: 90.62%
Epoch [81/100], Loss: 0.0549, Train Accuracy: 98.16%, Validation Accuracy: 90.62%
Epoch [82/100], Loss: 0.0525, Train Accuracy: 98.17%, Validation Accuracy: 91.11%
Epoch [83/100], Loss: 0.0495, Train Accuracy: 98.22%, Validation Accuracy: 89.28%
Epoch [84/100], Loss: 0.0699, Train Accuracy: 97.64%, Validation Accuracy: 90.50%
Epoch [85/100], Loss: 0.0618, Train Accuracy: 97.82%, Validation Accuracy: 90.86%
Epoch [86/100], Loss: 0.0632, Train Accuracy: 97.91%, Validation Accuracy: 88.43%
Epoch [87/100], Loss: 0.0617, Train Accuracy: 97.98%, Validation Accuracy: 89.40%
Epoch [88/100], Loss: 0.0622, Train Accuracy: 97.85%, Validation Accuracy: 91.11%
Epoch [89/100], Loss: 0.0444, Train Accuracy: 98.72%, Validation Accuracy: 91.72%
Epoch [90/100], Loss: 0.0478, Train Accuracy: 98.48%, Validation Accuracy: 89.28%
Epoch [91/100], Loss: 0.0606, Train Accuracy: 97.94%, Validation Accuracy: 90.86%
Epoch [92/100], Loss: 0.0462, Train Accuracy: 98.45%, Validation Accuracy: 91.72%
Epoch [93/100], Loss: 0.0447, Train Accuracy: 98.58%, Validation Accuracy: 91.23%
Epoch [94/100], Loss: 0.0500, Train Accuracy: 98.49%, Validation Accuracy: 90.74%
Epoch [95/100], Loss: 0.0716, Train Accuracy: 97.56%, Validation Accuracy: 91.35%
Epoch [96/100], Loss: 0.0453, Train Accuracy: 98.54%, Validation Accuracy: 90.13%
Epoch [97/100], Loss: 0.0485, Train Accuracy: 98.64%, Validation Accuracy: 91.11%
Epoch [98/100], Loss: 0.0542, Train Accuracy: 98.36%, Validation Accuracy: 91.84%
Epoch [99/100], Loss: 0.0417, Train Accuracy: 98.64%, Validation Accuracy: 89.52%
Epoch [100/100], Loss: 0.0609, Train Accuracy: 97.94%, Validation Accuracy: 90.74%

Evaluating on Test Set:
Test Accuracy: 93.19%
Test Precision: 0.94
Test Recall: 0.93
Test F1 Score: 0.93
Confusion matrix saved as 'confusion_matrix.png'
Training metrics saved as 'training_metrics.png'
Classification Report:
Class A: Precision = 0.88, Recall = 0.97, F1 = 0.92
Class B: Precision = 0.98, Recall = 0.97, F1 = 0.97
Class F: Precision = 0.99, Recall = 0.78, F1 = 0.87
Class G: Precision = 0.83, Recall = 0.89, F1 = 0.86
Class K: Precision = 0.86, Recall = 0.90, F1 = 0.88
Class M: Precision = 0.96, Recall = 0.99, F1 = 0.98
Class O: Precision = 0.96, Recall = 1.00, F1 = 0.98
Class QSO: Precision = 0.99, Recall = 0.96, F1 = 0.97
Class GALAXY: Precision = 0.99, Recall = 0.98, F1 = 0.98

Macro Precision: 0.94
Macro Recall: 0.94
Macro F1 Score: 0.94
Test Precision: 0.94
Test Recall: 0.94
Test F1 Score: 0.94
Model parameters saved successfully.









CNNTransformerModel(
  (conv1): Conv1d(2, 64, kernel_size=(15,), stride=(1,), padding=(7,))
  (bn1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool): MaxPool1d(kernel_size=5, stride=5, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv1d(64, 64, kernel_size=(15,), stride=(1,), padding=(7,))
  (bn2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv3): Conv1d(64, 64, kernel_size=(15,), stride=(1,), padding=(7,))
  (bn3): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv4): Conv1d(64, 64, kernel_size=(6,), stride=(1,), padding=(3,))
  (bn4): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (transformer_encoder): TransformerEncoder(
    (layers): ModuleList(
      (0): TransformerEncoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
        )
        (linear1): Linear(in_features=64, out_features=2048, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=2048, out_features=64, bias=True)
        (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.1, inplace=False)
      )
    )
  )
  (dropout): Dropout(p=0.4, inplace=False)
  (fc3): Linear(in_features=64, out_features=1024, bias=True)
  (fc1): Linear(in_features=1024, out_features=512, bias=True)
  (fc2): Linear(in_features=512, out_features=9, bias=True)
)
Epoch [1/100], Loss: 1.2879, Train Accuracy: 48.34%, Validation Accuracy: 57.25%
Epoch [2/100], Loss: 0.8333, Train Accuracy: 68.44%, Validation Accuracy: 69.55%
Epoch [3/100], Loss: 0.5763, Train Accuracy: 77.62%, Validation Accuracy: 79.05%
Epoch [4/100], Loss: 0.4924, Train Accuracy: 81.46%, Validation Accuracy: 84.29%
Epoch [5/100], Loss: 0.4180, Train Accuracy: 84.45%, Validation Accuracy: 83.80%
Epoch [6/100], Loss: 0.3934, Train Accuracy: 85.60%, Validation Accuracy: 81.85%
Epoch [7/100], Loss: 0.3629, Train Accuracy: 86.24%, Validation Accuracy: 86.97%
Epoch [8/100], Loss: 0.3401, Train Accuracy: 87.12%, Validation Accuracy: 87.94%
Epoch [9/100], Loss: 0.3080, Train Accuracy: 89.04%, Validation Accuracy: 88.79%
Epoch [10/100], Loss: 0.3108, Train Accuracy: 88.50%, Validation Accuracy: 86.11%
Epoch [11/100], Loss: 0.2793, Train Accuracy: 89.59%, Validation Accuracy: 89.89%
Epoch [12/100], Loss: 0.2685, Train Accuracy: 90.18%, Validation Accuracy: 89.52%
Epoch [13/100], Loss: 0.2841, Train Accuracy: 90.00%, Validation Accuracy: 86.97%
Epoch [14/100], Loss: 0.3063, Train Accuracy: 88.82%, Validation Accuracy: 88.79%
Epoch [15/100], Loss: 0.2665, Train Accuracy: 90.62%, Validation Accuracy: 89.52%
Epoch [16/100], Loss: 0.2392, Train Accuracy: 91.38%, Validation Accuracy: 89.16%
Epoch [17/100], Loss: 0.2491, Train Accuracy: 90.73%, Validation Accuracy: 89.52%
Epoch [18/100], Loss: 0.2365, Train Accuracy: 90.70%, Validation Accuracy: 85.87%
Epoch [19/100], Loss: 0.2380, Train Accuracy: 91.15%, Validation Accuracy: 89.16%
Epoch [20/100], Loss: 0.2138, Train Accuracy: 91.95%, Validation Accuracy: 86.11%
Epoch [21/100], Loss: 0.2161, Train Accuracy: 92.42%, Validation Accuracy: 90.13%
Epoch [22/100], Loss: 0.2152, Train Accuracy: 92.11%, Validation Accuracy: 89.40%
Epoch [23/100], Loss: 0.2118, Train Accuracy: 92.45%, Validation Accuracy: 89.40%
Epoch [24/100], Loss: 0.2032, Train Accuracy: 92.60%, Validation Accuracy: 89.77%
Epoch [25/100], Loss: 0.2281, Train Accuracy: 91.55%, Validation Accuracy: 89.28%
Epoch [26/100], Loss: 0.1983, Train Accuracy: 92.77%, Validation Accuracy: 85.87%
Epoch [27/100], Loss: 0.1897, Train Accuracy: 93.30%, Validation Accuracy: 89.77%
Epoch [28/100], Loss: 0.1815, Train Accuracy: 93.35%, Validation Accuracy: 90.26%
Epoch [29/100], Loss: 0.1958, Train Accuracy: 92.69%, Validation Accuracy: 90.13%
Epoch [30/100], Loss: 0.1610, Train Accuracy: 93.86%, Validation Accuracy: 90.74%
Epoch [31/100], Loss: 0.1878, Train Accuracy: 93.42%, Validation Accuracy: 89.89%
Epoch [32/100], Loss: 0.1665, Train Accuracy: 93.89%, Validation Accuracy: 89.52%
Epoch [33/100], Loss: 0.1658, Train Accuracy: 94.06%, Validation Accuracy: 88.55%
Epoch [34/100], Loss: 0.1486, Train Accuracy: 94.29%, Validation Accuracy: 89.40%
Epoch [35/100], Loss: 0.1668, Train Accuracy: 93.88%, Validation Accuracy: 90.26%
Epoch [36/100], Loss: 0.1655, Train Accuracy: 94.05%, Validation Accuracy: 88.67%
Epoch [37/100], Loss: 0.1679, Train Accuracy: 93.73%, Validation Accuracy: 89.16%
Epoch [38/100], Loss: 0.1444, Train Accuracy: 94.63%, Validation Accuracy: 88.67%
Epoch [39/100], Loss: 0.1244, Train Accuracy: 95.33%, Validation Accuracy: 90.01%
Epoch [40/100], Loss: 0.1329, Train Accuracy: 95.14%, Validation Accuracy: 89.89%
Epoch [41/100], Loss: 0.1281, Train Accuracy: 95.22%, Validation Accuracy: 90.50%
Epoch [42/100], Loss: 0.1306, Train Accuracy: 95.20%, Validation Accuracy: 88.43%
Epoch [43/100], Loss: 0.1307, Train Accuracy: 95.20%, Validation Accuracy: 88.55%
Epoch [44/100], Loss: 0.1463, Train Accuracy: 94.87%, Validation Accuracy: 90.01%
Epoch [45/100], Loss: 0.1221, Train Accuracy: 95.60%, Validation Accuracy: 90.26%
Epoch [46/100], Loss: 0.1112, Train Accuracy: 96.06%, Validation Accuracy: 89.89%
Epoch [47/100], Loss: 0.1019, Train Accuracy: 96.50%, Validation Accuracy: 89.40%
Epoch [48/100], Loss: 0.1529, Train Accuracy: 94.60%, Validation Accuracy: 89.04%
Epoch [49/100], Loss: 0.1318, Train Accuracy: 95.20%, Validation Accuracy: 88.43%
Epoch [50/100], Loss: 0.1065, Train Accuracy: 96.15%, Validation Accuracy: 90.01%
Epoch [51/100], Loss: 0.0986, Train Accuracy: 96.30%, Validation Accuracy: 88.92%
Epoch [52/100], Loss: 0.1022, Train Accuracy: 96.13%, Validation Accuracy: 90.74%
Epoch [53/100], Loss: 0.0881, Train Accuracy: 97.02%, Validation Accuracy: 86.97%
Epoch [54/100], Loss: 0.1574, Train Accuracy: 94.50%, Validation Accuracy: 88.92%
Epoch [55/100], Loss: 0.0848, Train Accuracy: 97.12%, Validation Accuracy: 88.67%
Epoch [56/100], Loss: 0.0993, Train Accuracy: 96.48%, Validation Accuracy: 88.31%
Epoch [57/100], Loss: 0.1069, Train Accuracy: 96.21%, Validation Accuracy: 89.65%
Epoch [58/100], Loss: 0.0853, Train Accuracy: 97.06%, Validation Accuracy: 90.99%
Epoch [59/100], Loss: 0.0927, Train Accuracy: 96.59%, Validation Accuracy: 89.65%
Epoch [60/100], Loss: 0.0796, Train Accuracy: 97.18%, Validation Accuracy: 90.26%
Epoch [61/100], Loss: 0.0972, Train Accuracy: 96.91%, Validation Accuracy: 88.31%
Epoch [62/100], Loss: 0.0832, Train Accuracy: 96.95%, Validation Accuracy: 90.13%
Epoch [63/100], Loss: 0.0933, Train Accuracy: 96.59%, Validation Accuracy: 90.74%
Epoch [64/100], Loss: 0.0856, Train Accuracy: 97.14%, Validation Accuracy: 89.77%
Epoch [65/100], Loss: 0.0788, Train Accuracy: 97.17%, Validation Accuracy: 89.52%
Epoch [66/100], Loss: 0.0861, Train Accuracy: 96.94%, Validation Accuracy: 90.62%
Epoch [67/100], Loss: 0.0771, Train Accuracy: 97.46%, Validation Accuracy: 89.89%
Epoch [68/100], Loss: 0.0798, Train Accuracy: 97.49%, Validation Accuracy: 89.16%
Epoch [69/100], Loss: 0.0741, Train Accuracy: 97.43%, Validation Accuracy: 90.74%
Epoch [70/100], Loss: 0.0642, Train Accuracy: 97.70%, Validation Accuracy: 90.38%
Epoch [71/100], Loss: 0.0657, Train Accuracy: 97.85%, Validation Accuracy: 90.86%
Epoch [72/100], Loss: 0.1082, Train Accuracy: 96.38%, Validation Accuracy: 89.77%
Epoch [73/100], Loss: 0.0737, Train Accuracy: 97.49%, Validation Accuracy: 89.28%
Epoch [74/100], Loss: 0.0705, Train Accuracy: 97.72%, Validation Accuracy: 89.52%
Epoch [75/100], Loss: 0.0605, Train Accuracy: 97.79%, Validation Accuracy: 90.38%
Epoch [76/100], Loss: 0.0737, Train Accuracy: 97.67%, Validation Accuracy: 90.50%
Epoch [77/100], Loss: 0.0630, Train Accuracy: 97.91%, Validation Accuracy: 89.52%
Epoch [78/100], Loss: 0.0544, Train Accuracy: 98.07%, Validation Accuracy: 90.50%
Epoch [79/100], Loss: 0.0663, Train Accuracy: 97.61%, Validation Accuracy: 91.11%
Epoch [80/100], Loss: 0.0448, Train Accuracy: 98.51%, Validation Accuracy: 90.50%
Epoch [81/100], Loss: 0.0477, Train Accuracy: 98.29%, Validation Accuracy: 90.50%
Epoch [82/100], Loss: 0.0521, Train Accuracy: 98.37%, Validation Accuracy: 90.99%
Epoch [83/100], Loss: 0.0461, Train Accuracy: 98.58%, Validation Accuracy: 90.26%
Epoch [84/100], Loss: 0.0601, Train Accuracy: 98.17%, Validation Accuracy: 90.99%
Epoch [85/100], Loss: 0.0805, Train Accuracy: 97.53%, Validation Accuracy: 91.11%
Epoch [86/100], Loss: 0.0598, Train Accuracy: 98.17%, Validation Accuracy: 89.77%
Epoch [87/100], Loss: 0.0486, Train Accuracy: 98.52%, Validation Accuracy: 90.38%
Epoch [88/100], Loss: 0.0669, Train Accuracy: 97.90%, Validation Accuracy: 91.11%
Epoch [89/100], Loss: 0.0473, Train Accuracy: 98.63%, Validation Accuracy: 90.74%
Epoch [90/100], Loss: 0.0540, Train Accuracy: 98.33%, Validation Accuracy: 90.50%
Epoch [91/100], Loss: 0.0384, Train Accuracy: 98.84%, Validation Accuracy: 91.47%
Epoch [92/100], Loss: 0.0535, Train Accuracy: 98.36%, Validation Accuracy: 89.52%
Epoch [93/100], Loss: 0.0484, Train Accuracy: 98.51%, Validation Accuracy: 90.13%
Epoch [94/100], Loss: 0.0577, Train Accuracy: 98.02%, Validation Accuracy: 89.89%
Epoch [95/100], Loss: 0.0373, Train Accuracy: 98.78%, Validation Accuracy: 89.65%
Epoch [96/100], Loss: 0.0545, Train Accuracy: 98.45%, Validation Accuracy: 90.26%
Epoch [97/100], Loss: 0.0478, Train Accuracy: 98.39%, Validation Accuracy: 89.89%
Epoch [98/100], Loss: 0.0434, Train Accuracy: 98.63%, Validation Accuracy: 91.11%
Epoch [99/100], Loss: 0.0378, Train Accuracy: 98.78%, Validation Accuracy: 90.50%
Epoch [100/100], Loss: 0.0510, Train Accuracy: 98.31%, Validation Accuracy: 91.96%

Evaluating on Test Set:
Test Accuracy: 93.07%
Test Precision: 0.93
Test Recall: 0.93
Test F1 Score: 0.93
Confusion matrix saved as 'confusion_matrix.png'
Training metrics saved as 'training_metrics.png'
Classification Report:
Class A: Precision = 0.93, Recall = 0.95, F1 = 0.94
Class B: Precision = 0.98, Recall = 0.95, F1 = 0.96
Class F: Precision = 0.87, Recall = 0.93, F1 = 0.90
Class G: Precision = 0.87, Recall = 0.83, F1 = 0.85
Class K: Precision = 0.90, Recall = 0.89, F1 = 0.89
Class M: Precision = 0.99, Recall = 0.98, F1 = 0.98
Class O: Precision = 0.84, Recall = 0.95, F1 = 0.89
Class QSO: Precision = 0.96, Recall = 0.97, F1 = 0.97
Class GALAXY: Precision = 0.97, Recall = 0.94, F1 = 0.95

Macro Precision: 0.92
Macro Recall: 0.93
Macro F1 Score: 0.93
Test Precision: 0.92
Test Recall: 0.93
Test F1 Score: 0.93
Model parameters saved successfully.