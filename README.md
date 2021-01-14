# LR_Experiment  
## 项目文件说明:
---MNIST&Fashion 对MNISt及Fashion MNIST的测试  
   ---train 训练代码，包括BP，BP+，LRS，LRT，LRS with 0-1 loss， LRT with 0-1 loss  
   ---adv 噪声及对抗样本鲁棒性测试  
---tinyImageNet 对tinyImageNet的测试  
   ---train 训练代码，包括BP-1, BP-2, LRS, LRT, LRS with 0-1 loss, LRT with 0-1 loss, LR with relu  
   ---adv 噪声及对抗样本鲁棒性测试  
---var 统计方差  
## 基本实验结果  
### MNIST  

|Activations + Entropy | Orig | AdvLBFGS | AdvFGSM|  
|  ----  | ----  |  ----  | ----  |  
|BP becnmark| 96.34\%| 57.29\%| 28.37\%|  
| Sigmoid| 95.54\%| 77.60\%| 45.91\%|   
| Threhold|95.33\%|73.30\%|54.35\%|  
| Activations + 0-1 loss |  Orig |  AdvLBFGS |  AdvFGSM| 
|  ----  | ----  |  ----  | ----  |  
| Sigmoid| 84.54\%| 73.50 \%| 58.05\%|  
| Threhold| 90.22\%| 77.74\%| 61.36\%|  




|Activations + Entropy | Orig | Gaussian | Impulse | Glass Blur | contrast|  
|  ----  | ----  |  ----  | ----  |  ----  | ----  |  
|BP becnmark|96.34\%|96.35\%|91.95\%|86.16\%|67.25\%|  
|Sigmoid|95.54\%|95.37\%|93.78\%|87.79\%|76.98\%|  
|Threhold|95.33\%|94.17\%|95.01\%|86.76\%|68.13\%|  
|Activations + 0-1 loss | Orig | Gaussian | Impulse | Glass Blur | contrast|  
|  ----  | ----  |  ----  | ----  |  ----  | ----  |  
|Sigmoid|84.54\%|82.53\%|81.23\%|79.25\%|80.92\%|  
|Threhold|90.22\%|90.54\%|90.10\%|87.19\%|79.81\%|  

