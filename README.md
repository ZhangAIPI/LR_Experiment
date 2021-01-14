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
#### 对抗鲁棒性
|Activations + Entropy | Orig | AdvLBFGS | AdvFGSM|  
|  ----  | ----  |  ----  | ----  |  
|BP becnmark| 96.34\%| 57.29\%| 28.37\%|  
| Sigmoid| 95.54\%| 77.60\%| 45.91\%|   
| Threhold|95.33\%|73.30\%|54.35\%|  

| Activations + 0-1 loss |  Orig |  AdvLBFGS |  AdvFGSM| 
|  ----  | ----  |  ----  | ----  |  
| Sigmoid| 84.54\%| 73.50 \%| 58.05\%|  
| Threhold| 90.22\%| 77.74\%| 61.36\%|  



### 噪声鲁棒性
|Activations + Entropy | Orig | Gaussian | Impulse | Glass Blur | contrast|  
|  ----  | ----  |  ----  | ----  |  ----  | ----  |  
|BP becnmark|96.34\%|96.35\%|91.95\%|86.16\%|67.25\%|  
|Sigmoid|95.54\%|95.37\%|93.78\%|87.79\%|76.98\%|  
|Threhold|95.33\%|94.17\%|95.01\%|86.76\%|68.13\%|  

|Activations + 0-1 loss | Orig | Gaussian | Impulse | Glass Blur | contrast|  
|  ----  | ----  |  ----  | ----  |  ----  | ----  |  
|Sigmoid|84.54\%|82.53\%|81.23\%|79.25\%|80.92\%|  
|Threhold|90.22\%|90.54\%|90.10\%|87.19\%|79.81\%|  

### TinyImageNet

|Activations + Entropy | Orig | AdvLBFGS | AdvFGSM|  
|  ----  | ----  |  ----  | ----  |  
|BP becnmark&21.70\%&11.6\%|8.70\%|  
|Sigmoid|20.30\%|14.6\%|17.04\%|  
|Threhold|17.30\%|15.6\%|14.34\%|  

Activations + 0-1 loss & Orig & AdvLBFGS & AdvFGSM\\
|  ----  | ----  |  ----  | ----  |  
|Sigmoid|15.22\%|7.00 \%|14.35\%|  
|Threhold|19.57\%|13.6\%|16.52\%|  


|Activations + Entropy | Orig | Gaussian | Impulse | Glass Blur | contrast|  
|  ----  | ----  |  ----  | ----  |  ----  | ----  | 
|BP becnmark|21.70\%|12.68\%|12.52\%|13.40\%|11.36\%|  
|Sigmoid|20.30\%|17.47\%|17.52\%|17|.68\%|1|.28\%| |
|Threhold|17.30\%|16.24\%|16.48\%|1|6.96\%|1|.88\%|  

|Activations + Entropy | Orig | Gaussian | Impulse | Glass Blur | contrast|  
|  ----  | ----  |  ----  | ----  |  ----  | ----  | 
|Sigmoid|15.22\%|14.56\%|14.47\%|14.68\%|14.40\%|  
|Threhold|19.57\%|15.40\%|15.48\%|15.92\%|13.99\%|  

