# LR_Experiment  
## 项目说明
通过LR方法训练人工神经网络提高模型噪声及对抗鲁棒性

## 项目文件说明:
---MNIST&Fashion 对MNISt及Fashion MNIST的测试  
   ---train 训练代码，包括BP，BP+，LRS，LRT，LRS with 0-1 loss， LRT with 0-1 loss  
   ---adv 噪声及对抗样本鲁棒性测试  
---tinyImageNet 对tinyImageNet的测试  
   ---train 训练代码，包括BP-1, BP-2, LRS, LRT, LRS with 0-1 loss, LRT with 0-1 loss, LR with relu  
   ---adv 噪声及对抗样本鲁棒性测试  
---var 统计方差  

## 运行环境及使用说明
python==3.7.4  pytorch==1.6.0  
各文件基本独立存在，执行测试功能，直接运行即可

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

### FashionMnist

|Activations + Entropy | Orig | AdvLBFGS | AdvFGSM|  
|  ----  | ----  |  ----  | ----  |  
|BP becnmark|86.45\%|49.61\%|17.86\%|  
|Sigmoid|83.93\%|53.54\%|52.85\%|  
|Threhold|85.54\%|52.92\%|42.62|  

|Activations + 0-1 loss | Orig | AdvLBFGS | AdvFGSM|
|  ----  | ----  |  ----  | ----  |  
|Sigmoid|58.68\%|51.00\%|46.35\%|  
|Threhold|62.05\%|60.15\%|57.19\%|  


|Activations + Entropy | Orig | Gaussian | Impulse | Glass Blur | contrast|  
|  ----  | ----  |  ----  | ----  |  ----  | ----  | 
|BP becnmark|86.45\%|81.05\%|65.40\%|55.02\%|41.05\%|  
|Sigmoid|83.93\%|82.53\%|80.23\%|79.09\%|49.96\%|  
|Threhold|85.54\%|82.79\%|82.65\%|77.16\%|45.82\%|   

|Activations + Entropy | Orig | Gaussian | Impulse | Glass Blur | contrast|  
|  ----  | ----  |  ----  | ----  |  ----  | ----  | 
|Sigmoid|58.68\%|54.26\%|52.43\%|51.05\%|40.77\%|  
|Threhold|62.05\%|57.93\%|57.25\%|53.13\%|44.68\%|  


### TinyImageNet

|Activations + Entropy | Orig | AdvLBFGS | AdvFGSM|  
|  ----  | ----  |  ----  | ----  |  
|BP becnmark|21.70\%|11.6\%|8.70\%|  
|Sigmoid|20.30\%|14.6\%|17.04\%|  
|Threhold|17.30\%|15.6\%|14.34\%|  

Activations + 0-1 loss | Orig | AdvLBFGS | AdvFGSM|  
|  ----  | ----  |  ----  | ----  |  
|Sigmoid|15.22\%|7.00 \%|14.35\%|  
|Threhold|19.57\%|13.6\%|16.52\%|  


|Activations + Entropy | Orig | Gaussian | Impulse | Glass Blur | contrast|  
|  ----  | ----  |  ----  | ----  |  ----  | ----  | 
|BP becnmark|21.70\%|12.68\%|12.52\%|13.40\%|11.36\%|  
|Sigmoid|20.30\%|17.47\%|17.52\%|17.68\%|16.28\%|  
|Threhold|17.30\%|16.24\%|16.48\%|16.96\%|14.88\%|  

|Activations + Entropy | Orig | Gaussian | Impulse | Glass Blur | contrast|  
|  ----  | ----  |  ----  | ----  |  ----  | ----  | 
|Sigmoid|15.22\%|14.56\%|14.47\%|14.68\%|14.40\%|  
|Threhold|19.57\%|15.40\%|15.48\%|15.92\%|13.99\%|  

