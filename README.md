训练，测试，参数可视化方法
1. clone 代码文件夹至本地
2. 进入文件夹，配置环境：使用 conda create -f environment.yml 创建环境
3. 从 google drive 下载数据集 CIFAR10 到本地文件夹
4. 运行 python train.py 进行训练, 可以指定超参数（lr，激活函数，隐藏层维度，L2 正则化强度，batch size，num-
ber of epochs），详见–help。模型会保存在 validation 时准确率最高的模型参数 Model.pkl 至./model_params/Model.pkl，
用于之后的 test.py。
5. 运行 python test.py –model ./model_params/model.pkl，执行测试。可以从训练好的模型地址处下载
Model.pkl 至本地，根据对应地址设置参数进行测试。
6. 运行 python visual_model.py –model ./model_params/model.pkl，执行参数可视化
