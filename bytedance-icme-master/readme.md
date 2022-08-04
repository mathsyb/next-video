# 

### 代码结构
├── data_io.py  
├── features.py  
├── input  
│   ├── final_track2_test_no_anwser.txt  
│   ├── final_track2_train.txt  
│   ├── track2_face_attrs.txt  
│   └── track2_title.txt  
├── output  
│   └── result.csv  
├── readme.md  
└── train.py  

### 运行顺序
1. 创建文件夹input和output
2. 下载数据集并解压到文件夹input
3. 运行train.py训练模型
4. 生成的结果在output文件夹下面

### Todo 
1. 使用深度学习对视频Title进行学习
2. 使用深度学习对视频特征进行学习