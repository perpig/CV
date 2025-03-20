# 人脸识别项目解释文档 - 验收

## melt_face_identical.py

> 基于MACOSX数据集

基于孔馨怡同学代码的主要模型，包括pca和knn模型构建与参数选择

主要体现了```n_neighbors```和```n_components```参数选择的结果影响

加入了pca模型的参数网格搜索（准确率结果无选择方差累计值95%时高，已注释）

删改了一些看起来像是gpt老师写的东西

## face_identical_lfw.py

> 基于lfw数据集

基于我个人实验代码，同样包括pca和knn模型构建

其中使用的参数为```parameter_search.py```中取得的最优值

其中对```min_faces_per_person```进行了限制，以讨论**分类个数**对结果的影响

## parameter_search.py

对pca和knn模型参数的网格搜索

## test_hog.py

探讨hog模型可否对准确率有进一步帮助

（实验结果：无明显辅助）

## 另一个md文件

原验收准备md文档


