## Balanced(Diversed+Accurate) Video Game Recommendation via Contextualized Graph Neural Network

> **Abstract:** 
在当今众多在线游戏平台中，游戏推荐系统对用户和平台都起着至关重要的作用。对用户而言，它能够帮助发现更多潜在的感兴趣游戏；对平台而言，则能够提升用户停留时间和参与度。本研究深入分析了Steam平台上用户的游戏行为特征。基于观察结果，一个理想的游戏推荐系统应具备三个关键特性：个性化推荐、游戏情境化和社交关系利用。然而，同时实现这三个目标具有相当的挑战性。首先，游戏推荐的个性化需要考虑用户在游戏中的停留时间，这在现有方法中往往被忽视。其次，游戏情境化应当反映游戏之间复杂的高阶关系。最后，由于社交关系中存在大量噪声，直接将其用于游戏推荐可能会产生问题。同时，现有游戏推荐系统往往存在推荐结果过于同质化的问题，在这里我引入了Kmeans聚类算法以及多样性损失，通过聚类增强推荐结果的多样性。
综合上述观察实验，我设计了一个推荐系统，该系统通过以下创新方法提升推荐效果：
1. 利用K-means聚类算法对游戏向量进行聚类，引入多样性损失，增强推荐结果的多样性
2. 引入基于注意力机制的用户兴趣聚合模块，更好地捕捉用户的个性化偏好
3. 利用社交关系图谱聚合好友之间的兴趣爱好，增强推荐的准确性
4. 采用对比学习策略，提升模型的学习效果

## Dataset

[Google drive link](https://drive.google.com/file/d/1F9kr_YWimBtexJEH-zkDzCOwl1q7GmFp/view)


## How to run
python main.py
environment: python 3.8 + pytorch 2.40 + cuda 12.1 + dgl py2.4+cu12.1


