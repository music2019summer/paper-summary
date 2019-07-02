# Wide & Deep Learning for Recommender Systems 摘录

梁聪 liangxcong@gmail.com 2019.7.2

## INTRODUCTION

wide: 需要feature engineering，但是高效可解释；(MEM ok)

deep: 对于稀疏和high-rank 矩阵不好。(GEN ok)

（Wide linear models can effectively memorize sparse feature interactions using cross-product feature transformations, while deep neural networks can generalize to previously unseen feature interactions through low-dimensional embeddings.）

`Memorizing`, given facts, is an obvious task in learning. This can be done by storing the input samples explicitly, or by identifying the concept behind the input data, and memorizing their general rules.

The ability to identify the rules, to `generalize`, allows the system to make predictions on unknown data.

Recommendations based on memorization（可以通过点积实现） are usually more topical and directly relevant to the items on which users have already performed actions. Compared with memorization, generalization（如果要通过点积实现，则可以使用更笼统的feature，如`AND(user_installed_category=video,
impression_category=music)`，但是不会出现没有出现过的组合） tends to improve the diversity of the recommended items.

`Embedding-based model`：Word embeddings are a type of word representation that allows words with similar meaning to have a similar representation.

“Embedding-based model， such as factorization machines or deep neural networks.
“

The Wide & Deep learning framework for jointly training feed-forward neural networks（参数从输入层向输出层单向传播，无环） with embeddings and linear model with feature transformations for generic recommender systems with sparse inputs.



## RECOMMENDER SYSTEM OVERVIEW

Play Store：

`query`

-a combination of machine-learned models and human-defined rules-

->`retrieval`

->`ranking`

## WIDE & DEEP LEARNING

### Wide

Linear model of the form $y = w^Tx + b$, x is a vector of d features（包括raw input features 和transformed features）, w is the model parameters and b is the bias.

Transformed features: 例如$\phi_k(x) = \prod_{i=1}^{d}x_i^{c_{k_{i}}}, c_{k_i} ={0,1} $。当该transformation$\phi_k$包含该feature时，c为1，否则为0。This captures the interactions between the binary features, and adds nonlinearity to the generalized linear model.

### Deep

The deep component is a feed-forward neural network.

1. Sparse, high-dimensional categorical features as inputs.
2. Convert inputs into a low-dimensional and dense real-valued vector（embedding vector(O(10) to O(100))）.
3. Feed the vectors into the hidden layers of a neural network in the foward pass: $a^{(l+1)} = f(W^{(l)}a^{(l)} + b^{(l)})$ (l-th layer), f is the activation function(Often ReLUs).

### Joint training

The wide component and deep component are combined using a weighted sum of their output log odds as the prediction, which is then fed to one common logistic loss function for joint training.

> Joint training & Ensemble:
>
> 1. 前者在training 时就考虑了各个feature 的权重，更新时同步考虑两个部分；而后者training 的时候相互独立，只有最后一步才合在一起。
> 2. 前者WIDE 部分的样本数不需要太大，作为DEEP 的补充即可；后者要足够的样本数目。

Joint training of a Wide & Deep Model is done by back-propagating（反向传播） the gradients from the output to both the wide and deep part of the model simultaneously using mini-batch stochastic optimization（WIDE：FTRL， DEEP：AdaGrad）.

For a logistic regression problem, the model’s prediction is: $P(Y=1|X) = \sigma(w^T_{wide}[x, \phi(x)] + w^T_{deep}a^{lf} + b)$, 包含了WIDE 和DEEP 两部分。

## SYSTEM IMPLEMENTATION

### Data Generation

`Vocabularies`: tables mapping categorical feature strings to integer IDs.

Continuous real-valued features are normalized to [0, 1].

### Model Training

#### Input

Input layer takes in training data and vocabularies and generate sparse and dense features together with a label.

* WIDE方面
  * The wide component consists of the `cross-product transformation` of user installed apps and impression apps.
  * 然后生成相应的embedding vectors. 
* DEEP方面
  * For the deep part of the model, A 32-dimensional embedding vector is learned for each categorical feature.
* JOINT TRAINING方面
  * Concatenate all the embeddings together with the dense features, resulting in a dense vector of approximately 1200 dimensions.

#### Training

The concatenated vector is then fed into 3 ReLU layers, and finally the logistic output unit.

（为了解决每次加载旧model的耗时问题，implemented a warm-starting system which initializes a new model with the embeddings and the linear model weights from the previous model.）

### Model Serving

In order to serve each request on the order of 10 ms, we optimized the performance using multithreading parallelism by running smaller batches in parallel, instead of scoring all candidate apps in a single batch inference step.

## EXPERIMENT RESULTS

`AUC`: 指ROC曲线的面积。AUC越大，则模型预测越准确。

WIDE：0

DEEP：+2.9%

W & D：+3.9%

While Wide & Deep has a slightly higher offline AUC, the impact is more significant on online traffic. One possible reason is that the impressions and labels in offline data sets are fixed, whereas the online system can generate new exploratory recommendations by blending generalization with memorization, and learn from new user responses.

服务延迟

200*1 31

100*2 17

50*4 14

## RELATED WORK

1. 在factorization machines 基础上得来（添加了神经网络，而非单纯的内积）。
2. Collaborative deep learning 融合了（对content information 的深度学习）和（对评分矩阵的协同过滤），是content-based 的。而W & D 是jointly 地学习user and impression data。
3. 再次强调，协同过滤是基于内容的。