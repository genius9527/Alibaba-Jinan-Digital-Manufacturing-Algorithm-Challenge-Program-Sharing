# Alibaba-Jinan-Digital-Manufacturing-Algorithm-Challenge-Program-Sharing
复赛第26名（26/2682）


**赛题背景**：
2019津南数字制造算法挑战赛，聚焦智能制造，赛场一以原料企业工艺优化为课题，要求选手以异烟酸生产过程中的各参数，设计精确智能的优秀算法，提升异烟酸的收率，助力企业实现转型升级，提升行业竞争力。大赛数据提供方天津汉德威药业有限公司，为大赛提供真实生产数据，提供工艺专家的专业指导，从软硬件环境诸多方面提供大赛支撑。

**赛题数据**：

https://tianchi.aliyun.com/competition/entrance/231695/information

**初赛**：

大赛要求选手通过使用生产过程的各参数，训练算法来预测最终异烟酸的收率，同时，选手还需给出生产过程的最优参数组合及最优参数情况下的收率。

初赛提交文件：每批次异烟酸预测收率，选手提交csv格式。该文件由两列组成，第一列为异烟酸批次id，由赛题方提供；第二列为预测的异烟酸预测收率，以小数形式表示，建议保留小数点后三位。

![初赛评估指标](https://github.com/genius9527/Alibaba-Jinan-Digital-Manufacturing-Algorithm-Challenge-Program-Sharing/blob/master/%E6%B4%A5%E5%8D%97%E5%A4%A7%E6%95%B0%E6%8D%AE%E5%88%9D%E8%B5%9B%E8%AF%84%E4%BC%B0%E6%8C%87%E6%A0%87.png
)

数据：

大赛包含有2000批次来自实际异烟酸生产中的各参数的监测指标和最终收率的数据。监测指标由两大工序数十个步骤构成。总生产步骤达30余项。我们将工序和步骤分别用字母和数字代号表示，比如A2，B5分别表示A工序第二步骤和B工序第五步骤。样例数据参考训练数据。

特征工程及模型：

初赛使用的模型是xgb，做了两个比赛，发现机器学习比赛的要义是造特征，特征工程决定算法的上限，所以初赛基本对模型没有太多调参，只是调了下随机种子。

因为本赛题数据量很小，而且是一个回归问题，容易形成误差积累，所以在特征设计上较为精简。主要设计了两类特征，类别特征和数字特征，另外还对类别特征作了One Hot处理，此外做了数据清洗，主要是对异常值进行了处理，填充了缺失值以及对重复率在0.9以上的列进行了丢弃。缺失值填充的方式试过最小值填充，和均值填充，以及邻近数据填充，但实验下来发现众数填充的效果最好，这个也可能与数据比较稀疏有关。

还有一个骚操作，这个是参考鱼佬的思路，对类别特征进行了一个分箱操作，效果还不错，整个的意思还是因为类别特征很稀疏，但在不考虑 ID的前提下，会有很多特征完全相同，但是Y值差异很大的数据：

![分箱操作](https://github.com/genius9527/Alibaba-Jinan-Digital-Manufacturing-Algorithm-Challenge-Program-Sharing/blob/master/%E5%88%86%E7%AE%B1%E6%93%8D%E4%BD%9C.png)

初赛有个bug，样本id竟然是个强特，照往常这种序号类的维度是可以丢弃的，但在初赛中很奇怪，可能样本id反应了生产批次，药品生产中，药品批次会涵盖很多其他维度的影响因素，所以对收率也会影响较大。

最后初赛B榜与c榜的综合成绩是47名：

![初赛成绩](https://github.com/genius9527/Alibaba-Jinan-Digital-Manufacturing-Algorithm-Challenge-Program-Sharing/blob/master/%E5%88%9D%E8%B5%9B%E6%88%90%E7%BB%A9.png)

**复赛**：

![复赛评估指标](https://github.com/genius9527/Alibaba-Jinan-Digital-Manufacturing-Algorithm-Challenge-Program-Sharing/blob/master/%E5%A4%8D%E8%B5%9B%E8%AF%84%E4%BC%B0%E6%8C%87%E6%A0%87.png)

数据：
（1）复赛新数据集去除了潜在的异常值，收率范围在0.85-1之间（不一定包含两端点）。
（2）复赛新数据B14列的取值范围在350-460之间（不一定包含两端点）。
（3）复赛样本id重新编码，随机生成id，且id值在10000以上，例如sample_10001
复赛评测：
考虑到工厂采集数据存在输入误差的可能性，我们将把大家预测结果与答案之间误差（误差定义为所有提交选手该条记录标准差-该条记录答案）最大的3条样本作为误差数据删除，剩余的数据再统一做评测。

模型：

复赛的最优收率值，我们的算法只有0.98615，但这个没有太多去调，因为只占10%的成绩，而且后来发现很多调到很接近1的大佬，测评时都超过1了，结果这一项就没有成绩了。

复赛的优化主要是模型调参后进行了融合，使用了lgb和xgb做了融合。因为复赛打乱了样本id，并进行了重新编码，样本id相关的特征已经没有用了，就都删掉了。此外进行了特征选择，对重要性不高，去掉后线下效果变好的特征进行了删除。另外也增加了一些新的特征，主要还是加、减、乘、除来构造新特征，虽然比较low，但是还是有点效果。

![复赛算法架构图](https://github.com/genius9527/Alibaba-Jinan-Digital-Manufacturing-Algorithm-Challenge-Program-Sharing/blob/master/%E6%B4%A5%E5%8D%97%E7%AE%97%E6%B3%95%E6%8C%91%E6%88%98%E8%B5%9B%E7%AE%97%E6%B3%95%E6%9E%B6%E6%9E%84%E5%9B%BE.png)



最后，取得了第26名的成绩，小白第二次做天池，很满意了，还是要多向大佬们学习

![复赛评估指标](https://github.com/genius9527/Alibaba-Jinan-Digital-Manufacturing-Algorithm-Challenge-Program-Sharing/blob/master/%E5%A4%8D%E8%B5%9B%E6%88%90%E7%BB%A9.png)

上传的是复赛的代码，数据可以自行去下载，放到/data 下面就可以，该程序运行时间为196s左右。代码只用执行一次，便可同时得到submit_optimize.csv和submit_FuSai.csv。

