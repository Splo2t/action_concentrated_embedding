# Action-Concentrated Embedding Framework: This is your captain sign-tokening
## Abstarct
Sign language is the primary communication medium for deaf and hard-of-hearing individuals. With the divergent sensory channels for communication between them, a noticeable communication gap exists. It's crucial to address these disparities and bridge the existing communication gap. In this paper, we present an action-concentrated embedding (ACE), a novel sign token embedding framework. Additionally, in an effort to provide a more structured foundation for sign language analysis, we've introduced a dedicated notation system tailored for sign languages. This system endeavors to encapsulate the nuanced gestures and movements integral to sign communication. The proposed ACE approach tracks a signer’s actions based on human posture estimation. Tokenizing these actions and capturing the token embedding using the short-time Fourier transform encapsulates time-based behavioral changes. Thus, ACE offers input embedding to translate sign language into natural language sentences. When tested against a disaster sign language dataset using automated machine translation measures, ACE notably surpasses prior research in its translation capabilities, improving performance by up to 2.91\% for BLEU-4 and 3.84\% for ROUGE-L metric.
### Datasets
+ [https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=636] (Sign language for diaster information)

|  Lines | Videos | 
| :---:| :---: |
|  164,375 | 201,026 |


## Work Steps

### 1. Transforming the Frame to Angular, Tokenizing and Embedding
```
python dataGenerator_ours.py --data_path="./" --output_path="./slt_stft_data" --fs=30 --nfft=40 --overlap=30
```
Then, ~~~~

### 2. Traning
```
torchrun --master_port 25001 --nproc_per_node 1 --nnodes 1 my_slt_train.py -type 8888 -lr 0.001
```

## Requirements
install requirements.txt
