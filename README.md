# DCMH_Flickr25k
A pytorch implement for DCMH (Deep Cross-Modal retrieval, cvpr 2017) on Flickr25k dataset.

### Dataset
  * MirFlickr25k<br>
    M. J. Huiskes, M. S. Lew (2008). [The MIR Flickr Retrieval Evaluation](The MIR Flickr Retrieval Evaluation).<br>
    ACM International Conference on Multimedia Information Retrieval (MIR'08), Vancouver, Canada

### Requirements
  * Python 3.X<br>
  * Pytorch 0.4

### Model
  * Pretrained CNN-F
  
### Instruction
  This implement add an extra negative log likelihood loss of intra-modal, so the final MAP is higher than the original paper.
  
### Result
 ![image](https://github.com/La-ji/DCMH_Flickr25k/blob/master/result/result.png)
  
### Thanks for:
  * [jiangqy/DPSH](https://github.com/jiangqy/DPSH-pytorch)<br>
  * [WendellGul/DCMH](https://github.com/WendellGul/DCMH)
 
