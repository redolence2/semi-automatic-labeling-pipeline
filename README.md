# semi-automatic-labeling-pipeline
![image](https://github.com/redolence2/semi-automatic-labeling-pipeline/assets/140776078/8e754c45-40fa-49db-bd7b-75eeb7a64336)

this pipeline is a semi-automatic lane labeling pipeline, model is based on hybridNets(https://github.com/datvuthanh/HybridNets)

*******************usage*******************
1. download bdd100k official dataset
2. use scripts in /preprocess to generate instance GT map
3. run train.sh in /model to train the model(stage1 to train backbone + segmentation head; stage2 to train instance head)
4. run infer.sh in /model to infer images and cluster embedding of lanes to generate instacce level lane results
5. run post_process_to_labelme.sh in /postprocess to convert infer results into json, which can be recognized by labelme


Original bdd100k dataset doesn't have instance map GT, so the script in /preprocess/ extract lanes GT from bdd100k json and merge two annotations of each lane into one, to construct instance GT map.

output instance map will be clusted and done curve fitting and converted to json and labelme will recognize this json.
This pre-annotation work save time from 2min/image to 30s/image
