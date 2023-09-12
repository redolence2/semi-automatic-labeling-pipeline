#!/bin/bash

NUM_OF_WORKER=10

for ((i=0; i<${NUM_OF_WORKER}; i++))
do

bash ./run_merge_gt.sh ${i} ${NUM_OF_WORKER} &

done
