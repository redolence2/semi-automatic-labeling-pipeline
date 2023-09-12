#!/bin/bash

NUM_OF_WORKER=1

for ((i=0; i<${NUM_OF_WORKER}; i++))
do

bash ./run_post_process_to_labelme.sh ${i} ${NUM_OF_WORKER} &

done