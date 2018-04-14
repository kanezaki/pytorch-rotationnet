#!/bin/bash

dataset=$1
output=$2

mkdir -p $output/{train,test}

for cls in `ls $dataset`
do
    for subset in train test
    do
	mkdir -p $output/$subset/$cls
	cd $output/$subset/$cls
	
	for f in `ls ../../../$dataset/$cls/$subset/`
	do
	    ln -s ../../../$dataset/$cls/$subset/$f .
	done
	cd ../../..
    done
done

cd $output/
ln -s test val
cd ..
