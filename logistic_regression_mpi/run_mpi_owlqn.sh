#!/bin/bash

#tt=`date`
#mkdir backup/"$tt"
#mv train backup/"$tt"
#mv *.log backup/"$tt"
#mv core backup/"$tt"
#make
#rm log/*
mpirun -np 4 ./train owlqn ./data/agaricus.txt.train ./data/agaricus.txt.test
