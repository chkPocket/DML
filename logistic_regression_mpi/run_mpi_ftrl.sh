#!/bin/bash

#tt=`date`
#mkdir backup/"$tt"
#mv train backup/"$tt"
#mv *.log backup/"$tt"
#mv core backup/"$tt"
#make
#rm log/*
scp train worker@10.101.2.89:/home/worker/xiaoshu/DML/logistic_regression_mpi/.
scp train worker@10.101.2.90:/home/worker/xiaoshu/DML/logistic_regression_mpi/.
mpirun -f ../hosts -np 6 ./train ftrl ./data/agaricus.txt.train ./data/agaricus.txt.test
