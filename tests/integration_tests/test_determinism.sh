#!/bin/bash

set -u

# set the chunks to small values to enforce data is read from disk in multiple chunks
export MOKAPOT_CONFIDENCE_CHUNK_SIZE=101
export MOKAPOT_CHUNK_SIZE_READ_ALL_DATA=103
export MOKAPOT_CHUNK_SIZE_ROWS_PREDICTION=107
export MOKAPOT_CHUNK_SIZE_COLUMNS_FOR_DROP_COLUMNS=2
export MOKAPOT_CHUNK_SIZE_ROWS_FOR_DROP_COLUMNS=109

# clear leftover files
rm -rf run1/ run2/ run3/ run4/
# two runs with one file
python -m mokapot.mokapot test1.tab --subset_max_train 2500 --keep_decoys -d run1/ --max_workers 8 -v 2 --ensemble --max_iter 2
python -m mokapot.mokapot test1.tab --subset_max_train 2500 --keep_decoys -d run2/ --max_workers 8 -v 2 --ensemble --max_iter 2
# two runs with anothe file where only the SpecIds are changed
python -m mokapot.mokapot test2.tab --subset_max_train 2500 --keep_decoys -d run3/ --max_workers 8 -v 2 --ensemble --max_iter 2
python -m mokapot.mokapot test2.tab --subset_max_train 2500 --keep_decoys -d run4/ --max_workers 8 -v 2 --ensemble --max_iter 2


for file in decoys.psms targets.psms decoys.peptides targets.peptides
do
  PREFIX=without-first-column-
  NEW_FILE=$PREFIX$file
  # cut uses tabs as default
  cut -f 2- run1/$file > run1/$NEW_FILE
  cut -f 2- run2/$file > run2/$NEW_FILE
  cut -f 2- run3/$file > run3/$NEW_FILE
  cut -f 2- run4/$file > run4/$NEW_FILE

  echo -n run1/$NEW_FILE "==" run2/$NEW_FILE
  if [[ $(diff run1/$NEW_FILE run2/$NEW_FILE | wc -l) != "0" ]]
  then
    echo " failed"
    exit 1
  else
    echo " ok"
  fi

  echo -n run1/$NEW_FILE run3/$NEW_FILE
  if [[ $(diff run1/$NEW_FILE run3/$NEW_FILE | wc -l) != "0" ]]
  then
    echo " failed"
    exit 1
  else
    echo " ok"
  fi

  echo -n run1/$NEW_FILE run4/$NEW_FILE
  if [[ $(diff run1/$NEW_FILE run4/$NEW_FILE | wc -l) != "0" ]]
  then
    echo " failed"
    exit 1
  else
    echo " ok"
  fi
done