#!/bin/bash

# (i)
# python maml_pytorch_clean_v2-LAMOL_MAML.py --tasks yelp ag dbpedia amazon yahoo --real_sample

# tasks=( "sst" "srl" "woz.en" )

# tasks_ignore=( "movie boolq scifact" 
#                 "movie scifact boolq" )
tasks_ignore=( )

containsElement () {
  local e match="$1"
  shift
  for e; do [[ "$e" == "$match" ]] && return 0; done
  return 1
}


for i in "${tasks[@]}"
do
    for j in "${tasks[@]}"
    do
        # Ignore if i is duplicate of j
        if [ $i == $j ]; then
            continue 
        fi
        for k in "${tasks[@]}"
        do
            # Ignore if k is duplicate of j and i
            if [ $k == $j ] || [ $k == $i ]; then
                continue 
            fi
            # Ignore if i j k is in tasks_ignore
            # https://stackoverflow.com/questions/3685970/check-if-a-bash-array-contains-a-value
            if containsElement "$i $j $k" "${tasks_ignore[@]}" ]]; then
                continue 
            fi
            echo python maml_pytorch_clean-LAMOL.py --tasks $i $j $k
            python maml_pytorch_clean-LAMOL.py --tasks $i $j $k
            
#             echo python maml_pytorch_clean_v2-LAMOL_MAML.py --tasks $i $j $k
#             python maml_pytorch_clean_v2-LAMOL_MAML.py --tasks $i $j $k
            
#             echo python maml_pytorch_clean_v2-LAMOL_MAML.py --tasks $i $j $k
#             python maml_pytorch_clean_v2-LAMOL_MAML.py --tasks $i $j $k
        done
    done
done