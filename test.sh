#!/bin/bash

# tasks=( "movie" "boolq" "scifact" )
# tasks=( "sst" "srl" "woz.en" )
tasks=( "ag" "yelp" "amazon" "yahoo" "dbpedia" )
# python maml_pytorch_test_v2-MAML.py --is_lamol --tasks ag yelp amazon yahoo dbpedia --model_dir_name 20211023T083631_agyelamayahdbp_LAMOL_MAML

tasks_ignore=( )
# model_dirs=( "20210912T154326_mbs_LAMOL" 
#              "20210912T123808_msb_LAMOL" 
#              "20210912T165256_bms_LAMOL"
#              "20210912T173404_bsm_LAMOL" 
#              "20210912T181303_smb_LAMOL"
#              "20210912T190652_sbm_LAMOL")
# model_dirs=( "20210918T092350_mbs_LAMOL_MAML" 
#              "20210918T114204_msb_LAMOL_MAML" 
#              "20210918T135020_bms_LAMOL_MAML"
#              "20210918T155817_bsm_LAMOL_MAML" 
#              "20210918T180433_smb_LAMOL_MAML"
#              "20210918T202148_sbm_LAMOL_MAML")
# model_dirs=( "20210925T200907_sstsrlwoz_LAMOL_MAML" 
#              "20210925T210317_sstwozsrl_LAMOL_MAML" 
#              "20210925T215826_srlsstwoz_LAMOL_MAML"
#              "20210925T225422_srlwozsst_LAMOL_MAML" 
#              "20210925T234850_wozsstsrl_LAMOL_MAML"
#              "20210926T004459_wozsrlsst_LAMOL_MAML")
# model_dirs=( "20210929T195014_sstsrlwoz_LAMOL" 
#              "20210929T200743_sstwozsrl_LAMOL" 
#              "20210929T202506_srlsstwoz_LAMOL"
#              "20210929T204307_srlwozsst_LAMOL" 
#              "20210929T210026_wozsstsrl_LAMOL"
#              "20210929T211758_wozsrlsst_LAMOL")

count=0

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
#             echo python maml_pytorch_test_v2-SEQ.py --tasks $i $j $k --model_dir_name ${model_dirs[$count]}
#             python maml_pytorch_test_v2-SEQ.py --tasks $i $j $k --model_dir_name ${model_dirs[$count]}
            echo python maml_pytorch_test_v2-SEQ.py --tasks $i $j $k --model_dir_name ${model_dirs[$count]}
            python maml_pytorch_test_v2-SEQ.py --tasks $i $j $k --model_dir_name ${model_dirs[$count]}
#             echo python maml_pytorch_test_v2-MAML.py --is_lamol --tasks $i $j $k --test_batch_size=6 --model_dir_name ${model_dirs[$count]}
#             python maml_pytorch_test_v2-MAML.py --is_lamol --tasks $i $j $k --test_batch_size=6 --model_dir_name ${model_dirs[$count]}
            ((count=count+1))
        done
    done
done