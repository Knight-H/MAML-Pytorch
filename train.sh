#!/bin/bash

# (i) yelp agnews dbpedia amazon yahoo - DONE
# (ii) dbpedia yahoo agnews amazon yelp - DONE
# (iii) yelp yahoo amazon dbpedia agnews - DONE
# (iv) agnews yelp amazon yahoo dbpedia - DONE

# (i)
# python maml_pytorch_clean_v2-LAMOL_MAML.py --tasks yelp ag dbpedia amazon yahoo --real_sample
# python maml_pytorch_clean_v2-LAMOL_MAML.py --tasks yelp10k ag10k dbpedia10k amazon10k yahoo10k --gen_lm_sample_percentage 0.2 --train_batch_size 4 --test_batch_size 4 --min_batch_size 4
# (ii)
# python maml_pytorch_clean_v2-LAMOL_MAML.py --tasks dbpedia10k yahoo10k ag10k amazon10k yelp10k --gen_lm_sample_percentage 0.2 --train_batch_size 4 --test_batch_size 4 --min_batch_size 4
# (iii)
# python maml_pytorch_clean_v2-LAMOL_MAML.py --tasks yelp10k yahoo10k amazon10k dbpedia10k ag10k --gen_lm_sample_percentage 0.2 --train_batch_size 4 --test_batch_size 4 --min_batch_size 4
# (iv)
# python maml_pytorch_clean_v2-LAMOL_MAML.py --tasks ag yelp amazon yahoo dbpedia  [NOT REAL SAMPLE CUZ IT BUG]
# python maml_pytorch_clean_v2-LAMOL_MAML.py --tasks ag10k yelp10k amazon10k yahoo10k dbpedia10k --gen_lm_sample_percentage 0.2 --train_batch_size 16 --test_batch_size 16 --min_batch_size 16
## FINAL - DONE
# python maml_pytorch_clean_v2-LAMOL_MAML.py --tasks ag10k yelp10k amazon10k yahoo10k dbpedia10k --gen_lm_sample_percentage 0.2 --train_batch_size 4 --test_batch_size 4 --min_batch_size 4

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