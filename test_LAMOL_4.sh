#!/bin/bash

# (i) yelp agnews dbpedia amazon yahoo - DONE
# (ii) dbpedia yahoo agnews amazon yelp - WRONG
# (iii) yelp yahoo amazon dbpedia agnews - DONE
# (iv) agnews yelp amazon yahoo dbpedia - DONE
# Test template 
# python maml_pytorch_test_v2-MAML.py --is_lamol --tasks ag yelp amazon yahoo dbpedia --model_dir_name 20211023T083631_agyelamayahdbp_LAMOL_MAML

# (i)
# python maml_pytorch_test_v2-SEQ.py --is_lamol --tasks yelp10k ag10k dbpedia10k amazon10k yahoo10k --train_batch_size 4 --test_batch_size 4 --min_batch_size 4 --gen_lm_sample_percentage 0.2 --model_dir_name 20220116T190836_yelag1dbpamayah_LAMOL
# (ii)
# (ii) RE RUN -- RUN TEST LATER
# python maml_pytorch_clean-LAMOL.py --tasks dbpedia10k yahoo10k ag10k amazon10k yelp10k --gen_lm_sample_percentage 0.2 --train_batch_size 4 --test_batch_size 4 --min_batch_size 4
#python maml_pytorch_test_v2-SEQ.py --is_lamol --tasks  dbpedia10k yahoo10k ag10k amazon10k yelp10k --gen_lm_sample_percentage 0.2 --train_batch_size 4 --test_batch_size 4 --min_batch_size 4 --model_dir_name 20220118T195023_dbpyahag1amayel_LAMOL
# (iii)
# python maml_pytorch_test_v2-SEQ.py --is_lamol --tasks  yelp10k yahoo10k amazon10k dbpedia10k ag10k --gen_lm_sample_percentage 0.2 --train_batch_size 4 --test_batch_size 4 --min_batch_size 4 --model_dir_name 20220117T001739_yelyahamadbpag1_LAMOL
# (iv)
# python maml_pytorch_test_v2-SEQ.py --is_lamol --tasks ag10k yelp10k amazon10k yahoo10k dbpedia10k --gen_lm_sample_percentage 0.2 --train_batch_size 4 --test_batch_size 4 --min_batch_size 4 --model_dir_name 20220117T025517_ag1yelamayahdbp_LAMOL

# Current Rerun
# (i)
# python3.8 maml_pytorch_test_v2-SEQ.py --is_lamol --tasks yelp ag dbpedia amazon yahoo --train_batch_size 4 --test_batch_size 4 --min_batch_size 4 --gen_lm_sample_percentage 0.2 --model_dir_name 20230420T080430_yelagdbpamayah_LAMOL
# (ii)
# python3.8 maml_pytorch_test_v2-SEQ.py --is_lamol --tasks  dbpedia yahoo ag amazon yelp --gen_lm_sample_percentage 0.2 --train_batch_size 4 --test_batch_size 4 --min_batch_size 4 --model_dir_name 20230426T081703_dbpyahagamayel_LAMOL
# (iii)
# python3.8 maml_pytorch_test_v2-SEQ.py --is_lamol --tasks  yelp yahoo amazon dbpedia ag --gen_lm_sample_percentage 0.2 --train_batch_size 4 --test_batch_size 4 --min_batch_size 4 --model_dir_name 20230427T183953_yelyahamadbpag_LAMOL
# (iv)
# python3.8 maml_pytorch_test_v2-SEQ.py --is_lamol --tasks ag yelp amazon yahoo dbpedia --gen_lm_sample_percentage 0.2 --train_batch_size 4 --test_batch_size 4 --min_batch_size 4 --model_dir_name 20230429T101645_agyelamayahdbp_LAMOL


# [TIME] Start Test 20230420T080430_yelagdbpamayah_LAMOL at 2023-05-02T08:30:27
# /data/model_runs/20230420T080430_yelagdbpamayah_LAMOL/yelp.model
# start to test { task: yelp (load) yelp (eval)}
# len of test dataset: 7600
#  87%|███████████████████████████████████████████████████        | 1644/1900 [02:51<00:26,  9.59it/s]
# [TIME] End Test 20230420T080430_yelagdbpamayah_LAMOL at 2023-05-02T10:16:19 within 1.7643845056162941 hours

# Real 0.05
# (i)
# python3.8 maml_pytorch_test_v2-SEQ.py --is_lamol --tasks yelp ag dbpedia amazon yahoo --train_batch_size 4 --test_batch_size 4 --min_batch_size 4 --gen_lm_sample_percentage 0.05 --real_sample --model_dir_name 20230606T101615_yelagdbpamayah_LAMOL
# (ii)
python3.8 maml_pytorch_test_v2-SEQ.py --is_lamol --tasks dbpedia yahoo ag amazon yelp --train_batch_size 4 --test_batch_size 4 --min_batch_size 4 --gen_lm_sample_percentage 0.05 --real_sample --model_dir_name 20230607T110925_dbpyahagamayel_LAMOL
# (iii)
python3.8 maml_pytorch_test_v2-SEQ.py --is_lamol --tasks yelp yahoo amazon dbpedia ag --train_batch_size 4 --test_batch_size 4 --min_batch_size 4 --gen_lm_sample_percentage 0.05 --real_sample --model_dir_name 20230608T003614_yelyahamadbpag_LAMOL
# (iv)
python3.8 maml_pytorch_test_v2-SEQ.py --is_lamol --tasks ag yelp amazon yahoo dbpedia --train_batch_size 4 --test_batch_size 4 --min_batch_size 4 --gen_lm_sample_percentage 0.05 --real_sample --model_dir_name 20230608T140603_agyelamayahdbp_LAMOL