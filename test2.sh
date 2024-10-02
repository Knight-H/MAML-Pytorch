#!/bin/bash

# (i) yelp agnews dbpedia amazon yahoo - DONE
# (ii) dbpedia yahoo agnews amazon yelp - DONE
# (iii) yelp yahoo amazon dbpedia agnews - DONE
# (iv) agnews yelp amazon yahoo dbpedia - DONE
# Test template 
# python maml_pytorch_test_v2-MAML.py --is_lamol --tasks ag yelp amazon yahoo dbpedia --model_dir_name 20211023T083631_agyelamayahdbp_LAMOL_MAML

# (i)
python maml_pytorch_test_v2-MAML.py --is_lamol --tasks yelp10k ag10k dbpedia10k amazon10k yahoo10k --train_batch_size 4 --test_batch_size 4 --min_batch_size 4 --gen_lm_sample_percentage 0.2 --model_dir_name 20220102T184856_yelag1dbpamayah_LAMOL_MAML
# (ii)
python maml_pytorch_test_v2-MAML.py --is_lamol --tasks  dbpedia10k yahoo10k ag10k amazon10k yelp10k --gen_lm_sample_percentage 0.2 --train_batch_size 4 --test_batch_size 4 --min_batch_size 4 --model_dir_name 20220103T033213_dbpyahag1amayel_LAMOL_MAML
# (iii)
python maml_pytorch_test_v2-MAML.py --is_lamol --tasks  yelp10k yahoo10k amazon10k dbpedia10k ag10k --gen_lm_sample_percentage 0.2 --train_batch_size 4 --test_batch_size 4 --min_batch_size 4 --model_dir_name 20220103T195845_yelyahamadbpag1_LAMOL_MAML
# (iv)
python maml_pytorch_test_v2-MAML.py --is_lamol --tasks ag10k yelp10k amazon10k yahoo10k dbpedia10k --gen_lm_sample_percentage 0.2 --train_batch_size 4 --test_batch_size 4 --min_batch_size 4 --model_dir_name 20211231T135835_ag1yelamayahdbp_LAMOL_MAML