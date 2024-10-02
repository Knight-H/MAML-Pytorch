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


# ~~~~~~~~~~~~~~ Normal LAMOL 10k ~~~~~~~~~~~~~~~~~~~~~~
# (i)
# python maml_pytorch_clean-LAMOL.py --tasks yelp10k ag10k dbpedia10k amazon10k yahoo10k --gen_lm_sample_percentage 0.2 --train_batch_size 4 --test_batch_size 4 --min_batch_size 4
# (ii)
# python maml_pytorch_clean-LAMOL.py --tasks dbpedia10k yahoo10k ag10k amazon10k yelp10k --gen_lm_sample_percentage 0.2 --train_batch_size 4 --test_batch_size 4 --min_batch_size 4
# (iii)
# python maml_pytorch_clean-LAMOL.py --tasks yelp10k yahoo10k amazon10k dbpedia10k ag10k --gen_lm_sample_percentage 0.2 --train_batch_size 4 --test_batch_size 4 --min_batch_size 4
# (iv)
# python maml_pytorch_clean-LAMOL.py --tasks ag10k yelp10k amazon10k yahoo10k dbpedia10k --gen_lm_sample_percentage 0.2 --train_batch_size 4 --test_batch_size 4 --min_batch_size 4


# ~~~~~~~~~~~~~ON FINALE TESTO (maybe need 0.05 real with 0.2 pseudo too???)~~~~~~~~~~~~~
# train_batch_size 8 still errors!! 
# (i) for full!!
# python3.8 maml_pytorch_clean-LAMOL.py --tasks yelp ag dbpedia amazon yahoo --gen_lm_sample_percentage 0.2 --min_batch_size 4 --train_batch_size 4 --test_batch_size 4  
# (ii)
# python maml_pytorch_clean-LAMOL.py --tasks dbpedia yahoo ag amazon yelp --gen_lm_sample_percentage 0.2 --train_batch_size 4 --test_batch_size 4 --min_batch_size 4
# (iii)
# python maml_pytorch_clean-LAMOL.py --tasks yelp yahoo amazon dbpedia ag --gen_lm_sample_percentage 0.2 --train_batch_size 4 --test_batch_size 4 --min_batch_size 4
# (iv)
# python maml_pytorch_clean-LAMOL.py --tasks ag yelp amazon yahoo dbpedia --gen_lm_sample_percentage 0.2 --train_batch_size 4 --test_batch_size 4 --min_batch_size 4

# 20230420T080430_yelagdbpamayah_LAMOL yahoo Done Saving Model at /data/model_runs/20230420T080430_yelagdbpamayah_LAMOL/yahoo.model
# [TIME] End Run 20230420T080430_yelagdbpamayah_LAMOL at 2023-04-21T19:38:32 within 35.56720151411162 hours


# Reference Run For (ii)
# Starting task dbpedia
#   100%|███████████████████████████████████████████████████████| 28750/28750 [2:40:32<00:00,  2.98it/s]
# Starting task yahoo
# Generating extra data! With gen_size 23000
# writing extra data in /data/model_runs/20230426T081703_dbpyahagamayel_LAMOL/lm-yahoo-dbpedia.csv ...
#   100%|███████████████████████████████████████████████████████| 34487/34487 [3:51:34<00:00,  2.48it/s]
# 20230426T081703_dbpyahagamayel_LAMOL yahoo Done Saving Model at /data/model_runs/20230426T081703_dbpyahagamayel_LAMOL/yahoo.model
# Starting task ag
# Generating extra data! With gen_size 23000
# writing extra data in /data/model_runs/20230426T081703_dbpyahagamayel_LAMOL/lm-ag-yahoo.csv ...
#   100%|█████████████████████████████████████████████████████████| 34479/34479 [2:56:39<00:00,  3.25it/s]
# 20230426T081703_dbpyahagamayel_LAMOL ag Done Saving Model at /data/model_runs/20230426T081703_dbpyahagamayel_LAMOL/ag.model
# Starting task amazon
# Generating extra data! With gen_size 22998
# writing extra data in /data/model_runs/20230426T081703_dbpyahagamayel_LAMOL/lm-amazon-ag.csv ...
#   100%|█████████████████████████████████████████████████████████| 34461/34461 [3:17:34<00:00,  2.91it/s]
# 20230426T081703_dbpyahagamayel_LAMOL amazon Done Saving Model at /data/model_runs/20230426T081703_dbpyahagamayel_LAMOL/amazon.model
# Starting task yelp
# Generating extra data! With gen_size 23000
# writing extra data in /data/model_runs/20230426T081703_dbpyahagamayel_LAMOL/lm-yelp-amazon.csv ...
#   100%|█████████████████████████████████████████████████████████| 34479/34479 [4:03:48<00:00,  2.36it/s]
# 20230426T081703_dbpyahagamayel_LAMOL yelp Done Saving Model at /data/model_runs/20230426T081703_dbpyahagamayel_LAMOL/yelp.model
# [TIME] End Run 20230426T081703_dbpyahagamayel_LAMOL at 2023-04-27T16:35:16 within 32.30351017395655 hours

# [TIME] End Run 20230429T101645_agyelamayahdbp_LAMOL at 2023-04-30T12:58:42 within 26.699177251259485 hours


#~~~~~~~~~~~ ON FINALE FINALE - For 0.05 Real~~~~~~~~~~~~~~~~~~~~~~~~
# (i) for full!!
# python3.8 maml_pytorch_clean-LAMOL.py --tasks yelp ag dbpedia amazon yahoo --gen_lm_sample_percentage 0.05 --min_batch_size 4 --train_batch_size 4 --test_batch_size 4  --real_sample
# (ii)
python3.8 maml_pytorch_clean-LAMOL.py --tasks dbpedia yahoo ag amazon yelp --gen_lm_sample_percentage 0.05 --min_batch_size 4 --train_batch_size 4 --test_batch_size 4  --real_sample
# (iii)
python3.8 maml_pytorch_clean-LAMOL.py --tasks yelp yahoo amazon dbpedia ag --gen_lm_sample_percentage 0.05 --min_batch_size 4 --train_batch_size 4 --test_batch_size 4  --real_sample
# (iv)
python3.8 maml_pytorch_clean-LAMOL.py --tasks ag yelp amazon yahoo dbpedia --gen_lm_sample_percentage 0.05 --min_batch_size 4 --train_batch_size 4 --test_batch_size 4  --real_sample

# Reference Run For (i)
# Starting Run with RUN_ID 20230606T101615_yelagdbpamayah_LAMOL                                          
# [TIME] Start Run 20230606T101615_yelagdbpamayah_LAMOL at 2023-06-06T10:16:15                           
# [ARGS] Namespace(data_dir='/data/lamol_data', device_ids=[0], fp32=False, gen_lm_sample_percentage=0.05, is_lamol=False, lm_lambda=0.25, max_grad_norm=1, max_len=1024, meta_lr=6.25e-05, min_batch_size=4, min_n_steps=1500, model_dir_name=None, n_gpus=1, n_train_epochs=1, num_updates=5, real_sample=True, seed=42, tasks=['yelp', 'ag', 'dbpedia', 'amazon', 'yahoo'], temperature_lm=1.0, temperature_qa=1.0, test_batch_size=4, tokens_weight=5, top_k_lm=20, top_k_qa=20, top_p_lm=0.0, top_p_qa=0.0, train_batch_size=4, update_lr=6.25e-05, use_sep=False)                                                                     
# Initializing Model...                                                                                  
# Starting task yelp
# 100%|███████████████████████████████████████████████████████| 28750/28750 [3:00:50<00:00,  2.65it/s]   
# 20230606T101615_yelagdbpamayah_LAMOL yelp Done Saving Model at /data/model_runs/20230606T101615_yelagdbpamayah_LAMOL/yelp.model
# Starting task ag                                                                                       
# using real data as extra data                                                                          
# Generating extra data! With gen_size 5750                                                              
# writing extra data in /data/model_runs/20230606T101615_yelagdbpamayah_LAMOL/real-yelp.csv ...          
# Actual extra data: 5733                                                                                
# 100%|█████████████████████████████████████████████████████████| 30184/30184 [2:18:02<00:00,  3.64it/s] 
# 20230606T101615_yelagdbpamayah_LAMOL ag Done Saving Model at /data/model_runs/20230606T101615_yelagdbpamayah_LAMOL/ag.model
# Starting task dbpedia                                                                                  
# using real data as extra data                                                                          
# Generating extra data! With gen_size 2875                                                              
# writing extra data in /data/model_runs/20230606T101615_yelagdbpamayah_LAMOL/real-ag.csv ...            
# Actual extra data: 5736                                                                                
# 100%|█████████████████████████████████████████████████████████| 30184/30184 [2:29:58<00:00,  3.35it/s]
# 20230606T101615_yelagdbpamayah_LAMOL dbpedia Done Saving Model at /data/model_runs/20230606T101615_yelagdbpamayah_LAMOL/dbpedia.model
# Starting task amazon
# using real data as extra data
# Generating extra data! With gen_size 1916
# writing extra data in /data/model_runs/20230606T101615_yelagdbpamayah_LAMOL/real-dbpedia.csv ...
# Actual extra data: 5741
# 100%|█████████████████████████████████████████████████████████| 30186/30186 [2:32:18<00:00,  3.30it/s]
# 20230606T101615_yelagdbpamayah_LAMOL amazon Done Saving Model at /data/model_runs/20230606T101615_yelagdbpamayah_LAMOL/amazon.model
# Starting task yahoo
# using real data as extra data
# Generating extra data! With gen_size 1437
# writing extra data in /data/model_runs/20230606T101615_yelagdbpamayah_LAMOL/real-amazon.csv ...
# Actual extra data: 5744
# 100%|█████████████████████████████████████████████████████████████| 30186/30186 [2:58:50<00:00,  2.81it/s]
# 20230606T101615_yelagdbpamayah_LAMOL yahoo Done Saving Model at /data/model_runs/20230606T101615_yelagdbpamayah_LAMOL/yahoo.model
# [TIME] End Run 20230606T101615_yelagdbpamayah_LAMOL at 2023-06-06T23:42:38 within 13.439479347401194 hours