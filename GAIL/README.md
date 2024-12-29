GAIL/
how to run
1. train the expert model and save it in minigrid-rl/storage
2. run python3 sb_gail.py --env < envname > --model < expert modelname in storage > --full_obs(for full obs expert)
3. with obs stack run python3 sb_gail_obstack.py --env ... --model ...
4. with no specified output modelname, it will be saved in GAIL/result/gail_trained_ppo.zip
5. run python3 evaluate.py --env .. --model < location of the trained model > --render(for vis)

when executing sb_gail_obstack.py, modified ./DIFO-on-POMDP/gailenv/lib/python3.10/site-packages/stable_baselines3/common/preprocessing.py row 10,
        def is_image_space_channels_first(observation_space: spaces.Box) -> bool:
            return True
make this function always return true
