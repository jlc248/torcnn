def config():

    c = {}
    c['outdir'] = '/raid/jcintineo/torcnn/detection/tests/test02'
    c['channels'] = ['Reflectivity', 'Velocity', 'RhoHV']
    c['PS'] = (512, 512)
    c['batchsize'] = 64
    c['bsinfo'] = {
        'Reflectivity':{'min':0., 'max':75.0},
        'Velocity':{'min':-80, 'max':80.0},
        'SpectrumWidth':{'min':0, 'max':70.0},
        'RhoHV':{'min':0.4, 'max':1.0},
        'Zdr': {'min':-4., 'max':6.},
    }
    c['obj_weight'] = 1.0  #50.0
    c['noobj_weight'] = 1.0 #0.1
    c['tvs_importance'] = 5.0
    c['lr'] = 5e-5
    c['rlr_cooldown'] = 2
    c['rlr_patience'] = 1
    c['rlr_factor'] = 0.1
    c['es_patience'] = 3
    c['monitoring_thresholds'] = [0.1, 0.3, 0.5, 0.7]   
 
    tfrec_dir = "/raid/jcintineo/torcnn/detection/tfrecs_100km60min"
    # N.B. Can't have any "//" in the train_list or val_list!!!
    c['train_list'] = [f"{tfrec_dir}/201?/20*/*tfrec", f"{tfrec_dir}/202[0-2]/20*/*tfrec"] 
    c['val_list'] = [f"{tfrec_dir}/2023/20*/*tfrec"]
       
    return c
