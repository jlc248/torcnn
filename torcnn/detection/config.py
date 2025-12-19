def config():

    c = {}
    c['outdir'] = '/raid/jcintineo/torcnn/detection/tests/test04'
    c['channels'] = ['Reflectivity', 'Velocity', 'RhoHV']
    c['PS'] = (512, 512)
    c['batchsize'] = 32
    c['bsinfo'] = {
        'Reflectivity':{'min':0., 'max':75.0},
        'Velocity':{'min':-80, 'max':80.0},
        'SpectrumWidth':{'min':0, 'max':70.0},
        'RhoHV':{'min':0.4, 'max':1.0},
        'Zdr': {'min':-4., 'max':6.},
    }
    c['obj_weight'] = 5.0
    c['noobj_weight'] = 0.5
    c['tvs_importance'] = 5.0
    c['lr'] = 1e-4
    c['rlr_cooldown'] = 2
    c['rlr_patience'] = 1
    c['rlr_factor'] = 0.1
    c['es_patience'] = 3
    c['monitoring_thresholds'] = [0.1, 0.3, 0.5, 0.7]   
 
    tfrec_dir = "/raid/jcintineo/torcnn/detection/tfrecs"
    # N.B. Can't have any "//" in the train_list or val_list!!!
    c['train_list'] = [f"{tfrec_dir}/2024/2024??[0,1]?/*tfrec"] #, f"{tfrec_dir}/2024/2024??[2,3]?//*tfrec"] 
    c['val_list'] = [f"{tfrec_dir}/2024/2024??[2,3]?//*tfrec"]
       
    return c
