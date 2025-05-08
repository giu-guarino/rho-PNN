config = {
    # Training Configuration
    'validation': True,

    # Satellite configuration
    'satellite': 'PRISMA',
    'ratio': 6,
    'nbits': 16,

    # Training settings
    'save_weights': True,
    'save_weights_path': 'weights',
    'save_training_stats': False,
    'compute_quality_indexes': False,

    # Training hyperparameters
    'learning_rate': 0.00001,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epochs': 200,
    'batch_size': 4,
    'semi_width': 18,
    'kernel_size': 61,

    'alpha_1': 0.5,
    'alpha_2': 0.25,

    'first_iter': 20,
    'epoch_nm': 15,
    'sat_val': 80,

    'net_scope': 6,
    'ms_scope': 1,

    'hp_fr': [0.65, 0.90, 2, 1.007, 20, 30],
    'hp_rr': [0.62, 0.90, 1, 1.003, 5, 30],

    'num_models': 10, # Number of trained models for model ensemble
    'me_epochs': 10, # Number of epochs for model ensemble

    'num_salient_patches': 4, # Number of salient pathces
    'patch_dim': 300, # Salien patches dimension
    'patch_dim_lr': 50, # Salien low resolution patches dimension

}