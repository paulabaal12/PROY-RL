| algo | config | overrides | validation_return | chosen |
|---|---|---|---|---|
| dqn | 0 | base | 29.29 | False |
| dqn | 1 | {'learning_rate': 0.0003} | -30.04 | False |
| dqn | 2 | {'target_update_interval': 1000, 'train_freq': 2} | 18.79 | False |
| dqn | 3 | {'exploration_steps': 120000} | 43.66 | True |
| ppo | 0 | base | -56.3 | False |
| ppo | 1 | {'learning_rate': 0.0001} | -152.77 | False |
| ppo | 2 | {'ent_coef': 0.08} | -77.34 | False |
| ppo | 3 | {'n_epochs': 10} | -13.78 | True |
