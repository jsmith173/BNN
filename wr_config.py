import json

config = json.loads('{}')

config['fig_N'] = 3
config['imgW'] = 28
config['conversion_ctl'] = 0
config['draw_original'] = 0
config['draw_aug'] = 0
config['gen_deterministic_model'] = 0
config['eval_deterministic_model'] = 0
config['gen_prob_model'] = 0
config['eval_prob_model'] = 0

with open('config.json', 'w') as f:
 json.dump(config, f)

