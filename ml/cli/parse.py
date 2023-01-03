import argparse

parser = argparse.ArgumentParser(prog='Visual AI Lab', description='Train or Test a Model')

# Positional
parser.add_argument('model', type=str, help='Model to use')

# Optional
parser.add_argument('--update', action='store_true', help='Pull recent changes before running')
parser.add_argument('--save', type=int, help='Model saving interval')
parser.add_argument('--load', type=str, help='Run to load')
parser.add_argument('--device', type=str, help='cpu or cuda')
parser.add_argument('--log', action='store_true', help='Log results to Weights and Biases')
parser.add_argument('--debug', action='store_true', help='Toggle debug mode')
parser.add_argument('--guide', action='store_true', help='Toggle reward guidance')
parser.add_argument('--beam_search', action='store_true', help='Use Beam Search')
parser.add_argument('--env', type=str, help='Environment to train / test')
parser.add_argument('--tag', type=str, help='Nickname for run')
parser.add_argument('--mode', type=str, help='Training / Testing mode')
parser.add_argument('--retrain', nargs='+', default=[], help='Submodel to retrain')
parser.add_argument('--freeze', nargs='+', default=[], help='Submodels to freeze during training')
parser.add_argument('--reload', action='store_true', help='Reload dataset')
parser.add_argument('--oracle', action='store_true', help='Oracle actions')
parser.add_argument('--iterative', action='store_true', help='Perform inference during each step in evaluation')
parser.add_argument('--horizon', type=int, help='Length of predicted sequence')
parser.add_argument('--steps', type=int, help='Number of diffusion steps')
parser.add_argument('--vocab', type=int, help='Vocabulary Size')
parser.add_argument('--parallel', type=int, help='Number of parallel environments')
parser.add_argument('--guide_method', type=str, help='Reward guidance method')
parser.add_argument('--guide_steps', type=int, help='Number of reward guidance steps')
parser.add_argument('--seq_len', type=int, help='Length of sequence')
parser.add_argument('--episodes', type=int, help='Max number of episodes')
parser.add_argument('--eval_seq_len', type=int, help='Length of evaluation sequence')
parser.add_argument('--rollout', type=int, help='Number of steps between each validation')
parser.add_argument('--layers', type=int, help='Number of layers in model')
parser.add_argument('--embed_size', type=int, help='Size of VQ-VAE embedding')
parser.add_argument('--batch_size', type=int, help='Training batch size')
parser.add_argument('--learning_rate', type=float, help='Training learning rate')
parser.add_argument('--eval', action='store_true', help='Evaluate model without training')

if __name__ == "__main__":

    from ..cli import console

    args = parser.parse_args()
    console.log(args.__dict__)
