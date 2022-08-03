import sys
import numpy as np
import argparse
import os

from preprocess import make_env

if __name__ == "__main__":

    algorithms = {
        "DQN": "from DQN.DQNAgent import Agent",
        "DDQN": "from DDQN.DDQNAgent import Agent",
        "DuDQN": "from DuDQN.DuDQNAgent import Agent",
        "DuDDQN": "from DuDDDQN.DuDDQNAgent import Agent",
    }

    parser = argparse.ArgumentParser(description="A script to train and test DQN based algorithms for atari games.")

    parser.add_argument("--train", type=bool, default=True, help="Whether to train the agent or test the agent.")

    parser.add_argument("--gamma", type=float, default=0.99, help="The discount factor.")
    parser.add_argument("--epsilon", type=float, default=1.0, help="The initial epsilon.")
    parser.add_argument("--lr", type=float, default=0.00025, help="The learning rate.")
    parser.add_argument("--memory-size", type=int, default=30000, help="The size of the replay buffer.")
    parser.add_argument("--batch-size", type=int, default=32, help="The batch size.")
    parser.add_argument("--eps-min", type=float, default=0.01, help="The final epsilon.")
    parser.add_argument("--decay-rate", type=float, default=1e-5, help="The decay rate for epsilon.")
    parser.add_argument("--update-every", type=int, default=int(1000), help="The number of timesteps between target network updates.")
    parser.add_argument("--alg", type=str, default="DQN", help="The algorithm to train.\n1. DQN\n2. DDQN\n3. DuDQN\n4. DuDDQN")
    parser.add_argument("--env", type=str, default="PongNoFrameskip-v4", help="The environment to train on.")
    parser.add_argument("--checkpoint-dir", type=str, default="tmp/dqn", help="The directory to save checkpoints to.")
    
    parser.add_argument("--num-episodes", type=int, default=1, help="The number of episodes to run.")

    parser.add_argument("--render", type=bool, default=False, help="Whether to render the environment.")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to use (0 or 1).")
    
    parser.add_argument("--num-frames", type=int, default=4, help="The number of frames to stack.")
    parser.add_argument("--clip-reward", type=bool, default=False, help="Whether to clip the reward.")
    parser.add_argument("--fire-first", type=bool, default=False, help="Whether to fire the first action.")
    parser.add_argument("--no-ops", type=int, default=0, help="The number of no-op actions to perform.")

    args = parser.parse_args()

    exec(algorithms[args.alg])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    env = make_env(args.env, repeat=args.num_frames,
                  clip_reward=args.clip_reward, fire_first=args.fire_first,
                  no_ops=args.no_ops)
    
    best_score = -np.inf

    agent = Agent(gamma=args.gamma, epsilon=args.epsilon, lr=args.lr,
                  input_dims=env.observation_space.shape, n_actions=env.action_space.n,
                  memory_size=args.memory_size, batch_size=args.batch_size,
                  eps_min=args.eps_min, decay_rate=args.decay_rate,
                  replace=args.update_every, checkpoint_dir=args.checkpoint_dir, env_name=args.env,
                  algo=args.alg)
    print(agent.__dict__)
    n_steps = 0
    scores, eps_history, steps_history = [], [], []
    
    # wandb.watch(agent.Q_eval, log="all")
    # wandb.watch(agent.Q_next, log="all")
    for i in range(1, args.num_episodes+1):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            if args.train:
                agent.store_transition(observation, action, reward, int(done), observation_)
                agent.learn()
            observation = observation_
            n_steps += 1
            # wandb.log({"score": score, "epsilon": agent.epsilon, "steps": n_steps, "games": i})

        scores.append(score)
        steps_history.append(n_steps)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])
        if np.mean(scores[-100:]) >= np.mean(scores[-101:-1]):
            if args.train:
                agent.save_models()
        if score > best_score:
            best_score = score
        

        sys.stdout.write(f"Game: {i}/{args.num_episodes} | Score: {score:.2f} | Average Score: {avg_score:.2f} | Best Score: {best_score} | Epsilon: {agent.epsilon:.2f} | steps: {n_steps}\n")
    
