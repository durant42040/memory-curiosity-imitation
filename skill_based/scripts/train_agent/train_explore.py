import argparse
import datetime
import sys
import time
import tensorboardX
import torch_ac
import utils
from model import ACModel

from utils import device

import envs.minigrid
from envs.minigrid.wrappers import FullyObsWrapper

from agents import GAILPPOAlgo

class ExploreTrainer:
    def __init__(self, args):
        self.args = args
        self.args.mem = self.args.recurrence > 1
        self.model_dir = self.set_model_dir()
        self.txt_logger, self.csv_file, self.csv_logger, self.tb_writer = self.setup_loggers()
        utils.seed(self.args.seed)
        self.txt_logger.info(f"Device: {device}\n")

        self.envs = self.load_environments()
        self.txt_logger.info("Environments loaded\n")

        self.status = self.load_status()
        self.txt_logger.info("Training status loaded\n")

        self.obs_space, self.preprocess_obss = self.load_observation_preprocessor()
        self.acmodel = self.load_model()
        self.algo = self.load_algorithm()

    def set_model_dir(self):
        date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        default_model_name = f"{self.args.env}_{self.args.algo}_seed{self.args.seed}_{date}"
        model_name = self.args.model or default_model_name
        return utils.get_model_dir(model_name)

    def setup_loggers(self):
        txt_logger = utils.get_txt_logger(self.model_dir)
        csv_file, csv_logger = utils.get_csv_logger(self.model_dir)
        tb_writer = tensorboardX.SummaryWriter(self.model_dir)
        txt_logger.info("{}\n".format(" ".join(sys.argv)))
        txt_logger.info("{}\n".format(self.args))
        return txt_logger, csv_file, csv_logger, tb_writer

    def load_environments(self):
        envs = []
        for i in range(self.args.procs):
            env = utils.make_env(self.args.env, self.args.seed + 10000 * i)
            if self.args.full_obs:
                env = FullyObsWrapper(env)
            envs.append(env)
        return envs

    def load_status(self):
        try:
            return utils.get_status(self.model_dir)
        except OSError:
            return {"num_frames": 0, "update": 0}

    def load_observation_preprocessor(self):
        obs_space, preprocess_obss = utils.get_obss_preprocessor(self.envs[0].observation_space)
        if "vocab" in self.status:
            preprocess_obss.vocab.load_vocab(self.status["vocab"])
        self.txt_logger.info("Observations preprocessor loaded")
        return obs_space, preprocess_obss

    def load_model(self):
        acmodel = ACModel(self.obs_space, self.envs[0].action_space, self.args.mem, self.args.text)
        if "model_state" in self.status:
            acmodel.load_state_dict(self.status["model_state"])
        acmodel.to(device)
        self.txt_logger.info("Model loaded\n")
        self.txt_logger.info("{}\n".format(acmodel))
        return acmodel

    def load_algorithm(self):
        algo = GAILPPOAlgo(
            self.envs,
            self.acmodel,
            device,
            self.args.frames_per_proc,
            self.args.discount,
            self.args.lr,
            self.args.gae_lambda,
            self.args.entropy_coef,
            self.args.value_loss_coef,
            self.args.max_grad_norm,
            self.args.recurrence,
            self.args.optim_eps,
            self.args.clip_eps,
            self.args.epochs,
            self.args.batch_size,
            self.preprocess_obss,
            self.args.disc_path,
            self.args.regularize_disc,
            self.args.encoder,
            self.args.policy_ckpt_path
        )

        if "optimizer_state" in self.status:
            algo.optimizer.load_state_dict(self.status["optimizer_state"])
        self.txt_logger.info("Optimizer loaded\n")
        return algo

    def train(self):
        num_frames = self.status["num_frames"]
        update = self.status["update"]
        start_time = time.time()
        update_this_round = 0
        # while num_frames < self.args.frames :
        # while update_this_round < self.args.save_interval:
        while update_this_round < self.args.update_per_gail_iter:
            update_start_time = time.time()
            exps, logs1 = self.algo.collect_experiences()
            logs2 = self.algo.update_parameters(exps)
            logs = {**logs1, **logs2}
            update_end_time = time.time()

            num_frames += logs["num_frames"]
            update += 1
            update_this_round += 1
            if update % self.args.log_interval == 0:
                self.log_progress(logs, num_frames, update, start_time, update_start_time, update_end_time)

            if self.args.save_interval > 0 and update % self.args.save_interval == 0:
                self.save_status(num_frames, update)

    def log_progress(self, logs, num_frames, update, start_time, update_start_time, update_end_time):
        fps = logs["num_frames"] / (update_end_time - update_start_time)
        duration = int(time.time() - start_time)
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

        header = ["update", "frames", "FPS", "duration"]
        data = [update, num_frames, fps, duration]
        header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
        data += rreturn_per_episode.values()
        header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
        data += num_frames_per_episode.values()
        header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
        data += [
            logs["entropy"],
            logs["value"],
            logs["policy_loss"],
            logs["value_loss"],
            logs["grad_norm"],
        ]

        self.txt_logger.info(
            "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}".format(
                *data
            )
        )

        header += ["return_" + key for key in return_per_episode.keys()]
        data += return_per_episode.values()

        if self.status["num_frames"] == 0:
            self.csv_logger.writerow(header)
        self.csv_logger.writerow(data)
        self.csv_file.flush()

        for field, value in zip(header, data):
            self.tb_writer.add_scalar(field, value, num_frames)

    def save_status(self, num_frames, update):
        status = {
            "num_frames": num_frames,
            "update": update,
            "model_state": self.acmodel.state_dict(),
            "optimizer_state": self.algo.optimizer.state_dict(),
        }
        if hasattr(self.preprocess_obss, "vocab"):
            status["vocab"] = self.preprocess_obss.vocab.vocab
        utils.save_status(status, self.model_dir)
        self.txt_logger.info("Status saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", required=True, help="algorithm to use: a2c | ppo (REQUIRED)")
    parser.add_argument("--env", required=True, help="name of the environment to train on (REQUIRED)")
    parser.add_argument("--disc_path", required=True, help="path to the discriminator model (REQUIRED)")
    parser.add_argument("--model", default=None, help="name of the model (default: {ENV}_{ALGO}_{TIME})")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=1, help="number of updates between two logs (default: 1)")
    parser.add_argument("--save-interval", type=int, default=10, help="number of updates between two saves (default: 10, 0 means no saving)")
    parser.add_argument("--procs", type=int, default=16, help="number of processes (default: 16)")
    parser.add_argument("--frames", type=int, default=10**7, help="number of frames of training (default: 1e7)")
    parser.add_argument("--epochs", type=int, default=4, help="number of epochs for PPO (default: 4)")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size for PPO (default: 256)")
    parser.add_argument("--frames-per-proc", type=int, default=None, help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
    parser.add_argument("--discount", type=float, default=0.99, help="discount factor (default: 0.99)")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate (default: 0.001)")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="entropy term coefficient (default: 0.01)")
    parser.add_argument("--value-loss-coef", type=float, default=0.5, help="value loss term coefficient (default: 0.5)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="maximum norm of gradient (default: 0.5)")
    parser.add_argument("--optim-eps", type=float, default=1e-8, help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="clipping epsilon for PPO (default: 0.2)")
    parser.add_argument("--recurrence", type=int, default=4, help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
    parser.add_argument("--text", action="store_true", default=False, help="add a GRU to the model to handle text input")
    parser.add_argument("--full_obs", action="store_true", default=False, help="use full observation")


    args = parser.parse_args()



    trainer = ExploreTrainer(args)
    trainer.train()
