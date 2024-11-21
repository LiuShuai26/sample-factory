from typing import Dict, Tuple
import torch

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config, StatusCode
from sample_factory.utils.utils import debug_log_every_n, experiment_dir, log


def save(cfg: Config, obs_num, state_size) -> Tuple[StatusCode, float]:

    cfg = load_from_checkpoint(cfg)

    eval_env_frameskip: int = cfg.env_frameskip if cfg.eval_env_frameskip is None else cfg.eval_env_frameskip
    assert (
            cfg.env_frameskip % eval_env_frameskip == 0
    ), f"{cfg.env_frameskip=} must be divisible by {eval_env_frameskip=}"
    render_action_repeat: int = cfg.env_frameskip // eval_env_frameskip
    cfg.env_frameskip = cfg.eval_env_frameskip = eval_env_frameskip
    log.debug(f"Using frameskip {cfg.env_frameskip} and {render_action_repeat=} for evaluation")

    cfg.num_envs = 1

    render_mode = "human"
    if cfg.save_video:
        render_mode = "rgb_array"
    elif cfg.no_render:
        render_mode = None

    env = make_env_func_batched(
        cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=render_mode
    )
    env_info = extract_env_info(env, cfg)

    if hasattr(env.unwrapped, "reset_on_init"):
        # reset call ruins the demo recording for VizDoom
        env.unwrapped.reset_on_init = False

    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()

    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
    actor_critic.model_to_device(device)

    policy_id = cfg.policy_index
    name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
    checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
    checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict["model"])
    print(actor_critic.summaries())


    # --- Save traced model ---

    traced_model_name = "cat_traced_model.pt"
    example_input = {'obs': torch.randn(1, obs_num)}
    example_state = torch.zeros(1, state_size)
    # Encountering a dict at the output of the tracer might cause the trace to be incorrect,
    # this is only valid if the container structure does not change based on the module's inputs.
    traced_model = torch.jit.trace(actor_critic, (example_input, example_state), strict=False)
    traced_model.save(traced_model_name)

    print("-----------------------------")
    print("-----------------------------")
    print("Cat Traced Model Saved")
    print("-----------------------------")

    # Load the traced model
    loaded_model = torch.jit.load(traced_model_name)

    # Use the traced model
    # normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
    output = loaded_model(example_input, example_state)

    print(output)
    print("-----------------------------")
    print("-----------------------------")
    # --- Save traced model end ---
