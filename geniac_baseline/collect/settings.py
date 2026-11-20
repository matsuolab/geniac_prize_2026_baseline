"""Model-server and launch settings used to produce the baseline data and adapters.

These are the defaults of ``collect plan`` / ``eval plan``; docs/collect.md and docs/eval.md describe each one.
CaP-X file references are to geniac_prize_2026_code at the commit patches/geniac_prize/setup.sh pins, unless stated.

Nothing in this module runs anything; ``geniac_baseline.collect.plan`` turns these into a shell script.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field

# Coding model (collection, evaluation, training base) and VDM model of the baseline.
MODEL = "Qwen/Qwen3.8-27B"
# Pointing model used by nut_assembly's API; served next to the coding model.
MOLMO_MODEL = "allenai/Molmo2-8B"

# Default CaP-X ports: LLM proxy 8110 (capx/envs/launch.py LaunchArgs.server_url), perception servers
# 8113-8117 (capx/integrations/vision/*.py, capx/integrations/motion/pyroki.py), Molmo 8122
# (capx/integrations/vision/molmo.py). The evaluation code-writer server sits on 8111.
LLM_PORT = 8110
CODE_PORT = 8111
MOLMO_PORT = 8122
PERCEPTION_PORTS = (8113, 8114, 8115, 8116, 8117)

# With unix_sockets (the default) no server takes requests on a TCP port: run.sh makes a directory only its user
# can enter and exports it as CAPX_SOCKET_DIR. geniac_prize_2026_code's servers listen only on socket files in that
# directory, and its clients address them as http+unix://<name>.sock/<path> (capx/utils/unix_http.py). The vLLM
# servers started here follow the same rule: server <name> listens on $CAPX_SOCKET_DIR/<name>.sock (vLLM --uds).
# The ports above are only used with unix_sockets: false (the ASPIRE tree).
SOCKET_DIR = "$CAPX_SOCKET_DIR"
# api_servers launcher -> the socket name its client in capx/integrations/* sends to
PERCEPTION_SOCKETS = {
    "launch_sam3_server": "sam3.sock",
    "launch_sam2_server": "sam2.sock",
    "launch_contact_graspnet_server": "graspnet.sock",
    "launch_pyroki_server": "pyroki.sock",
    "launch_owlvit_server": "owlvit.sock",
    "launch_curobo_server": "curobo.sock",
}


def unix_url(name: str, path: str = "") -> str:
    """URL of ``path`` on the server whose socket is ``<name>.sock`` in CAPX_SOCKET_DIR."""
    return f"http+unix://{name}.sock{path}"


def shell_arg(x: object) -> str:
    """Quote one argument for run.sh; a socket path keeps its $CAPX_SOCKET_DIR for the shell to expand."""
    s = str(x)
    return f'"{s}"' if s.startswith(SOCKET_DIR + "/") else shlex.quote(s)


# Launch arguments.
# temperature: inert under use_parallel_ensemble (candidates use 0.1..0.9 and synthesis 0.2,
#   capx/llm/client.py SINGLE_MODEL_ENSEMBLE_TEMPERATURES L68, synthesis L633); only used for single calls.
TEMPERATURE = 1.0
# reasoning_effort: CaP-X sends it only for models in GPT_MODELS (capx/llm/client.py _payload L123-133), so
#   for Qwen it never reaches the server; the effective effort comes from the chat template's default ("low").
REASONING_EFFORT = "low"
MAX_TOKENS = 12288

# Hardware workarounds, not method settings: on B300 (sm_100/103) vLLM 0.26 picks FLASH_ATTN, whose kernel fails
# to JIT; these select other attention kernels (they change which kernel computes attention, not what the model is
# asked or answers). TORCH_CUDA_ARCH_LIST=10.0 targets the same GPU.
B300_VLLM_FLAGS = ("--attention-backend", "TRITON_ATTN", "--mm-encoder-attn-backend", "TORCH_SDPA")
B300_ENV = {"TORCH_CUDA_ARCH_LIST": "10.0"}

# Process environment of the baseline runs.
RUN_ENV = {"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1", "PYTORCH_JIT": "0", "PYTORCH_NVFUSER_DISABLE": "1"}

# The chat template the Qwen servers were started with: Qwen/Qwen3.8-27B's own template (revision 1d4bf0f) with the
# default reasoning effort "low" and the system-message image/video checks removed. Not redistributed here;
# `geniac-baseline template` derives it from the snapshot (collect/template.py). SHA-256 of the file used:
CHAT_TEMPLATE_SHA256 = "e1960ce0434778bee2c1059c74377a5a8748e88ca6f710da5ce66ccf3caaf84c"


@dataclass
class VllmServer:
    """One ``vllm.entrypoints.openai.api_server`` process."""

    name: str
    model: str
    served_model_name: str
    port: int
    max_model_len: int
    gpu_memory_utilization: float
    max_num_seqs: int = 8
    tensor_parallel_size: int = 1
    dtype: str = "bfloat16"
    reasoning_parser: str | None = None
    chat_template: str | None = None
    limit_mm_per_prompt: dict[str, int] | None = None
    default_chat_template_kwargs: dict[str, object] | None = None
    trust_remote_code: bool = False
    enforce_eager: bool = False
    max_num_batched_tokens: int | None = None
    lora_modules: dict[str, str] = field(default_factory=dict)
    max_lora_rank: int | None = None
    extra_flags: tuple[str, ...] = ()
    unix_socket: bool = False  # listen on $CAPX_SOCKET_DIR/<name>.sock instead of 127.0.0.1:<port>

    def argv(self, python: str) -> list[str]:
        a = [
            python,
            "-m",
            "vllm.entrypoints.openai.api_server",
            *(
                ["--uds", f"{SOCKET_DIR}/{self.name}.sock"]
                if self.unix_socket
                else ["--host", "127.0.0.1", "--port", str(self.port)]
            ),
            "--model",
            self.model,
            "--served-model-name",
            self.served_model_name,
            "--dtype",
            self.dtype,
            "--max-model-len",
            str(self.max_model_len),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--max-num-seqs",
            str(self.max_num_seqs),
        ]
        if self.tensor_parallel_size != 1:
            a += ["--tensor-parallel-size", str(self.tensor_parallel_size)]
        if self.trust_remote_code:
            a.append("--trust-remote-code")
        if self.enforce_eager:
            a.append("--enforce-eager")
        if self.reasoning_parser:
            a += ["--reasoning-parser", self.reasoning_parser]
        if self.max_num_batched_tokens:
            a += ["--max-num-batched-tokens", str(self.max_num_batched_tokens)]
        if self.limit_mm_per_prompt is not None:
            a += ["--limit-mm-per-prompt", json.dumps(self.limit_mm_per_prompt, separators=(", ", ": "))]
        if self.chat_template:
            a += ["--chat-template", self.chat_template]
        if self.default_chat_template_kwargs is not None:
            a += ["--default-chat-template-kwargs", json.dumps(self.default_chat_template_kwargs)]
        if self.lora_modules:
            a += [
                "--enable-lora",
                "--max-lora-rank",
                str(self.max_lora_rank or 32),
                "--max-loras",
                str(len(self.lora_modules)),
                "--lora-modules",
                *[f"{k}={v}" for k, v in self.lora_modules.items()],
            ]
        a += list(self.extra_flags)
        return a

    def shell(self, python: str) -> str:
        return " ".join(shell_arg(x) for x in self.argv(python))


def qwen_server(
    *,
    name: str,
    model_path: str,
    port: int,
    gpu_util: float,
    chat_template: str | None,
    thinking: bool,
    b300: bool,
    served_model_name: str = MODEL,
    max_model_len: int = 262144,
    tensor_parallel_size: int = 1,
) -> VllmServer:
    """The Qwen server of the runs.

    thinking=True  = collection setting and the evaluation VDM server:
                     ``--reasoning-parser qwen3``, the low-effort chat template, thinking on.
    thinking=False = evaluation code-writer server: no reasoning parser (the raw output is the training
                     target form) and ``--default-chat-template-kwargs {"enable_thinking": false}``.
    Shared: bf16, max-model-len 262144 (default), max-num-seqs 8, up to 16 images / 2 videos per prompt.
    """
    return VllmServer(
        name=name,
        model=model_path,
        served_model_name=served_model_name,
        port=port,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_util,
        max_num_seqs=8,
        tensor_parallel_size=tensor_parallel_size,
        reasoning_parser="qwen3" if thinking else None,
        chat_template=chat_template,
        limit_mm_per_prompt={"image": 16, "video": 2},
        default_chat_template_kwargs=None if thinking else {"enable_thinking": False},
        extra_flags=B300_VLLM_FLAGS if b300 else (),
    )


def molmo_server(*, model_path: str, port: int, b300: bool, gpu_util: float = 0.28) -> VllmServer:
    """Molmo2-8B pointing server (bf16, 16k context, util 0.28 by default, eager)."""
    return VllmServer(
        name="molmo",
        model=model_path,
        served_model_name=MOLMO_MODEL,
        port=port,
        max_model_len=16384,
        gpu_memory_utilization=gpu_util,
        max_num_seqs=8,
        trust_remote_code=True,
        enforce_eager=True,
        max_num_batched_tokens=32768,
        extra_flags=B300_VLLM_FLAGS if b300 else (),
    )


# GPU memory shares of the baseline runs (one B300, 275 GB): collection runs one Qwen server at 0.62; evaluation
# runs two at 0.30 each. Molmo takes 0.28 in both. The run spec can change all of them (docs/collect.md).
COLLECT_QWEN_GPU_UTIL = 0.62
EVAL_VDM_GPU_UTIL = 0.30
EVAL_CODE_GPU_UTIL = 0.30
MOLMO_GPU_UTIL = 0.28
MAX_MODEL_LEN = 262144
