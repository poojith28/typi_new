from __future__ import annotations

import re
import shlex
from pathlib import Path


ROOT = Path("/scratch/s219110279")
RUN_ROOT = ROOT / "Typiclust_runs"
OUT_DIR = RUN_ROOT / "rerun_lidcover_lowid_20260524"
SUFFIX = "rerunlowid20260524"

SOURCE_SCRIPTS = [
    RUN_ROOT / "cifar10_full_50b/LIDCOVER.sh",
    RUN_ROOT / "cifar100_full_50b/LIDCOVER.sh",
    RUN_ROOT / "tinyimagenet_full_50b/LIDCOVER.sh",
    RUN_ROOT / "cifar100_budget_ablation/LIDCOVER_100b.sh",
    RUN_ROOT / "cifar100_ablation_50b/LIDCOVER_alpha0.sh",
    RUN_ROOT / "cifar100_ablation_50b/LIDCOVER_alpha05.sh",
    RUN_ROOT / "cifar100_ablation_50b/LIDCOVER_alpha1.sh",
    RUN_ROOT / "cifar100_ablation_50b/LIDCOVER_alpha2.sh",
    RUN_ROOT / "cifar100_ablation_50b/LIDCOVER_d020.sh",
    RUN_ROOT / "cifar100_ablation_50b/LIDCOVER_d030.sh",
    RUN_ROOT / "cifar100_ablation_50b/LIDCOVER_k20.sh",
    RUN_ROOT / "cifar100_ablation_50b/LIDCOVER_k75.sh",
    RUN_ROOT / "cifar100_ablation_50b/LIDCOVER_lowID.sh",
]


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def parse_commands(script: Path) -> list[list[str]]:
    text = script.read_text()
    logical_lines = []
    current = ""
    for line in text.splitlines():
        stripped = line.rstrip()
        if not stripped:
            continue
        if stripped.endswith("\\"):
            current += stripped[:-1] + " "
            continue
        current += stripped
        logical_lines.append(current)
        current = ""
    if current:
        logical_lines.append(current)
    commands = []
    for line in logical_lines:
        if line.strip().startswith("python train_al.py"):
            parts = shlex.split(line)
            al_value = None
            for idx, part in enumerate(parts):
                if part == "--al":
                    al_value = parts[idx + 1]
                    break
                if part.startswith("--al="):
                    al_value = part.split("=", 1)[1]
                    break
            if al_value and al_value.startswith("idprobcover"):
                commands.append(parts)
    return commands


def value(parts: list[str], flag: str) -> str:
    prefix = flag + "="
    for idx, part in enumerate(parts):
        if part == flag:
            return parts[idx + 1]
        if part.startswith(prefix):
            return part[len(prefix) :]
    raise KeyError(flag)


def replace_value(parts: list[str], flag: str, new_value: str) -> list[str]:
    out = list(parts)
    prefix = flag + "="
    for idx, part in enumerate(out):
        if part == flag:
            out[idx + 1] = new_value
            return out
        if part.startswith(prefix):
            out[idx] = prefix + new_value
            return out
    raise KeyError(flag)


def dataset_from_cfg(cfg: str) -> str:
    if "/cifar10/" in cfg:
        return "cifar10"
    if "/cifar100/" in cfg:
        return "cifar100"
    if "/tinyimagenet/" in cfg:
        return "tinyimagenet"
    return "unknown"


def rerun_exp_name(old: str) -> str:
    m = re.match(r"(.+)_([0-9]+)_([0-9]+b)$", old)
    if not m:
        return f"{old}_{SUFFIX}"
    prefix, seed, budget = m.groups()
    return f"{prefix}_{SUFFIX}_{seed}_{budget}"


def write_job(parts: list[str], source: Path) -> Path:
    cfg = value(parts, "--cfg")
    old_exp = value(parts, "--exp-name")
    budget = int(value(parts, "--budget"))
    seed = int(value(parts, "--seed"))
    if budget == 500:
        raise ValueError("budget 500 is intentionally excluded")

    dataset = dataset_from_cfg(cfg)
    new_exp = rerun_exp_name(old_exp)
    command = replace_value(parts, "--exp-name", new_exp)
    stem = f"{dataset}_{new_exp}"
    job_path = OUT_DIR / f"{stem}.sh"
    stdout = OUT_DIR / f"{stem}.out"
    stderr = OUT_DIR / f"{stem}.err"
    job_name = f"lid_{dataset}_{seed}_{budget}"
    if len(job_name) > 32:
        job_name = job_name[:32]

    job_path.write_text(
        "\n".join(
            [
                "#!/bin/bash",
                "",
                f"#SBATCH --job-name={job_name}",
                f"#SBATCH --output={stdout}",
                f"#SBATCH --error={stderr}",
                "#SBATCH --nodes=1",
                "#SBATCH --partition=gpu",
                "#SBATCH --gpus=1",
                "#SBATCH --time=18:00:00",
                "#SBATCH --qos=batch-short",
                "#SBATCH --cpus-per-task=3",
                "#SBATCH --mem=20G",
                "#SBATCH --mail-type=END",
                "#SBATCH --mail-user=s219110279@deakin.edu.au",
                "",
                "module purge",
                "module load Anaconda3",
                "source activate",
                "conda activate AutoAL",
                "",
                "cd /scratch/s219110279/TypiClust/deep-al/tools",
                "",
                "nvidia-smi -L",
                "nvidia-smi -q -d ECC",
                "",
                f"# Source script: {source}",
                f"# Original exp-name: {old_exp}",
                shell_join(command),
                "",
            ]
        )
    )
    return job_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for script in SOURCE_SCRIPTS:
        for command in parse_commands(script):
            if int(value(command, "--budget")) == 500:
                continue
            jobs.append(write_job(command, script))
    submit = OUT_DIR / "submit_all_lidcover_reruns.sh"
    submit.write_text(
        "#!/bin/bash\nset -u\n\n"
        + "\n".join(f"sbatch {shlex.quote(str(job))}" for job in jobs)
        + "\n"
    )
    print(f"Wrote {len(jobs)} rerun jobs to {OUT_DIR}")
    print(submit)


if __name__ == "__main__":
    main()
