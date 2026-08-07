#!/usr/bin/env python3
"""Check that a TRIAL checkout is portable and contains its runtime contract."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md",
    "requirements.txt",
    "trial/train.py",
    "trial/config/trial_trainer.yaml",
    "trial/core/trainer.py",
    "verl/trainer/ppo/trial_utils.py",
    "agent_system/environments/env_manager.py",
    "scripts/run.sh",
    "scripts/evaluate.sh",
    "scripts/smoke_gpu.py",
    "tests/test_trial.py",
)
FORBIDDEN = (
    "/" + "apdcephfs",
    "/" + "dockerdata",
    "star" + "-proxy",
    "NVME" + "_CKPT_ROOT",
)
LEGACY_METHOD_TERMS = (
    "ahd",
    "dwahd",
    "bchd",
    "serl",
    "opid",
    "opsd",
    "cahd",
)
INTERNAL_TERMS = (
    "self_distillation",
    "teacher_advantage",
    "reprompt",
    "outcome_advantage",
    "opd_loss",
    "two_step_outcome",
    "cha_utility",
    "tft preview",
)
TEXT_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".md", ".toml", ".txt"}
PUBLIC_PATHS = ("README.md", "scripts", "trial", "agent_system/multi_turn_rollout", "verl/workers/actor")


def main() -> None:
    errors: list[str] = []
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")
    links = [path for path in ROOT.rglob("*") if path.is_symlink()]
    errors.extend(f"portable release cannot contain symlink: {path.relative_to(ROOT)}" for path in links)
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            text = path.read_text(errors="replace")
            for marker in FORBIDDEN:
                if marker in text:
                    errors.append(f"machine-specific marker {marker!r} in {path.relative_to(ROOT)}")
            if path != Path(__file__):
                lowered = text.lower()
                for term in LEGACY_METHOD_TERMS:
                    if term in lowered:
                        errors.append(f"legacy method term {term!r} in {path.relative_to(ROOT)}")
            relative = path.relative_to(ROOT).as_posix()
            if relative == "scripts/check_release.py":
                continue
            if any(relative == scope or relative.startswith(scope + "/") for scope in PUBLIC_PATHS):
                lowered = text.lower()
                for term in INTERNAL_TERMS:
                    if term in lowered:
                        errors.append(f"internal implementation term {term!r} in {relative}")
    if errors:
        raise SystemExit("TRIAL release check failed:\n- " + "\n- ".join(errors))
    print(f"TRIAL release check passed ({len(REQUIRED)} required files, no non-portable paths).")


if __name__ == "__main__":
    main()
