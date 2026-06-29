"""
MLflow tracking for the DINOv3 LoRA finetuning experiments.

All comparisons go into ONE experiment; each run is named "<site>__<scenario>" so the
runs line up in the UI. The training run logs params + per-step metrics; later the
evaluation script reattaches to the SAME run (by name or run_id) and logs its scores
there too — so train + eval live on one run.

Auth: MLflow reads MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD (or
MLFLOW_TRACKING_TOKEN) from the environment — call load_dotenv('.env') first.
"""

import mlflow

TRACKING_URI = "http://10.52.128.161/"   # the tracking server
EXPERIMENT = "dino_lora_finetune"        # one experiment for every comparison


def _flatten(d, prefix=""):
    """Flatten a (possibly nested) config dict to scalar params MLflow accepts."""
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, (list, tuple)):
            out[key] = ",".join(map(str, v))
        else:
            out[key] = v
    return out


def scenario_name(config):
    """Build a compact scenario tag from a config dict.

    e.g. {lora_r:16, out_dim:65536, use_ibot:True, use_gram:True, weights_tag:'sat'}
         -> 'r16-K65k-dino+ibot+gram+koleo-sat'
    Pass your own short string instead if you prefer.
    """
    parts = []
    if config.get("lora_r") is not None:
        parts.append(f"r{config['lora_r']}")
    if config.get("out_dim") is not None:
        parts.append(f"K{config['out_dim'] // 1000}k")
    losses = [n for n in ("dino", "ibot", "gram", "koleo")
              if config.get(f"use_{n}", n == "dino")]
    if losses:
        parts.append("+".join(losses))
    if config.get("weights_tag"):
        parts.append(config["weights_tag"])
    return "-".join(parts) or "run"


class DinoRun:
    """Context manager wrapping one MLflow run for a (site, scenario).

    Usage:
        with DinoRun("monrovia", scenario_name(cfg), config=cfg) as run:
            for step in ...:
                run.log({"loss/dino": ld, "loss/ibot": li}, step=step)
            run.log_artifact("input_site_data/monrovia/viz/overview.png")
    """

    def __init__(self, site, scenario, config=None,
                 experiment=EXPERIMENT, tracking_uri=TRACKING_URI):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        self.site, self.scenario = site, scenario
        self.run_name = f"{site}__{scenario}"
        self.run = mlflow.start_run(run_name=self.run_name)
        self.run_id = self.run.info.run_id
        mlflow.set_tags({"site": site, "scenario": scenario, "phase": "train"})
        if config:
            mlflow.log_params(_flatten(config))

    def log(self, metrics, step=None):
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_artifact(self, path, artifact_path=None):
        mlflow.log_artifact(str(path), artifact_path)

    def end(self, status="FINISHED"):
        mlflow.end_run(status=status)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        self.end("FAILED" if exc_type else "FINISHED")
        return False


def find_run_id(site, scenario, experiment=EXPERIMENT, tracking_uri=TRACKING_URI):
    """Locate an existing run_id by its '<site>__<scenario>' name (for eval reattach)."""
    mlflow.set_tracking_uri(tracking_uri)
    run_name = f"{site}__{scenario}"
    df = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string=f"tags.`mlflow.runName` = '{run_name}'",
        order_by=["start_time DESC"], max_results=1,
    )
    return None if df.empty else df.iloc[0]["run_id"]


def resume_run(site=None, scenario=None, run_id=None,
               experiment=EXPERIMENT, tracking_uri=TRACKING_URI):
    """Reopen an existing run to log MORE to it (e.g. evaluation scores -> same run).

    Usage (in the eval script, later):
        import mlflow
        from src.train.tracking import resume_run
        with resume_run("monrovia", scen):
            mlflow.log_metrics({"eval/purity": 0.81, "eval/nmi": 0.74})
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    if run_id is None:
        run_id = find_run_id(site, scenario, experiment, tracking_uri)
        if run_id is None:
            raise ValueError(f"no run named '{site}__{scenario}' in experiment '{experiment}'")
    return mlflow.start_run(run_id=run_id)
