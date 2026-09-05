"""Read-only compact extraction of authorized experiment diagnostics."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
HOSTS = {"jnin": ("216.249.100.66", "22309"), "jinn": ("154.54.102.30", "15924")}
DIAGNOSTIC_KEYS = {
    "fps/policy",
    "fps/train",
    "replay/behavior_sample_age_mean",
    "replay/world_sample_age_mean",
    "counters/environment_steps",
    "counters/ppo_update_calls",
    "counters/learner_update_calls",
    "train/ppo/batch_reward",
    "train/ppo/batch_return",
    "train/ppo/batch_target_value",
    "train/ppo/batch_valid_fraction",
    "train/ppo/critic/explained_variance",
    "train/ppo/critic/rmse",
    "train/ppo/actor/exact_kl",
    "train/ppo/actor/final_exact_kl",
    "train/ppo/actor/entropy",
    "train/ppo/actor/final_clip_fraction",
    "train/ppo/actor/entropy_coefficient",
    "train/replay_views/behavior_reward_mean",
    "train/replay_views/world_reward_mean",
    "train/ctde/reward_loss",
    "train/opt/local_world/updates",
    "train/opt/actor/updates",
    "train/opt/actor/grad_norm",
    "train/opt/critic/grad_norm",
    "train/ctde/teammate_belief_active_peer_top1",
    "train/ctde/teammate_plan_q4_active_top1",
    "report/ctde/embedding_cosine",
    "report/ctde/attack_mask_positive_recall",
    "report/ctde/action_mask_positive_recall",
    "report/ctde/teammate_belief_active_peer_top1",
    "report/ctde/teammate_belief_attack_top1",
    "report/ctde/teammate_belief_repeat_top1",
    "report/ctde/teammate_plan_q4_active_top1",
    "report/ctde/multistep_jepa_h4_prediction_std",
    "report/ctde/multistep_jepa_h4_target_std",
    "report/ctde/multistep_jepa_h4_cosine",
}
SUMMARY_KEYS = {
    "episodes",
    "wins",
    "win_rate",
    "return_mean",
    "return_std",
    "corrected_return_mean",
    "timeout_rate",
    "enemy_deaths_mean",
    "ally_deaths_mean",
    "action_target_switch_count_mean",
    "action_attack_fraction",
    "action_move_fraction",
    "evaluation_protocol",
}
WINDOW_KEYS = {
    "train/ppo/batch_reward",
    "train/ppo/batch_return",
    "train/ppo/batch_target_value",
    "train/ppo/critic/explained_variance",
    "train/ppo/critic/rmse",
    "train/ppo/actor/final_exact_kl",
    "train/ppo/actor/entropy",
    "train/replay_views/behavior_reward_mean",
    "train/replay_views/world_reward_mean",
    "train/ctde/reward_loss",
    "train/opt/local_world/updates",
    "replay/behavior_sample_age_mean",
    "replay/world_sample_age_mean",
    "fps/policy",
}


def trim(data):
    for run in data["runs"]:
        for field in ("latest", "latest_step"):
            run[field] = {k: v for k, v in run[field].items() if k in DIAGNOSTIC_KEYS}
        run["windows"] = {
            step: {k: v for k, v in values.items() if k in WINDOW_KEYS}
            for step, values in run["windows"].items()
        }
        run["final"] = {k: v for k, v in run["final"].items() if k in SUMMARY_KEYS}
    return data


REMOTE = r"""
import collections, datetime, json, os, pathlib, statistics, time
base = pathlib.Path('/workspace/majepa_ppo_treatments_cf97200')
selected_suffixes={
    'embedding_cosine','reward_loss','continuation_loss','alive_loss',
    'action_mask_loss','action_mask_positive_recall','action_mask_negative_specificity',
    'attack_mask_positive_recall','attack_mask_prediction_rate','attack_mask_target_rate',
    'teammate_belief_active_peer_nll','teammate_belief_active_peer_top1',
    'teammate_belief_attack_top1','teammate_belief_repeat_top1',
    'teammate_belief_policy_flip_vs_zero','teammate_belief_policy_kl_vs_zero',
    'teammate_belief_policy_flip_vs_peer_shuffle','teammate_belief_policy_kl_vs_peer_shuffle',
    'teammate_plan_q4_active_top1','teammate_plan_q4_active_nll',
    'multistep_jepa_h1_cosine','multistep_jepa_h4_cosine','multistep_jepa_h8_cosine',
    'multistep_jepa_h4_action_counterfactual_cosine_drop',
    'multistep_jepa_h4_prediction_std','multistep_jepa_h4_target_std',
    'multistep_jepa_h4_prediction_effective_rank','multistep_jepa_h4_target_effective_rank',
    'multistep_jepa_h4_within_team_mean_rank',
    'multistep_jepa_recent_training_view','controllable_alive_fraction',
}
def keep(key):
    return key.startswith(('train/ppo/','train/schedule/','counters/','schedule/','train/replay_views/')) or key in {'fps/policy','fps/train','replay/behavior_sample_age_mean','replay/world_sample_age_mean','replay/behavior_sample_age_p95','replay/world_sample_age_p95','replay/optimized_replay_ratio'} or key.split('/')[-1] in selected_suffixes or key.startswith('train/opt/') and key.split('/')[-1] in {'updates','grad_norm','active','skipped','param_count'}
def read(path):
    if not path.is_file(): return {}
    return json.loads(path.read_text())
def compact(x):
    return {k:v for k,v in x.items() if not isinstance(v,list)}
result = {'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runs':[]}
for run in sorted((base/'runs').iterdir()):
    if not run.is_dir(): continue
    manifest = read(run/'manifest.json')
    launch = read(run/'train/launch.json')
    mpath = run/'train/run/metrics.jsonl'
    rows=[]
    if mpath.is_file():
        for line in mpath.open():
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: pass
    latest={}; laststep={}; series=collections.defaultdict(list); curves=[]
    for row in rows:
        step=row.get('step',0)
        if any(k.startswith('eval/') for k in row): curves.append({k:v for k,v in row.items() if k=='step' or k in {'eval/win_rate','eval/wins','eval/episodes','eval/return_mean','eval/corrected_return_mean','eval/timeout_rate','eval/enemy_deaths_mean','eval/ally_deaths_mean','eval/action_target_switch_count_mean','eval/action_attack_fraction','eval/action_move_fraction'}})
        for key,value in row.items():
            if key=='step' or not keep(key): continue
            if isinstance(value,(float,int)):
                latest[key]=value; laststep[key]=step
                if not key.startswith(('episode/','eval/')):series[key].append((step,value))
    final=compact(read(run/'final128/run/evaluation_summary.json'))
    windows={str(step):{} for step in (5000,10000,20000,30000,40000,50000)}
    for key,values in series.items():
        for step in (5000,10000,20000,30000,40000,50000):
            vals=[v for s,v in values if max(0,step-5000)<s<=step]
            if vals: windows[str(step)][key]=statistics.fmean(vals)
    elapsed=mpath.stat().st_mtime-launch.get('started_at',mpath.stat().st_mtime) if mpath.exists() else 0
    result['runs'].append({'name':run.name,'manifest':manifest,'final':final,'completed':read(run/'outcome.json').get('completed',False),'last_step':max((r.get('step',0) for r in rows),default=0),'elapsed_seconds':elapsed,'log_age_seconds':time.time()-mpath.stat().st_mtime if mpath.exists() else None,'latest':latest,'latest_step':laststep,'windows':windows,'curve':curves,'source':str(run)})
print(json.dumps(result,separators=(',',':')))
"""


def collect(item):
    name, (host, port) = item
    proc = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-i",
            str(Path.home() / ".runpod/ssh/runpodctl-ssh-key"),
            "-p",
            port,
            "root@" + host,
            "python3 -",
        ],
        input=REMOTE,
        text=True,
        capture_output=True,
        check=True,
    )
    data = trim(json.loads(proc.stdout))
    (ROOT / f"{name}_treatments.json").write_text(
        json.dumps(data, separators=(",", ":"), sort_keys=True) + "\n"
    )
    return name, len(data["runs"])


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    for item in ThreadPoolExecutor(2).map(collect, HOSTS.items()):
        print(item)
