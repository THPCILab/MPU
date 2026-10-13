"""Paired deployment evaluation: same QAT checkpoint plus active-MZI phase error."""
import math
import numpy as np
import torch

VOLTAGE_RMS_V = 0.010
DELTA_V_SQUARED = 1.78
PHASE_STEP_RAD = math.pi / 4.0
PHASE_GAIN = PHASE_STEP_RAD / DELTA_V_SQUARED
MEAN_V_SQUARED = 4.0 * DELTA_V_SQUARED
PHASE_RMS_RAD = math.pi * VOLTAGE_RMS_V / math.sqrt(DELTA_V_SQUARED)
NOISE_DRAWS = 5

def install(exp):
    original_condition = exp.run_condition
    def paired_condition(args, model, masks, train_loaders, test_loaders, task_ids, device):
        baseline = original_condition(args,model,masks,train_loaders,test_loaders,task_ids,device)
        saved_cuda_rng = torch.cuda.get_rng_state_all()
        saved_cpu_rng = torch.get_rng_state()
        draws = []
        totals = {}
        try:
            for layer in model.mzi_layers:
                layer.active_phase_noise_rms = PHASE_RMS_RAD
            for draw in range(NOISE_DRAWS):
                noise_seed = 441000 + draw
                torch.cuda.manual_seed_all(noise_seed)
                matrices = exp.compute_confusion_matrices(model,test_loaders,task_ids,device)
                task_acc = {tid:m["accuracy"] for tid,m in matrices.items()}
                draws.append({"noise_seed":noise_seed,"mean_accuracy":float(np.mean(list(task_acc.values()))),
                              "task_accuracy":task_acc})
                for tid, m in matrices.items():
                    counts = np.asarray(m["counts"],dtype=np.int64)
                    if tid not in totals:
                        totals[tid] = {"labels":m["labels"],"counts":counts}
                    else:
                        assert totals[tid]["labels"] == m["labels"]
                        totals[tid]["counts"] += counts
        finally:
            for layer in model.mzi_layers:
                layer.active_phase_noise_rms = 0.0
            torch.cuda.set_rng_state_all(saved_cuda_rng)
            torch.set_rng_state(saved_cpu_rng)
        confusion = {}
        for tid, item in totals.items():
            counts = item["counts"]
            confusion[tid] = {"labels":item["labels"],"counts":counts.tolist(),
                              "accuracy":float(np.trace(counts)/counts.sum())}
        task_mean = {tid:m["accuracy"] for tid,m in confusion.items()}
        mean_accuracy = float(np.mean(list(task_mean.values())))
        baseline["paired_mzi_voltage_error"] = {
            "condition":"same_QAT_checkpoint_with_active_MZI_10mV_RMS_error",
            "training_seed":args.seed,
            "additional_finetuning":False,
            "voltage_rms_V":VOLTAGE_RMS_V,
            "delta_V_squared_V2":DELTA_V_SQUARED,
            "assumed_phase_step_rad":PHASE_STEP_RAD,
            "assumed_phase_operating_range_rad":[0.0,2.0*math.pi],
            "phase_gain_rad_per_V2":PHASE_GAIN,
            "mean_V_squared_V2":MEAN_V_SQUARED,
            "phase_rms_rad":PHASE_RMS_RAD,
            "phase_rms_degree":math.degrees(PHASE_RMS_RAD),
            "scope":"theta and phi phase shifters of each remaining active/private MZI; passive shared MPU matrices unchanged",
            "noise_model":"independent zero-mean Gaussian equivalent phase offsets; resampled per forward batch; common hardware error within each batch",
            "test_noise_draws":NOISE_DRAWS,
            "mean_accuracy":mean_accuracy,
            "noise_draw_sd_accuracy":float(np.std([d["mean_accuracy"] for d in draws],ddof=1)),
            "task_accuracy":task_mean,
            "delta_vs_same_checkpoint_pp":(mean_accuracy-baseline["final_mean"])*100.0,
            "draw_results":draws,
            "confusion_matrices":confusion,
        }
        print("PAIRED_MZI_NOISE_RESULT " + str({"mean_accuracy":mean_accuracy,
              "delta_pp":(mean_accuracy-baseline["final_mean"])*100.0,
              "phase_rms_rad":PHASE_RMS_RAD}),flush=True)
        return baseline
    exp.run_condition = paired_condition
