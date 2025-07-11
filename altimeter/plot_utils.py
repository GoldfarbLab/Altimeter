import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb
import lightning as L
from dataset import filter_by_scan_range, match, norm_base_peak

plt.close('all')

def scoreDistPlot(losses, dataset, logger, epoch=0):
    plt.close('all')
    fig, ax = plt.subplots()
    ax.hist(losses.cpu(), 100, histtype='bar', color='blue')
    logger.experiment.log({"cs_dist_plot_" + dataset: wandb.Image(plt)})
    plt.close()

class MirrorPlotCallback(L.Callback):
    def __init__(self, dm):
        super().__init__()
        self.dm = dm

    def on_validation_end(self, trainer, pl_module):
        val_dataset = self.dm.getAltimeterDataset("val")

        entry = val_dataset.get_target_plot(0)
        [sample, targ, mask, seq, mod, charge, nce, min_mz, max_mz, LOD, weight, moverz, annotated] = entry

        trainer.model.eval()
        with torch.no_grad():
            sample[0] = sample[0].unsqueeze(0)
            pred = trainer.model(sample)
            pred = pred.squeeze(0).cpu()

            pred, mzpred, ionspred = val_dataset.ConvertToPredictedSpectrum(pred.numpy(), seq, mod, charge)
            self.mirrorplot(entry, pred, mzpred, ionspred, pl_module, maxnorm=True, save=True)

            sample = val_dataset[0]
            pred_ions = np.array([val_dataset.D.index2ion[ind] for ind in range(len(val_dataset.D.index2ion))])
            self.smoothplot(trainer, pl_module, sample['targ'], sample['samples'], pred_ions)
        # self.checkSmoothness(trainer, pl_module, val_dataset, ionspred)

    def checkSmoothness(self, trainer, pl_module, val_dataset):
        samples = [val_dataset[i] for i in range(1)]
        unique_precursors = set([sample["seq"] + "_" + sample["mod"] + "_" + str(sample["charge"]) for sample in samples])
        for sample in samples:
            sample_id = sample["seq"] + "_" + sample["mod"] + "_" + str(sample["charge"])
            if sample_id in unique_precursors:
                targ = sample['targ']
                sample = sample['samples']
                sample[0] = sample[0].unsqueeze(0).repeat(201, 1, 1)
                sample[1] = sample[1].repeat(201)
                sample[2] = torch.linspace(20, 40, steps=201)
                pred = trainer.model(sample)
                pred = pred.cpu()
                pred = torch.where(targ > 0, pred, 0.0)
                non_empty_mask = pred.abs().sum(dim=0).bool()
                pred = pred[:, non_empty_mask]

                # pred = pred - pred.min(dim=0, keepdim=True)[0]
                # pred = pred / pred.max(dim=0, keepdim=True)[0]
                self.smoothplot(pl_module, pred)

    def smoothplot(self, trainer, pl_module, targ, sample, ionspred):
        min_NCE = 20
        max_NCE = 40
        num_steps = 201
        sample[0] = sample[0].unsqueeze(0).repeat(num_steps, 1, 1)
        sample[1] = sample[1].repeat(num_steps)
        sample[2] = torch.linspace(min_NCE, max_NCE, steps=num_steps)

        pred = trainer.model(sample)
        pred = pred.cpu()
        pred = torch.where(targ > 0, pred, 0.0)
        non_empty_mask = pred.abs().sum(dim=0).bool()
        pred = pred[:, non_empty_mask]
        ionspred = ionspred[non_empty_mask]

        num_frags = pred.shape[1]
        num_cols = int(np.ceil(np.sqrt(num_frags)))
        num_rows = num_cols

        plt.close('all')
        fig, axs = plt.subplots(num_rows, num_cols)
        fig.tight_layout()

        NCEs = np.linspace(min_NCE, max_NCE, num=num_steps)

        for frag_i in range(pred.shape[1]):
            col = frag_i % num_cols
            row = int(frag_i / num_cols)
            axs[row, col].plot(NCEs, pred[:, frag_i].numpy())
            axs[row, col].set_title(ionspred[frag_i])

        pl_module.logger.experiment.log({"smoothplot": wandb.Image(plt)})
        plt.close()

    def mirrorplot(self, entry, pred, mzpred, ionspred, pl_module, maxnorm=True, save=True):
        [sample, targ, mask, seq, mod, charge, nce, min_mz, max_mz, LOD, weight, mz, annotated] = entry

        plt.close('all')

        if maxnorm:
            pred /= pred.max()
        if maxnorm:
            targ /= targ.max()

        sort_pred = mzpred.argsort()
        sort_targ = mz.argsort()

        pred = pred[sort_pred]
        targ = targ[sort_targ]
        mz = mz[sort_targ]
        mzpred = mzpred[sort_pred]
        ionspred = ionspred[sort_pred]
        mask = mask[sort_targ]
        annotated = annotated[sort_targ]

        plt.close('all')
        fig, ax = plt.subplots()
        fig.set_figwidth(15)
        ax.set_xlabel("m/z")
        ax.set_ylabel("Intensity")

        if np.max(pred) > 0:
            max_mz_plot = max(np.max(mz), np.max(mzpred[pred > 0])) + 10
        else:
            max_mz_plot = np.max(mz) + 10

        rect_lower = plt.Rectangle((0, -1), min_mz, 2, facecolor="#EEEEEE")
        rect_upper = plt.Rectangle((max_mz, -1), max_mz_plot, 2, facecolor="#EEEEEE")

        ax.add_patch(rect_lower)
        ax.add_patch(rect_upper)

        linestyles = ["solid" if m == 0 else (0, (1, 1)) for m in mask[annotated]]
        ax.vlines(mz[annotated], ymin=0, ymax=targ[annotated], linewidth=1, color='#111111', linestyle=linestyles)
        ax.vlines(mz[annotated == False], ymin=0, ymax=targ[annotated == False], linewidth=1, color='#BBBBBB')

        plot_indices = np.logical_and(np.char.find(ionspred, "Int") != 0, np.char.find(ionspred, "I") == 0)
        colors = ["#ff7f00" if m >= LOD else "#BBBBBB" for m in pred[plot_indices]]
        ax.vlines(mzpred[plot_indices], ymin=-pred[plot_indices], ymax=0, linewidth=1, color=colors)

        plot_indices = np.logical_and(np.char.find(ionspred, "-") != 0, np.char.find(ionspred, "p") == 0)
        colors = ["#a65628" if m >= LOD else "#BBBBBB" for m in pred[plot_indices]]
        ax.vlines(mzpred[plot_indices], ymin=-pred[plot_indices], ymax=0, linewidth=1, color=colors)
        plot_indices = np.logical_and(np.char.find(ionspred, "-") == 0, np.char.find(ionspred, "p") == 0)
        colors = ["#a65628" if m >= LOD else "#BBBBBB" for m in pred[plot_indices]]
        ax.vlines(mzpred[plot_indices], ymin=-pred[plot_indices], ymax=0, linewidth=1, color=colors, alpha=0.5)

        plot_indices = np.logical_and(np.char.find(ionspred, "-") != 0, np.char.find(ionspred, "b") == 0)
        colors = ["#e41a1c" if m >= LOD else "#BBBBBB" for m in pred[plot_indices]]
        ax.vlines(mzpred[plot_indices], ymin=-pred[plot_indices], ymax=0, linewidth=1, color=colors)
        plot_indices = np.logical_and(np.char.find(ionspred, "-") == 0, np.char.find(ionspred, "b") == 0)
        colors = ["#e41a1c" if m >= LOD else "#BBBBBB" for m in pred[plot_indices]]
        ax.vlines(mzpred[plot_indices], ymin=-pred[plot_indices], ymax=0, linewidth=1, color=colors, alpha=0.5)

        plot_indices = np.logical_and(np.char.find(ionspred, "-") != 0, np.char.find(ionspred, "y") == 0)
        colors = ["#377eb8" if m >= LOD else "#BBBBBB" for m in pred[plot_indices]]
        ax.vlines(mzpred[plot_indices], ymin=-pred[plot_indices], ymax=0, linewidth=1, color=colors)
        plot_indices = np.logical_and(np.char.find(ionspred, "-") == 0, np.char.find(ionspred, "y") == 0)
        colors = ["#377eb8" if m >= LOD else "#BBBBBB" for m in pred[plot_indices]]
        ax.vlines(mzpred[plot_indices], ymin=-pred[plot_indices], ymax=0, linewidth=1, color=colors, alpha=0.5)

        ac_or = np.logical_or(np.char.find(ionspred, "a") == 0, np.char.find(ionspred, "c") == 0)
        xz_or = np.logical_or(np.char.find(ionspred, "x") == 0, np.char.find(ionspred, "z") == 0)
        other_term_or = np.logical_or(ac_or, xz_or)
        plot_indices = np.logical_and(np.char.find(ionspred, "-") != 0, other_term_or)
        colors = ["#f781bf" if m >= LOD else "#BBBBBB" for m in pred[plot_indices]]
        ax.vlines(mzpred[plot_indices], ymin=-pred[plot_indices], ymax=0, linewidth=1, color=colors)
        plot_indices = np.logical_and(np.char.find(ionspred, "-") == 0, other_term_or)
        colors = ["#f781bf" if m >= LOD else "#BBBBBB" for m in pred[plot_indices]]
        ax.vlines(mzpred[plot_indices], ymin=-pred[plot_indices], ymax=0, linewidth=1, color=colors, alpha=0.5)

        plot_indices = np.logical_and(np.char.find(ionspred, "-") != 0, np.char.find(ionspred, "Int") == 0)
        colors = ["#984ea3" if m >= LOD else "#BBBBBB" for m in pred[plot_indices]]
        ax.vlines(mzpred[plot_indices], ymin=-pred[plot_indices], ymax=0, linewidth=1, color=colors)
        plot_indices = np.logical_and(np.char.find(ionspred, "-") == 0, np.char.find(ionspred, "Int") == 0)
        colors = ["#984ea3" if m >= LOD else "#BBBBBB" for m in pred[plot_indices]]
        ax.vlines(mzpred[plot_indices], ymin=-pred[plot_indices], ymax=0, linewidth=1, color=colors, alpha=0.5)

        ax.set_xlim([0, ax.get_xlim()[1]])
        ax.set_ylim([-1.1, 1.1])
        ax.set_xlim([0, max_mz_plot])
        ax.set_xticks(np.arange(0, ax.get_xlim()[1], 200))
        ax.set_xticks(np.arange(0, ax.get_xlim()[1], 50), minor=True)

        targ, mz, annotated = filter_by_scan_range(mz, targ, min_mz, max_mz, annotated)
        targ_anno = targ[annotated]
        mz_anno = mz[annotated]
        pred, mz_pred, _ = filter_by_scan_range(mzpred, pred, min_mz, max_mz)

        targ_aligned, pred_aligned, mz_aligned = match(targ_anno, mz_anno, pred, mz_pred)

        targ_aligned = norm_base_peak(targ_aligned)
        pred_aligned = norm_base_peak(pred_aligned)

        pred_aligned[np.logical_and(pred_aligned <= LOD, targ_aligned == 0)] = 0

        cs = (pred_aligned * targ_aligned).sum() / max(
            np.linalg.norm(pred_aligned) * np.linalg.norm(targ_aligned), 1e-8
        )
        sa = 1 - 2 * (np.arccos(cs) / np.pi)

        charge = int(charge)
        nce = float(nce)
        annotated_percent = 100 * np.power(targ[annotated], 2).sum() / np.power(targ, 2).sum()
        ax.set_title(
            "Seq: %s(%d); Charge: +%d; NCE: %.2f; Mod: %s; Annotated: %.2f%%; SA=%.5f"
            % (seq, len(seq), charge, nce, mod, annotated_percent, sa)
        )

        pl_module.logger.experiment.log({"mirroplot": wandb.Image(plt)})
        plt.close()
