# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
import torch.nn.functional as F


@dataclass
class ScoringRuleConfig(FairseqDataclass):
    score_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for score smoothing, 0 means no score smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    scoring_rule: str = field(
        default = "logarithmic",
        metadata={"help": "choose from logarithmic, brier, spherical, default is logarithmic"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

@register_criterion(
    "scoring_rule_loss", dataclass=ScoringRuleConfig
)
class ScoringRuleLoss(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        score_smoothing,
        scoring_rule,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = score_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.scoring_rule = scoring_rule

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        if self.scoring_rule == 'logarithmic':
            loss, score_loss = self.compute_logarithmic_loss(model, net_output, sample, reduce=reduce)
        elif self.scoring_rule == 'brier':
            loss, score_loss = self.compute_brier_loss(model, net_output, sample, reduce=reduce)
        elif self.scoring_rule == 'spherical':
            loss, score_loss = self.compute_spherical_loss(model, net_output, sample, reduce=reduce)
        else:
            raise NotImplementedError

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "score_loss": score_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_logarithmic_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_brier_loss(self, model, net_output, sample, reduce=True):
        probs = model.get_normalized_probs(net_output, log_probs=False)
        batch_size, length, vocab_size = probs.size()
        target = model.get_targets(sample, net_output).unsqueeze(-1)
        probs_golden = probs.gather(dim=-1, index=target).squeeze()
        non_pad_mask = target.eq(self.padding_idx).view(batch_size, length)

        quadratic_sum = torch.sum(torch.pow(probs, 2), dim = -1)
        brier_loss = quadratic_sum - 2 * probs_golden
        smooth_loss_1 = quadratic_sum - 2 * torch.mean(probs, dim = -1)
        smooth_mask = (probs < (self.eps / vocab_size)).float()
        smooth_loss_2 = - self.eps * torch.mean(torch.log(probs) * smooth_mask, dim = -1)

        loss = (1 - self.eps) * brier_loss + self.eps * smooth_loss_1 + smooth_loss_2
        loss.masked_fill_(non_pad_mask, 0.0)
        brier_loss.masked_fill_(non_pad_mask, 0.0)

        return torch.sum(loss), torch.sum(brier_loss)

    def compute_spherical_loss(self, model, net_output, sample, reduce=True):
        probs = model.get_normalized_probs(net_output, log_probs=False)
        batch_size, length, vocab_size = probs.size()
        target = model.get_targets(sample, net_output).unsqueeze(-1)
        probs_golden = probs.gather(dim=-1, index=target).squeeze()
        non_pad_mask = target.eq(self.padding_idx).view(batch_size, length)

        probs_norm = torch.norm(probs, dim = -1)
        spherical_loss = - probs_golden / probs_norm
        smooth_loss_1 = - (1 / vocab_size) / probs_norm
        smooth_mask = (probs < (self.eps / vocab_size)).float()
        smooth_loss_2 = - self.eps * torch.mean(torch.log(probs) * smooth_mask, dim = -1)

        loss = (1 - self.eps) * spherical_loss + self.eps * smooth_loss_1 + smooth_loss_2
        loss.masked_fill_(non_pad_mask, 0.0)
        spherical_loss.masked_fill_(non_pad_mask, 0.0)

        return torch.sum(loss), torch.sum(spherical_loss)

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        score_loss_sum = sum(log.get("score_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size , sample_size, round=3
        )
        metrics.log_scalar(
            "score_loss", score_loss_sum / ntokens, ntokens, round=3
        )
        
        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
