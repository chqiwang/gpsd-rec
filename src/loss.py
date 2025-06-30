import torch
from torch import nn
import torch.nn.functional as F


class SoftmaxLoss(nn.Module):
    def __init__(self, projection: torch.Tensor, temperature :float=1.0, l2_norm=True):
        super().__init__()
        self.projection = projection
        self.temperature = temperature
        self.l2_norm = l2_norm
        self.ignore_index = -100

    def may_be_norm(self, inputs):
        eps = 1e-6
        if self.l2_norm:
            return inputs / torch.clamp(
            torch.linalg.norm(inputs, ord=None, dim=-1, keepdim=True),
            min=eps,
        )
        else:
            return inputs 

    def forward(self, inputs, labels, weights: torch.Tensor=None):
        vocab_size = self.projection.shape[0]
        scores = torch.einsum('blh,vh->blv', self.may_be_norm(inputs), self.may_be_norm(self.projection)) # [BL,H] x [V,H] -> [BL,V]
        scores /= self.temperature
        if weights is None:
            loss = F.cross_entropy(scores.view([-1, vocab_size]), labels.view(-1), ignore_index=self.ignore_index)
        else:
            weights = weights.reshape([-1])
            loss = F.cross_entropy(scores.view([-1, vocab_size]), labels.view(-1), ignore_index=self.ignore_index, reduction="none")
            loss = (loss*weights).sum()/(weights.sum()+1e-6)
        return loss


class SampledSoftmaxLoss(SoftmaxLoss):
    def __init__(self, projection: torch.Tensor, n_samples :int=4096, included_ids :list=[], temperature :float=1.0, l2_norm=True):
        super().__init__(projection=projection, temperature=temperature, l2_norm=l2_norm)
        self.included_ids = included_ids
        self.projection = projection
        self.temperature = temperature
        self.n_samples = n_samples
        self.l2_norm = l2_norm

    def sample(self, positive_ids: torch.Tensor):
        n_samples_ = self.n_samples - len(self.included_ids)
        output_shape = positive_ids.size() + (n_samples_,)
        sampled_ids = torch.randint(
            low=0,
            high=self.projection.size()[0],
            size=output_shape,
            dtype=positive_ids.dtype,
            device=positive_ids.device,
        )
        if self.included_ids:
            included_ids = torch.zeros(positive_ids.size() + (len(self.included_ids),), dtype=torch.int64, device=positive_ids.device) + \
                            torch.tensor(self.included_ids, dtype=torch.int64, device=positive_ids.device)
            sampled_ids = torch.cat([included_ids, sampled_ids], axis=-1)
        return sampled_ids

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor=None):
        # flatten the tensors
        inputs = inputs.reshape([-1, inputs.size()[-1]]) # [BL,H]
        labels = labels.reshape([-1]) # [BL]
        # maybe norm inputs
        inputs = self.may_be_norm(inputs)
        # Get target label scores
        label_scores = (inputs * self.may_be_norm(self.projection[labels, :])).sum(-1, keepdim=True) # [BL,H] x [BL,H] -> [BL]
        # Sample noise & get scores
        samples = self.sample(labels) # [BL,N]
        noise_scores = torch.einsum('abc,ac->ab', self.may_be_norm(self.projection[samples, :]), inputs)  # [BL,N,H] x [BL,H] -> [BL,N]
        # Reject samples matching target label & correct for remaining samples
        reject_samples = labels[:,None] == samples # [BL,N]
        noise_scores -= 1e6 * reject_samples
        # noise_scores -= torch.log((self.n_samples - reject_samples.sum(-1, keepdims=True)).float())
        # Apply regular softmax cross entropy
        scores = torch.cat([label_scores, noise_scores], dim=-1)/self.temperature
        labels = torch.where(labels==self.ignore_index, self.ignore_index, torch.zeros_like(labels))
        scores = scores.reshape([-1, self.n_samples+1])
        labels = labels.reshape([-1])
        if weights is None:
            loss = F.cross_entropy(scores, labels, ignore_index=self.ignore_index)
        else:
            weights = weights.reshape([-1])
            loss = F.cross_entropy(scores, labels, ignore_index=self.ignore_index, reduction="none")
            loss = (loss*weights).sum()/(weights.sum()+1e-6)
        return loss


class MemoryEfficientV1SampledSoftmaxLoss(SampledSoftmaxLoss):
    def sample(self, positive_ids: torch.Tensor):
        n_samples_ = self.n_samples - len(self.included_ids)
        output_shape = positive_ids.size()[:1] + (n_samples_,)
        sampled_ids = torch.randint(
            low=0,
            high=self.projection.size()[0],
            size=output_shape,
            dtype=positive_ids.dtype,
            device=positive_ids.device,
        )
        if self.included_ids:
            included_ids = torch.zeros(positive_ids.size()[:1] + (len(self.included_ids),), dtype=torch.int64, device=positive_ids.device) + \
                            torch.tensor(self.included_ids, dtype=torch.int64, device=positive_ids.device)
            sampled_ids = torch.cat([included_ids, sampled_ids], axis=-1)
        return sampled_ids

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor=None):
        assert len(inputs.size()) == 3
        # maybe norm inputs
        inputs = self.may_be_norm(inputs)
        # Get target label scores
        label_scores = (inputs * self.may_be_norm(self.projection[labels, :])).sum(-1, keepdim=True) # [B,L,H] x [B,L,H] -> [B,L,1]
        # Sample noise & get scores
        samples = self.sample(labels) # [B,N]
        noise_scores = torch.einsum('acd,abd->abc', self.may_be_norm(self.projection[samples, :]), inputs)
        # Reject samples matching target label & correct for remaining samples
        reject_samples = labels.unsqueeze(2) == samples.unsqueeze(1) # [B,L,N]
        noise_scores -= 1e6 * reject_samples
        # Apply regular softmax cross entropy
        scores = torch.cat([label_scores, noise_scores], dim=-1)/self.temperature
        labels = torch.where(labels==self.ignore_index, self.ignore_index, torch.zeros_like(labels))
        scores = scores.reshape([-1, self.n_samples+1])
        labels = labels.reshape([-1])
        if weights is None:
            loss = F.cross_entropy(scores, labels, ignore_index=self.ignore_index)
        else:
            weights = weights.reshape([-1])
            loss = F.cross_entropy(scores, labels, ignore_index=self.ignore_index, reduction="none")
            loss = (loss*weights).sum()/(weights.sum()+1e-6)
        return loss


class MemoryEfficientV2SampledSoftmaxLoss(MemoryEfficientV1SampledSoftmaxLoss):
    def sample(self, positive_ids: torch.Tensor):
        n_samples_ = self.n_samples - len(self.included_ids)
        output_shape = (n_samples_,)
        sampled_ids = torch.randint(
            low=0,
            high=self.ext_modules["embedding"].num_embeddings,
            size=output_shape,
            dtype=positive_ids.dtype,
            device=positive_ids.device,
        )
        if self.included_ids:
            included_ids = torch.tensor(self.included_ids, dtype=torch.int64, device=positive_ids.device)
            sampled_ids = torch.cat([included_ids, sampled_ids], axis=-1)
        return sampled_ids

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor=None):
        assert len(inputs.size()) == 3
        # maybe norm inputs
        inputs = self.may_be_norm(inputs)
        # Get target label scores
        label_scores = (inputs * self.may_be_norm(self.ext_modules["embedding"](torch.clamp(labels, min=0)))).sum(-1, keepdim=True) # [B,L,H] x [B,L,H] -> [B,L]
        # Sample noise & get scores
        samples = self.sample(labels) # [N]
        noise_scores = torch.einsum('cd,abd->abc', self.may_be_norm(self.ext_modules["embedding"](samples)), inputs) # [N,H] x [B,L,H] -> [B,L,N]
        # Reject samples matching target label & correct for remaining samples
        reject_samples = labels.unsqueeze(2) == samples # [B,L,N]
        noise_scores -= 1e6 * reject_samples
        # noise_scores -= torch.log((self.n_samples - reject_samples.sum(-1, keepdims=True)).float())
        # Apply regular softmax cross entropy
        scores = torch.cat([label_scores, noise_scores], dim=-1)/self.temperature
        labels = torch.where(labels==self.ignore_index, self.ignore_index, torch.zeros_like(labels))
        scores = scores.reshape([-1, self.n_samples+1])
        labels = labels.reshape([-1])
        if weights is None:
            loss = F.cross_entropy(scores, labels, ignore_index=self.ignore_index)
        else:
            weights = weights.reshape([-1])
            loss = F.cross_entropy(scores, labels, ignore_index=self.ignore_index, reduction="none")
            loss = (loss*weights).sum()/(weights.sum()+1e-6)
        return loss


if __name__ == '__main__':
    loss_fn = SoftmaxLoss(torch.randn([10000,4]))
    loss = loss_fn.forward(torch.randn([12,16,4]), torch.ones([12,16], dtype=torch.int64), torch.clamp(torch.randn([12,16]), 0))
    print(loss)