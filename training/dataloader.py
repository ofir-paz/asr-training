"""The data collator: applies the given augmentations to each example's features, then
pads/stacks the batch and builds the decoder inputs/labels."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Union

import numpy as np
import torch
from transformers import BatchFeature


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    # Applied in order to each example's (unpadded) log-mel features; empty for eval.
    augmentations: List[Callable[[torch.Tensor], torch.Tensor]] = field(default_factory=list)

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Ensure input_features are decompressed if needed:
        input_features = []
        for feature in features:
            pad_amount = feature.get("pad_amount", 0)
            base_features = torch.tensor(feature["input_features"])  # (d, feat_len)

            for augment in self.augmentations:
                base_features = augment(base_features)

            if pad_amount > 0:
                # Broadcast the stored pad column, rather than torch.tensor([pad_value]*n)
                # which converts a list of numpy arrays and is slow.
                pad_value = torch.as_tensor(np.asarray(feature["pad_value"], dtype=np.float32))
                pad_tensor = pad_value.unsqueeze(-1).expand(-1, pad_amount)
                input_features.append(torch.concatenate([base_features, pad_tensor], dim=-1))
            else:
                input_features.append(base_features)

        batch = BatchFeature({"input_features": torch.stack(input_features)})

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"]

        # Labels, represent the input to the decoder
        batch["decoder_input_ids"] = labels[:, :-1]

        # Shift all labels to the left, thus the expected generated label
        # is at the same index of the generated output id from the decoder
        # and the loss function would compare them (cross entropy loss in this case)
        # Note - this means there is no loss calculated for the first "start of transcript" token id
        # since it is not expected to be predicted but always provided.
        # The loss is calculated for the task/lang/notimestamp tokens since the model needs to know
        # to associate them with the proper output
        # **Warning!** the labels are shifted here, and some version of transformers will assume
        # they are not if using the default "ForCausalLMLoss"
        # Once Whisper is updated to use that built-in loss - need to reconsider the collator.
        # Atm the custom loss function expects this shift to be done here.
        labels = labels[:, 1:]
        labels_mask = labels_batch.attention_mask[:, 1:]

        # Where we do not need to attend when calculating loss - -100 is the agreed
        # ignored value for the pytorch loss functions
        labels = labels.masked_fill(labels_mask.ne(1), -100)

        # replace initial prompt tokens with -100 to ignore correctly when computing the loss
        bos_index = torch.argmax((labels == self.decoder_start_token_id).long(), dim=1)
        bos_index = torch.where(bos_index > 0, bos_index + 1, bos_index)
        prompt_mask = torch.arange(labels.shape[1]) < bos_index[:, None]
        labels = torch.where(prompt_mask, -100, labels)

        batch["labels"] = labels
        return batch
