"""
A wrapper that unrolls the second (time) dimension of a tensor
into the first (batch) dimension, applies some other `Module`,
and then rolls the time dimension back up.
"""

from typing import List


import torch


class TimeDistributed(torch.nn.Module):
    """
    Given an input shaped like `(batch_size, time_steps, [rest])` and a `Module` that takes
    inputs like `(batch_size, [rest])`, `TimeDistributed` reshapes the input to be
    `(batch_size * time_steps, [rest])`, applies the contained `Module`, then reshapes it back.

    Note that while the above gives shapes with `batch_size` first, this `Module` also works if
    `batch_size` is second - we always just combine the first two dimensions, then split them.

    It also reshapes keyword arguments unless they are not tensors or their name is specified in
    the optional `pass_through` iterable.
    """

    def __init__(self, module):
        super().__init__()
        self._module = module


    def forward(self, *inputs, pass_through: List[str] = None, **kwargs):

        pass_through = pass_through or []

        reshaped_inputs = [self._reshape_tensor(input_tensor) for input_tensor in inputs]

        # Need some input to then get the batch_size and time_steps.
        some_input = None
        if inputs:
            some_input = inputs[-1]

        reshaped_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and key not in pass_through:
                if some_input is None:
                    some_input = value

                value = self._reshape_tensor(value)

            reshaped_kwargs[key] = value

        reshaped_outputs = self._module(*reshaped_inputs, **reshaped_kwargs)

        if some_input is None:
            raise RuntimeError("No input tensor to time-distribute")

        # Now get the output back into the right shape.
        # (batch_size, time_steps, **output_size)
        new_size = some_input.size()[:2] + reshaped_outputs.size()[1:]
        outputs = reshaped_outputs.contiguous().view(new_size)

        return outputs

    @staticmethod
    def _reshape_tensor(input_tensor):
        input_size = input_tensor.size()
        if len(input_size) <= 2:
            raise RuntimeError(f"No dimension to distribute: {input_size}")
        # Squash batch_size and time_steps into a single axis; result has shape
        # (batch_size * time_steps, **input_size).
        squashed_shape = [-1] + list(input_size[2:])
        return input_tensor.contiguous().view(*squashed_shape)
