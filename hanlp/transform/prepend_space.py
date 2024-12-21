from hanlp.common.transform import ConfigurableNamedTransform
from hanlp_common.util import infer_space_after


class PrependSpace(ConfigurableNamedTransform):

    def __init__(self, src: str = 'token', dst: str = None, add_prefix_space=True) -> None:
        super().__init__(src, dst)
        self.add_prefix_space = add_prefix_space

    def __call__(self, sample: dict) -> dict:
        tokens = sample[self.src]
        if isinstance(tokens, str):
            if self.add_prefix_space:
                sample[self.dst or self.src] = ' ' + tokens
        else:
            spaces = infer_space_after(tokens)
            if self.add_prefix_space:
                first_token = ' ' + tokens[0]
            else:
                first_token = tokens[0]
            appended_tokens = [first_token]
            for token, space in zip(tokens[1:], spaces):
                if space:
                    token = ' ' + token
                appended_tokens.append(token)
            sample[self.dst or self.src] = appended_tokens
        return sample
