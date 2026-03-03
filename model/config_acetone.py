from transformers import AutoConfig, Qwen2_5_VLConfig


class AceToneConfig(Qwen2_5_VLConfig):
    model_type = "acetone"

    def __init__(
        self,
        mm_vocab_size: int = 256,  # size of your custom vocab
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mm_vocab_size = mm_vocab_size


# Register so AutoConfig can load it by name
AutoConfig.register("acetone", AceToneConfig)
