from torchaudio.models import Conformer
from conformer.model_utils import get_model_params

model = Conformer(input_dim=144,
                  num_heads=4,
                  ffn_dim=144,
                  num_layers=16,
                  depthwise_conv_kernel_size=31)

print(f"{sum(get_model_params(model=model))} "
      f"{model}")