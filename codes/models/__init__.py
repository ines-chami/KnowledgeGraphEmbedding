from .euc_models import EKGEModel
from .hyp_models import HKGEModel

# Do not forget to modify this line when you add a new model in the "forward" function
EUC_MODELS = ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE', 'ReflectionE', 'RotationE']
HYP_MODELS = ['TranslationH', 'ReflectionH', 'RotationH']
