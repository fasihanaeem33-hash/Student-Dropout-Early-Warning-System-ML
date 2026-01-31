import sklearn
import importlib
cct = importlib.import_module('sklearn.compose._column_transformer')
print('sklearn.__version__=' + sklearn.__version__)
print('has _RemainderColsList=' + str(hasattr(cct, '_RemainderColsList')))
print('\nmodule members:\n')
for n in sorted([n for n in dir(cct) if not n.startswith('__')]):
    print(n)
