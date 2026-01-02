import sys
try:
    import torch
except Exception as e:
    print('ImportError:', e)
    print('sys.executable:', sys.executable)
    sys.exit(0)

print('sys.executable:', sys.executable)
print('torch.__version__:', getattr(torch, '__version__', None))
print('torch.version.cuda:', getattr(torch.version, 'cuda', None))
print('torch.cuda.is_available():', torch.cuda.is_available())
print('torch.cuda.device_count():', torch.cuda.device_count())
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    try:
        print('device_name(0):', torch.cuda.get_device_name(0))
    except Exception as e:
        print('get_device_name error:', e)
