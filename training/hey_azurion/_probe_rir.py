from datasets import load_dataset
ds = load_dataset('davidscripka/MIT_environmental_impulse_responses', split='train', streaming=True)
ex = next(iter(ds))
print('keys:', list(ex.keys()))
for k, v in ex.items():
    if k == 'audio':
        print(f"  audio: sr={v['sampling_rate']} len={len(v['array'])}")
    else:
        s = str(v)
        print(f"  {k}: {s[:200]}")
