import os
roots=[r'C:\Users\jhama', r'C:\Users\jhama\Downloads', r'C:\Users\jhama\AppData\Local\Temp', r'C:\Windows\Temp', r'C:\Temp']
for root in roots:
    if not os.path.exists(root):
        continue
    for dirpath, dirnames, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(('.xlsx','.xls','.csv')):
                p=os.path.join(dirpath,f)
                try:
                    sz=os.path.getsize(p)
                except Exception:
                    continue
                if 1800000 <= sz <= 2500000:
                    print(p, sz)
                    raise SystemExit
print('NOT_FOUND')
