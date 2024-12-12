# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('design.kv', '.'),  
        ('Assets/*', 'Assets'),
        ('binary/*', 'binary'),
    ],
    hiddenimports=[
        'sklearn', 
        'sklearn.naive_bayes',
        'sklearn.feature_extraction.text',  # Add this
        'sklearn.utils._typedefs',         # Add common missing utils
        'sklearn.utils._heap',             # Add common missing utils
        'scipy.sparse.csgraph._validation', # Add scipy dependencies
        'win32timezone'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
