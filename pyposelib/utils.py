def import_module_symbols(globals_dict, module, exclude=None):
    """Import all public symbols from a module into globals."""
    if exclude is None:
        exclude = set()
    
    symbols = []
    for name in dir(module):
        if not name.startswith('_') and name not in exclude:
            globals_dict[name] = getattr(module, name)
            symbols.append(name)
    return symbols 