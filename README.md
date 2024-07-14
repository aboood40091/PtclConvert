# PtclConvert
Tool for editing PTCL files present in New Super Mario Bros. U and New Super Luigi U.  

## Usage
The tool can be ran in two modes:  

### 1. Specifying command line arguments:
To run this script with arguments, supply them in the following format:  
* PTCL to YAML: `python main.py 0 <PTCL file path> <YAML file path>`  
* YAML to PTCL: `python main.py 1 <YAML file path> <PTCL file path>`  

(`python` can be omitted if this script has been built into an EXE.)

### 2. Using the interactive mode:
Simply running the script, without specifying arguments, will run it in interactive mode, where the tool will walk you through the process.  

## Dependencies
This tool requires several Python packages, which are listed in the included `requirements.txt` file.  
You can install each separately, or install all using the command `pip install -r requirements.txt`.  
Additionally, `Cython` is not listed in `requirements.txt` as it's not required. However, installing it properly is encouraged, as the tool will be *very* slow without it.  

## Shader Compiler
This tool requires the `gshCompile` shader compiler **in some special cases only**. You can find `gshCompile` in:
* Cafe SDK: `system/bin/win32/gshCompile.exe` or `system/bin/win64/gshCompile.exe`  
* NintendoWare for Cafe: `Tool/EffectMaker/Converter/shader/gshCompile.exe`  

### Cases where the shader compiler is required
The shader compiler is only required when doing YAML to PTCL conversion, when any emitter uses new graphics-related features that were not used previously by other emitters and the required shader does not already exist in the `shaderBinPath` path specified in the project YAML. In that case, the tool tries to compile new shaders.
