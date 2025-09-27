- Make a folder in downloads folder called 'virtual-envs' for example.
- 
run 
```
python -m venv finals_venv
```
- Copy It's path
- VSCode -> Ctrl+Shift+P -> Python: Select Interpreter -> Browse In files -> "finals_venv" -> Scripts -> python.exe
- check if you're in `cmd` otherwise shift to `cmd`
run 
```
pip install -r requirements.txt
```
- then in your terminal, in the virtual environment, in your current working directory
- 
run this:
```
python main.py
```