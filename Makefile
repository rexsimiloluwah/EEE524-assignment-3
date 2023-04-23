.PHONY: build

build:
	pyinstaller --name crudepid --onefile --windowed --icon=pid.ico main.py