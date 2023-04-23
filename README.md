## Description of submitted files:

- `main.py`: A python script which runs a GUI (via Qt). The GUI can be used to manually tune a PID controller for the third-order system example or any other system, and display the step response of the closed-loop system and its performance specifications (i.e. rise time, settling time, steady state value etc.).

To run `main.py`, you will require Python 3.x and an environment containing the required dependencies. The dependencies can be installed using the `requirements.txt` file and `pip` by running the command below in your terminal:

```bash
$ pip install -r requirements.txt
```

After installing the dependencies, the `main.py` script can be executed using:

```bash
$ python main.py
```

A GUI window should pop up for testing the application.

- `crudepid.exe`: For conveniently running the application without having to set up any dependencies or Python, I bundled the application as an executable. This can be downloaded and opened on your PC to use the app.

## Sidenotes

1. The app might break as it was hurriedly built for the purpose of the assignment and has not been thoroughly tested.

2. A video has been included demonstrating. [Watch the video](https://www.loom.com/share/c9cf173d817c44638a9db3c6d614c4a2)
