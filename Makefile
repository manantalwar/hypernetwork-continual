# Set the name of your virtual environment
VENV_NAME = .venv

# Set the path to your requirements file
REQ_FILE = requirements.txt

# Define the environment target
environment:
    virtualenv $(VENV_NAME)
    source ./$(VENV_NAME)/bin/activate && \
        pip install -r $(REQ_FILE)

# Define the clean target
clean:
    rm -rf $(VENV_NAME)

.PHONY: environment clean
