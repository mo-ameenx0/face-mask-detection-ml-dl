TOKEN = ""

# STATES
STATES = {"INITIAL": "initial",
          "LOADING": "loading",
          "PREDICT": "predict",
          "RESULTS": "result",
          "HELP": "help",
          "ERROR": "error",
          "NO_IMAGE": "no_image"}

# MESSAGES
WELCOME_MESSAGE = "This bot uses two different AI techniques:" \
                  "\n1. Deep Learning" \
                  "\n2. Machine Learning" \
                  "\nTo predict a person is wearing a mask or not." \
                  "\n" \
                  "\nEnter your image to start predicting!"

# Help Messages
HELP_MESSAGES = {"initial": "HELP: Enter '/start' to initiate the bot",
                 "loading": "HELP: Load an image to predict"}

# Error Messages
ERROR_MESSAGES = {"error": "ERROR: Unknown command, enter 'help' for more information",
                  "no_image": "ERROR: Load your image"}

# Models Paths
TORCH_PATH = "./models/pytorch_model/model_epoch_10.pt"
SKLEARN_PATH = "./models/sklearn_model/model.joblib"
