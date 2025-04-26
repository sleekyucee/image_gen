import os
import wandb

class Logger:
    def __init__(self, experiment_name, project="dcgan_project", log_dir="/users/adgs898/sharedscratch/dcgan_project/.wandb_offline"):
        """Initializes WandB logging in offline mode."""
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_DIR"] = log_dir  #set local log storage directory

        self.logger = wandb.init(project=project, name=experiment_name)

    def log(self, metrics):
        """Logs metrics like loss, scores, etc., during training."""
        self.logger.log(metrics)

    def save_model(self, model, filename="dcgan_generator.pth"):
        """Saves the trained model checkpoint."""
        wandb.save(filename)
        print(f"Model saved as {filename}")

    def finish(self):
        """Closes the WandB logging session."""
        self.logger.finish()

