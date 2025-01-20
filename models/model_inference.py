import os
import torch
import wandb
from monai.networks.nets import SegResNet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    AsDiscrete,
    Activations,
)


class SingleFileInference:
    def __init__(self, device="cuda"):
        """
        Initialize the inference class.
        Args:
            wandb_artifact (str): Path to the wandb artifact
            device (str): Device to use for inference ("cuda" or "cpu")
        """
        self.device = torch.device(device)
        self.model = self._model_initialize()
        self.transforms = self._setup_transforms()
        self.post_transforms = self._post_transforms()

    def _model_initialize(self):
        """Initialize the SegResNet model"""
        try:
            model = SegResNet(
                blocks_down=[1, 2, 2, 4],
                blocks_up=[1, 1, 1],
                init_filters=16,
                in_channels=4,
                out_channels=3,
                dropout_prob=0.2,
            ).to(self.device)
            return model
        except Exception as e:
            # raise Exception(f"Cuda support is {torch.cuda.is_available()}")
            raise Exception(f"Failed to initialize model: {e}")

    def load_model(self, wandb_artifact):
        """Load model weights from wandb artifact"""
        try:
            model_artifact = wandb.use_artifact(wandb_artifact, type="model")
            model_dir = model_artifact.download()
            model_path = os.path.join(model_dir, "model.pth")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")

            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
        except wandb.errors.CommError as e:
            raise Exception(f"WandB communication error: {e}")
        except FileNotFoundError as e:
            raise Exception(f"Model file error: {e}")
        except RuntimeError as e:
            raise Exception(f"Model loading error: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error loading model: {e}")

    def _setup_transforms(self):
        """Setup input transforms"""
        try:
            transforms = Compose(
                [
                    LoadImaged(keys=["image"]),
                    EnsureChannelFirstd(keys=["image"]),
                    EnsureTyped(keys=["image"]),
                    Orientationd(keys=["image"], axcodes="RAS"),
                    Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
                    NormalizeIntensityd(
                        keys=["image"], nonzero=True, channel_wise=True
                    ),
                ]
            )
            return transforms
        except Exception as e:
            raise Exception(f"Failed to setup transforms: {e}")

    def _post_transforms(self):
        """Setup post-processing transforms"""
        try:
            transforms = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
            return transforms
        except Exception as e:
            raise Exception(f"Failed to setup post-transforms: {e}")

    def inference(self, input_tensor):
        """
        Custom inference function matching training setup
        Args:
            input_tensor (torch.Tensor): Input tensor to process
        Returns:
            torch.Tensor: Model predictions
        """

        def _compute(input_tensor):
            return sliding_window_inference(
                inputs=input_tensor,
                roi_size=(240, 240, 160),
                sw_batch_size=1,
                predictor=self.model,
                overlap=0.5,
            )

        try:
            with torch.amp.autocast("cuda"):
                return _compute(input_tensor)
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                raise Exception(
                    "GPU out of memory. Try reducing batch size or image size."
                )
            raise Exception(f"Inference error: {e}")

    def predict(self, input_path):
        """
        Run prediction on input file
        Args:
            input_path (str): Path to input .nii.gz file
        Returns:
            torch.Tensor: Predictions tensor
        """
        # Input validation
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if not input_path.endswith(".nii.gz"):
            raise ValueError("Input file must be a .nii.gz file")

        try:
            # Clear GPU memory before starting
            torch.cuda.empty_cache()

            print("Input path:", input_path)
            # Load and transform input
            data = self.transforms({"image": input_path})

            with torch.no_grad():
                # Move input to device and add batch dimension
                input_tensor = data["image"].unsqueeze(0).to(self.device)

                # Run inference
                prediction = self.inference(input_tensor)
                prediction = self.post_transforms(prediction[0])

                # Move result to CPU
                result = prediction.cpu()

                # Clean up GPU memory
                del input_tensor, prediction
                torch.cuda.empty_cache()

                return result

        except Exception as e:
            torch.cuda.empty_cache()  # Clean up on error
            raise Exception(f"Prediction failed: {e}")

    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            torch.cuda.empty_cache()
        except:
            pass
