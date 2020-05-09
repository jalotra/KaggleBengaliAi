# Dispatches the Model 
import models

MODEL_DISPATCHER ={
    "resnet34" : models.CustomResnet34
}


if __name__ == "__main__":
    model = MODEL_DISPATCHER["resnet34"](training = True)
    model.verbose()