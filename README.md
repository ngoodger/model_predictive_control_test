# Description 
Pytorch project for testing model based reinforcement learning.
Primarily trained on Google Kubernetes Engine on GPUs however it can
also be trained locally on cpu.
Pre-trained models included with the project.
# Local Installation
Run: `make install`
# Using Pre-trained models
## Run interactive Demo with Pretrained model
Generates frames using the model open-loop by feeding back the models own output into the model so error increases on time.
`make draw_test_policy`
## Generate test model frames
Generates frames using the model open-loop by feeding back the models own output into the model so error increases on time.
`make clean`
`make draw_test_policy`
## Generate test policy frames
Generates frames using the model and policy by feeding the simulator output into the policy and model.
`make clean`
`make draw_test_policy`
![](gif/model.gif)
![](gif/policy.gif)
