import torch

if __name__ == "__main__":
    weights_file = '/Users/huaxinrui/.cache/huggingface/hub/models--lmsys--vicuna-7b-delta-v1.1/snapshots/b113200fb2c263258d21b4f2273304ccad43d75f/pytorch_model-00001-of-00002.bin'
    state_dict = torch.load(weights_file, map_location='cpu')

    for key in  state_dict:
        weight = state_dict[key]
        print("%s %s" %(key, weight[0]))
    # model.layers[30].mlp.up_proj.weight.data = torch.Tensor(weight)


