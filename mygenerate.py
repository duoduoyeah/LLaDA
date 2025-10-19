import torch
import numpy as np
import torch.nn.functional as F
from generate import add_gumbel_noise, get_num_transfer_tokens 


@ torch.no_grad()
def my_generate(model, prompt, steps=128, gen_length=128, block_length=128,
            mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        mask_id: The toke id of [MASK] is 126336.
    '''
    completion_buffer = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    completion_buffer[:, :prompt.shape[1]] = prompt.clone()


    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (completion_buffer[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        
        for i in range(steps):
            mask_index = (completion_buffer == mask_id)

            logits = model(completion_buffer).logits

            logits_with_noise = add_gumbel_noise(logits)
            
            # x0 is the generated tokens at this step
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            # random remasking
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)

            # Set the confidence of tokens beyond the current block to -inf to exclude them from selection
            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            # Replace masked tokens in x0 with the corresponding tokens from completion_buffer 
            x0 = torch.where(mask_index, x0, completion_buffer) 
            
            confidence = torch.where(mask_index, x0_p, -np.inf)

            
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device) # init with false

            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True # mark selected tokens that will be unmasked
            completion_buffer[transfer_index] = x0[transfer_index]

    return completion_buffer


