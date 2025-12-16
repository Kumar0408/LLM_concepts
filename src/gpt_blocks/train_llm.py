import torch
from src.gpt_blocks.loss_calc import calc_loss_batch, calc_loss_loader
from src.gpt_blocks.utils import generate_and_print_sample

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """ Evaluate the model on training and validation data loaders """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss
 

def train_model(model,
                train_loader,
                val_loader,
                optimizer,
                device,
                num_epochs,
                eval_freq, 
                eval_iter,
                start_context,
                tokenizer):
    """ Train the LLM model 
    model: The language model to be trained
    train_loader: DataLoader for training data
    val_loader: DataLoader for validation data
    optimizer: Optimizer for model parameters
    device: Device to run the training on
    num_epochs: Number of epochs to train
    eval_freq: Frequency (in epochs) to evaluate the model on validation data
    eval_iter: Number of batches to use for evaluation
    start_context: Initial context string for evaluation
    tokenizer: Tokenizer to encode the context
    """

    train_losses,val_losses, track_tokens_seen = [], [], []

    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()

            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            global_step += 1
            tokens_seen += input_batch.numel()

        

        if global_step % eval_freq == 0:
            train_loss, val_loss = evaluate_model(
                model, train_loader, val_loader, device, eval_iter
                )
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            track_tokens_seen.append(tokens_seen)
            print(f"Ep {epoch+1} (Step {global_step:06d}), Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        
        generate_and_print_sample(model, tokenizer, device, start_context)
        
    return train_losses, val_losses, track_tokens_seen
    
        