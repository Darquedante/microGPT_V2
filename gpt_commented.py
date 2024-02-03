import math
import torch
import inspect
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from utils import top_k_top_p_filter

# Select device based on availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiInputSequential(nn.Sequential):
    """
    A sequential container that can process multiple inputs.
    Extends nn.Sequential to handle models with multiple inputs.
    """
    def forward(self, *inputs):
        """
        Forward pass that allows multiple inputs to be processed through the sequence of modules.
        
        Args:
            *inputs: Variable number of input tensors.
            
        Returns:
            The output of the last module in the sequence.
        """
        for module in self._modules.values():
            if isinstance(inputs, tuple):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class CausalSelfAttention(nn.Module):
    """
    Implements causal self-attention module typically used in transformer architectures.
    Causal attention ensures that the prediction for a token can only depend on previously known tokens.
    """
    def __init__(self, config):
        """
        Initializes the CausalSelfAttention module.
        
        Args:
            config: Configuration object with model hyperparameters.
        """
        super().__init__()
        # Ensure the embedding dimension is divisible by the number of attention heads
        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by the number of heads."
        
        # Attention layers
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Model configuration
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Check for availability of Flash Attention in the PyTorch version
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("Flash Attention requires PyTorch >= 2.0. Using normal attention.")
            # Lower triangular matrix for causal masking
            self.register_buffer("bias", torch.tril(torch.ones(
                config.max_length, config.max_length)).view(1, 1, config.max_length, config.max_length))

    def forward(self, x, attn_mask):
        """
        Forward pass for the CausalSelfAttention layer.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, embedding_dimension).
            attn_mask: Attention mask.
            
        Returns:
            The output tensor after applying causal self-attention.
        """
        B, T, C = x.shape

        # Split the linear output into query, key, and value components
        query, key, value = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape and transpose for multi-head attention processing
        query = query.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        key = key.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        value = value.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Apply causal mask to ensure attention is only applied to previous tokens
        if attn_mask is not None:
            attn_mask = torch.ones(B, T, T).tril(diagonal=0).to(device) * attn_mask.unsqueeze(1)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, -float('inf')).unsqueeze(1).to(query.dtype)
            attn_mask = attn_mask.masked_fill(attn_mask == 1, 0)
        
        # Use Flash Attention if available
        if self.flash:
            out = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # Compute scaled dot-product attention manually
            if attn_mask is None:
                attn_mask = torch.ones(B, T, T).tril(diagonal=0).unsqueeze(1)  # (B, 1, T, T)
            qK = query @ key.transpose(-2, -1) * (1.0 / math.sqrt(query.shape[-1]))
            qK = qK + attn_mask
            qK = F.softmax(qK, dim=-1)
            qK = self.attn_dropout(qK)
            qK = qK.masked_fill(torch.isnan(qK), 0)
            out = qK @ value

        # Combine heads and project output
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.c_proj(out))
        return out
class FeedForward(nn.Module):
    """
    Implements a feed-forward layer as defined in transformer architectures.
    """
    def __init__(self, config):
        """
        Initializes the feed-forward layer.
        
        Args:
            config: Configuration object with model hyperparameters.
        """
        super().__init__()
        # Define the feed-forward network structure
        self.out = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        """
        Forward pass for the FeedForward layer.
        
        Args:
            x: Input tensor.
            
        Returns:
            The output tensor after applying the feed-forward network.
        """
        return self.out(x)

class Block(nn.Module):
    """
    Defines a single transformer block with self-attention and feed-forward layers.
    """
    def __init__(self, config):
        """
        Initializes the Block module.
        
        Args:
            config: Configuration object with model hyperparameters.
        """
        super().__init__()
        self.sa_head = CausalSelfAttention(config)  # Self-attention head
        self.feed_fwd = FeedForward(config)  # Feed-forward network
        self.ln1 = nn.LayerNorm(config.n_embd)  # Pre-normalization layer 1
        self.ln2 = nn.LayerNorm(config.n_embd)  # Pre-normalization layer 2

    def forward(self, x, attn_mask):
        """
        Forward pass for the Block module.
        
        Args:
            x: Input tensor.
            attn_mask: Attention mask.
            
        Returns:
            The output tensor after processing through the block.
        """
        # Apply self-attention and add the input (residual connection)
        x = x + self.sa_head(self.ln1(x), attn_mask)
        # Apply feed-forward network and add the input (residual connection)
        x = x + self.feed_fwd(self.ln2(x))
        return x, attn_mask

class GPT(nn.Module):
    """
    Defines the GPT model architecture with embedding, positional encoding, transformer blocks, and output projection.
    """
    def __init__(self, config):
        """
        Initializes the GPT model.
        
        Args:
            config: Configuration object with model hyperparameters.
        """
        super().__init__()
        self.embedding_table = nn.Embedding(config.vocab_size, config.n_embd)  # Token embeddings
        self.positional_embedding = nn.Embedding(config.max_length, config.n_embd)  # Positional embeddings
        self.blocks = MultiInputSequential(*[Block(config) for _ in range(config.n_layer)])  # Transformer blocks
        self.ln_final = nn.LayerNorm(config.n_embd)  # Final normalization layer
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # Output projection layer
        self.max_length = config.max_length
        self.apply(self._init_weights)  # Apply weight initialization

    def _init_weights(self, module):
        """
        Initializes the weights of the model.
        
        Args:
            module: A module in the model.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, idx, attention_mask=None, target=None, pad_token=None):
        """
        Forward pass for the GPT model.
        
        Args:
            idx: Input indices tensor.
            attention_mask: Optional attention mask (not used in this implementation).
            target: Optional target indices for calculating loss.
            pad_token: Optional pad token index to ignore in loss calculation.
            
        Returns:
            Tuple of logits and optional loss if target is provided.
        """
        B, T = idx.shape  # Batch size and sequence length

        # Token and positional embeddings
        tok_emb = self.embedding_table(idx)  # Token embeddings (B, T, C)
        pos_emb = self.positional_embedding(torch.arange(T, device=device))  # Positional embeddings for each token
        x = tok_emb + pos_emb  # Combine token and positional embeddings

        # Process through transformer blocks
        x, _ = self.blocks(x, attention_mask)

        # Apply final layer normalization and project to vocabulary size
        logits = self.lm_head(self.ln_final(x))  # Logits (B, T, vocab_size)

        # Calculate loss if targets are provided
        if target is None:
            loss = None
        else:
            # Reshape logits for loss calculation
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # Flatten output for cross-entropy
            target = target.view(-1)  # Flatten target indices
            if pad_token is not None:
                loss = F.cross_entropy(logits, target, ignore_index=pad_token)  # Ignore pad token in loss
            else:
                loss = F.cross_entropy(logits, target)  # Calculate cross-entropy loss
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configures the optimizer for the GPT model.
        
        Args:
            weight_decay: Weight decay parameter.
            learning_rate: Learning rate for the optimizer.
            betas: Betas for Adam optimizer.
            device_type: Type of device ('cuda' or 'cpu') to determine if fused Adam should be used.
            
        Returns:
            Configured optimizer.
        """
        # Separate parameters for weight decay
        decay_params = [p for n, p in self.named_parameters() if p.dim() >= 2]
        nodecay_params = [p for n, p in self.named_parameters() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # Check for fused Adam optimizer availability
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        # Create optimizer
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"Using fused AdamW: {use_fused}")

        return optimizer
    def generate(self, idx: Tensor, max_tokens_generate: int, temperature: float, top_k: int, top_p: float):
        """
        Generate text using the trained GPT model.
        
        Args:
            idx: Input tensor of shape (B, T) where B is batch size and T is sequence length.
            max_tokens_generate: Maximum number of tokens to generate.
            temperature: Temperature parameter for controlling the randomness of predictions by scaling the logits.
            top_k: Top-K filtering parameter to focus on the top k probabilities and set others to -infinity.
            top_p: Top-p (nucleus) filtering parameter to retain the cumulative probability distribution up to p.
        
        Returns:
            Tensor of generated indices with shape (T,), where T includes the original and generated tokens.
        """
        for _ in range(max_tokens_generate):
            # Crop input to handle the maximum length limitation
            idx_condition = idx[:, -self.max_length:]
            logits, _ = self.forward(idx_condition)  # Get logits for the last set of tokens

            # Apply temperature scaling and filter logits
            logits = logits[:, -1, :] / temperature  # Scale logits by temperature
            logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)  # Apply top-k and top-p filtering

            # Convert logits to probabilities and sample the next token
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # Sample from the probabilities

            # Break if the model outputs the termination token (e.g., <EOS> token with index 0)
            if idx_next.item() == 0:
                break

            # Concatenate the newly generated token to the sequence
            idx = torch.cat((idx, idx_next), dim=-1)

        # Return the generated index sequence, excluding the batch dimension
        return idx.view(idx.shape[1], )
