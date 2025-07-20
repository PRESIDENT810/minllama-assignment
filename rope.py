from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    _, seqlen, n_heads, head_dim = query.shape # (b, s, n_heads, head_dim)
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
    half_dim = head_dim // 2
    device = query.device
    # todo
    #
    # Please refer to slide 22 in https://phontron.com/class/anlp2024/assets/slides/anlp-05-transformers.pdf
    # and Section 3 in https://arxiv.org/abs/2104.09864.

    # reshape xq and xk to match the complex representation
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1) # (b, s, n_heads, half_dim), (b, s, n_heads, half_dim)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1) # (b, s, n_heads, half_dim), (b, s, n_heads, half_dim)
    # This separates each query/key vector into its odd and even indices (assuming *one-indexing*).
    # query_real contains q_1, q_3, q_5, ... and query_imag contains q_2, q_4, q_6, ...

    # First, compute the trigonometric values in the second and fourth columns in
    # slide 22 (linked above).

    # Then, combine these trigonometric values with the tensors query_real, query_imag,
    # key_real, and key_imag.
    assert max_seq_len >= seqlen, f"max_seq_len {max_seq_len} must be >= seqlen {seqlen}"
    thetas = torch.arange(0, half_dim, device=device, dtype=query.dtype) / head_dim # (half_dim,)
    thetas = torch.pow(10000, thetas * -2)  # (half_dim,)
    seq_indices = torch.arange(seqlen, device=device, dtype=query.dtype) # (seqlen,)
    freqs = seq_indices[:, None] * thetas[None, :]  # (seqlen, half_dim)
    consine = torch.cos(freqs)  # (seqlen, half_dim)
    sine = torch.sin(freqs)  # (seqlen, half_dim)
    consine = consine[None, :, None, :] # (1, seqlen, 1, half_dim)
    sine = sine[None, :, None, :] # (1, seqlen, 1, half_dim)

    def apply_rotary(x_real: torch.Tensor, x_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to the real and imaginary parts of the input tensor.
        """
        real = consine * x_real - sine * x_imag  # (b, s, n_heads, half_dim)
        imag = consine * x_imag + sine * x_real # (b, s, n_heads, half_dim)
        return torch.stack((real, imag), dim=-1).flatten(-2)  # (b, s, n_heads, half_dim, 2)

    query_out = apply_rotary(query_real, query_imag)  # (b, s, n_heads, half_dim, 2)
    key_out = apply_rotary(key_real, key_imag)  # (b, s, n_heads, half_dim, 2)

    # Return the rotary position embeddings for the query and key tensors
    return query_out, key_out