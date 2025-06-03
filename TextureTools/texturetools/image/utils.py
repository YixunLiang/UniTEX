import torch


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    B, C, H, W = masks.shape
    bounding_boxes = torch.zeros((B, 4), device=masks.device, dtype=torch.int64)
    masks = masks.sum(dim=1, keepdim=False)
    for index, mask in enumerate(masks):
        iy, ix = torch.where(mask > 0)
        if ix.numel() > 0 and iy.numel() > 0:
            bounding_boxes[index, :] = torch.stack([ix.min(), iy.min(), ix.max(), iy.max()], dim=0)
        else:
            bounding_boxes[index, :] = torch.full_like(bounding_boxes[index, :], fill_value=-1)
    return bounding_boxes


