import awkward
import torch

def normalize_x(dataset):
    x=dataset["data"][:,1:2,:]
    x_std=awkward.std(x)
    x_mean=awkward.mean(x)
    norm_x=(x-x_mean)/x_std
    return norm_x

def normalize_y(dataset):
    y=dataset["data"][:,2:3,:]
    y_std=awkward.std(y)
    y_mean=awkward.mean(y)
    norm_y=(y-y_mean)/y_std
    return norm_y

def normalize_time(dataset):
    times = dataset["data"][:, 0:1, :]  # important to index the time dimension with 0:1 to keep this dimension (n_events, 1, n_hits)
                                               # with [:,0,:] we would get a 2D array of shape (n_events, n_hits)
    # make times relative per event
    t_rel = times- awkward.min(times,axis=2)
    #print(t_rel[0])
    # global normalization
    mean_t = awkward.mean(awkward.flatten(t_rel))
    std_t = awkward.std(awkward.flatten(t_rel))

    norm_times = (t_rel - mean_t) / std_t
    return norm_times


def denormalize(pred, target_x_mean, target_x_std, target_y_mean, target_y_std):
    pred[:, 0] = (
        (pred[:, 0] * target_x_std)
        + target_x_mean
    )

    pred[:, 1] = (
        (pred[:, 1] * target_y_std)
        + target_y_mean
    )

    return pred


def collate_fn_transformer(batch: list[dict]) -> tuple[list[list | torch.Tensor], torch.Tensor]:
    """
    Custom function that defines how batches are formed.

    To process the batch items that each have a different number of hits, it is efficient
    to first concatenate all the data into a single tensor and save the lengths of each
    individual event to be able to split the data again later.

    # F: input_dim, number of features (time, x, y)
    # N: number of hits (different for each event)
    # B: batch size

    The resulting 2D tensor has the shape (B x N, F) where B is the batch size, N is the total number of hits of all events
    in the batch, and F is the number of features (time, x, y).


    Parameters
    ----------
    batch : list
        A list of dictionaries containing the data and labels for each graph.
        The data is available in the "data" key and the labels are in the "xpos" and "ypos" keys.
    Returns
    -------
    packed_data : Batch
        A batch of graph data objects.
    labels : torch.Tensor
        A tensor containing the labels for each graph.
    """
    data_list: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    lengths: list[int] = []
    
    for b in batch:
        # this is a loop over each event within the batch
        # b["data"] is the first entry in the batch with dimensions (n_features, n_hits)
        # where the features are (time, x, y)
        tensor_data = torch.from_numpy(b["data"].to_numpy()).T
        # the original data is in double precision (float64), for our case single precision is sufficient
        # we let's convert to single precision (float32) to save memory and computation time
        tensor_data = tensor_data.to(dtype=torch.float32)

        lengths.append(tensor_data.shape[0])

        data_list.append(tensor_data)

        # also the labels need to be packaged as pytorch tensors
        labels.append(torch.Tensor([b["xpos"], b["ypos"]]).unsqueeze(0))

    ## return a list [data_list, lengths]
    return [
        torch.cat(data_list),  # (B, N, F)  -> (BxN, F) where B is the batch size, N is the number of hits, and F is the number of features (time, x, y)
        lengths,
    ], torch.cat(labels, dim=0)

def normalize_data(og_dataset):
    dataset = awkward.copy(og_dataset)
    norm_times=normalize_time(dataset)
    norm_x=normalize_x(dataset)
    norm_y=normalize_y(dataset)
# Concatenate the normalized data back together
    dataset["data"] = awkward.concatenate([norm_times, norm_x, norm_y], axis=1)
    # Normalize labels (this can be done in-place), e.g. by
    x_mean=awkward.mean(dataset["xpos"])
    y_mean=awkward.mean(dataset["ypos"])
    x_std=awkward.std(dataset["xpos"])
    y_std=awkward.std(dataset["ypos"])
    dataset["xpos"] = (dataset["xpos"] - x_mean) / x_std
    dataset["ypos"] = (dataset["ypos"] - y_mean) / y_std
    return dataset
